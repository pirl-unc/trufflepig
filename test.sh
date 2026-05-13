#!/usr/bin/env bash
# Run the trufflepig test suite with a memory-aware pytest-xdist worker
# count. ``pyproject.toml`` defaults to ``addopts = "-n auto"`` once the
# in-progress migration PRs land, which spawns one worker per core;
# each worker loads its own DataFrame state and can collectively OOM
# a 32 GB Mac when other pytest suites are running concurrently. The
# override here picks the smaller of (cores, available_RAM /
# per_worker_gb) and passes it as the final ``-n`` (xdist resolves
# multiple ``-n`` flags to the last value).

set -e

# Per-worker memory budget. A soft heuristic, not a hard cap.
readonly PER_WORKER_GB=1.5

# macOS available-RAM heuristic: free + inactive + speculative pages
# are reclaimable on demand, so they count as headroom.
macos_available_bytes() {
    local page_size pages
    page_size=$(sysctl -n hw.pagesize)
    pages=$(vm_stat | awk '
        /Pages free/        { gsub(/\./, "", $3); free     = $3 }
        /Pages inactive/    { gsub(/\./, "", $3); inactive = $3 }
        /Pages speculative/ { gsub(/\./, "", $3); spec     = $3 }
        END                 { print free + inactive + spec }
    ')
    echo $(( pages * page_size ))
}

# Pick a pytest -n value that respects both CPU and available RAM.
pytest_workers() {
    local cpus avail_bytes
    cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu)
    if [[ "$(uname)" == "Darwin" ]]; then
        avail_bytes=$(macos_available_bytes)
    else
        avail_bytes=$(awk '/MemAvailable/ { print $2 * 1024 }' /proc/meminfo)
    fi
    awk -v cpus="$cpus" -v bytes="$avail_bytes" -v budget="$PER_WORKER_GB" '
        BEGIN {
            by_memory = int(bytes / 1024^3 / budget)
            n = (cpus < by_memory ? cpus : by_memory)
            print (n < 1 ? 1 : n)
        }
    '
}

workers=$(pytest_workers)
echo "Running pytest with -n ${workers} (per-worker budget ≈ ${PER_WORKER_GB} GB)"
exec pytest -n "$workers" tests "$@"
