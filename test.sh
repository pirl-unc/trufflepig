#!/usr/bin/env bash
# Run the trufflepig test suite with a memory- and CPU-aware pytest-xdist
# worker count.
#
# Background: `-n auto` in pyproject.toml spawns one worker per logical
# core. Each worker loads its own DataFrame state, and running several
# repos' suites concurrently across the laptop has turned that into a
# fork bomb that OOMs a 32 GB Mac. We cap the worker count two ways:
#
#   - CPU reserve: leave 1-2 cores idle so the box stays responsive
#     (1 core -> 1; 2-3 cores -> cores-1; 4+ cores -> cores-2).
#   - Memory budget: cap at available_RAM / PER_WORKER_GB.
#
# xdist resolves multiple -n flags to the last value, so this overrides
# any `-n auto` baked into config.
#
# Tunables (env vars):
#   PER_WORKER_GB    per-worker memory budget in GB (default: 1.5)
#   TEST_SH_MIN      floor on workers (default: 1)
#   TEST_SH_MAX      hard ceiling on workers (default: unset)

set -eo pipefail

PER_WORKER_GB="${PER_WORKER_GB:-1.5}"
TEST_SH_MIN="${TEST_SH_MIN:-1}"
TEST_SH_MAX="${TEST_SH_MAX:-0}"

log() { printf '[test.sh] %s\n' "$*" >&2; }

case "$(uname -s)" in
    Darwin) OS=macos ;;
    Linux)  OS=linux ;;
    *)      OS=unknown ;;
esac

cpu_count() {
    local n=""
    if command -v getconf >/dev/null 2>&1; then
        n=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
    fi
    if [[ -z "$n" && "$OS" == "macos" ]]; then
        n=$(sysctl -n hw.logicalcpu 2>/dev/null || true)
    fi
    if [[ -z "$n" && -r /proc/cpuinfo ]]; then
        n=$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || true)
    fi
    if [[ -z "$n" || "$n" -lt 1 ]]; then n=1; fi
    echo "$n"
}

# Reserve cores so the system stays usable while the suite runs.
cpu_cap() {
    local c="$1"
    if   (( c <= 1 )); then echo 1
    elif (( c <= 3 )); then echo $(( c - 1 ))
    else                    echo $(( c - 2 ))
    fi
}

# free + inactive + speculative pages are reclaimable on demand and
# count as headroom for our purposes.
mac_available_bytes() {
    local page_size
    page_size=$(sysctl -n hw.pagesize 2>/dev/null) || return 1
    vm_stat 2>/dev/null | awk -v ps="$page_size" '
        /Pages free/        { gsub(/\./, "", $3); free     = $3 }
        /Pages inactive/    { gsub(/\./, "", $3); inactive = $3 }
        /Pages speculative/ { gsub(/\./, "", $3); spec     = $3 }
        END { print (free + inactive + spec) * ps }
    '
}

linux_available_bytes() {
    [[ -r /proc/meminfo ]] || return 1
    awk '
        /^MemAvailable:/ { print $2 * 1024; found=1; exit }
        END              { if (!found) exit 1 }
    ' /proc/meminfo
}

available_bytes() {
    case "$OS" in
        macos) mac_available_bytes ;;
        linux) linux_available_bytes ;;
        *)     return 1 ;;
    esac
}

CPUS=$(cpu_count)
CPU_CAP=$(cpu_cap "$CPUS")

avail=""
if avail=$(available_bytes 2>/dev/null) && [[ -n "$avail" ]]; then
    MEM_CAP=$(awk -v b="$avail" -v g="$PER_WORKER_GB" 'BEGIN {
        n = int(b / 1024^3 / g)
        if (n < 1) n = 1
        print n
    }')
    AVAIL_GB=$(awk -v b="$avail" 'BEGIN { printf "%.1f", b / 1024^3 }')
    mem_note="ram_free=${AVAIL_GB}GB mem_cap=${MEM_CAP}"
else
    MEM_CAP=$CPU_CAP
    mem_note="ram_free=? (probe unavailable) mem_cap=cpu_cap"
fi

if (( CPU_CAP < MEM_CAP )); then WORKERS=$CPU_CAP; else WORKERS=$MEM_CAP; fi
if (( TEST_SH_MAX > 0 && WORKERS > TEST_SH_MAX )); then WORKERS=$TEST_SH_MAX; fi
if (( WORKERS < TEST_SH_MIN )); then WORKERS=$TEST_SH_MIN; fi

log "platform=${OS} cpus=${CPUS} cpu_cap=${CPU_CAP} ${mem_note} per_worker=${PER_WORKER_GB}GB"
log "workers=${WORKERS} → exec pytest -n ${WORKERS} tests $*"

exec pytest -n "$WORKERS" tests "$@"
