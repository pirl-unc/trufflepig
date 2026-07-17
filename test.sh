#!/usr/bin/env bash
# Run the trufflepig test suite with a memory- and CPU-aware pytest-xdist
# worker count.
#
# Background: each xdist worker loads its own DataFrame state, and running
# multiple suites concurrently across the laptop turned that into a workload
# that OOMed a 32 GB Mac. Direct/IDE pytest now defaults to serial, while this
# wrapper enables only the xdist concurrency that current RAM can support:
#
#   - CPU reserve: leave 1-2 cores idle so the box stays responsive
#     (1 core -> 1; 2-3 cores -> cores-1; 4+ cores -> cores-2).
#   - Memory budget: reserve RAM for the OS/apps, then cap at the remaining
#     RAM / PER_WORKER_GB.
#   - Cross-invocation lock: refuse to start a second local xdist pool while
#     one from this checkout is already active.  Two independently "safe"
#     pools can otherwise race on the same pre-launch free-RAM reading.
#
# A root conftest independently rejects worker counts above this wrapper's
# approval and takes the same lock for direct/IDE launches.
#
# Tunables (env vars):
#   PER_WORKER_GB    peak per-worker memory budget in GB (default/minimum: 12)
#   RAM_RESERVE_GB   RAM kept for the OS and other apps (default/minimum: 8)
#   TEST_SH_MIN      floor on workers (default: 1)
#   TEST_SH_MAX      hard ceiling on workers (default: unset)
#   TEST_SH_ALLOW_CONCURRENT
#                    set to 1 only when the caller coordinates memory itself
#   TEST_SH_LOCK_DIR override the per-user cross-invocation lock directory

set -eo pipefail

PER_WORKER_GB="${PER_WORKER_GB:-12}"
RAM_RESERVE_GB="${RAM_RESERVE_GB:-8}"
TEST_SH_MIN="${TEST_SH_MIN:-1}"
TEST_SH_MAX="${TEST_SH_MAX:-0}"
TEST_SH_ALLOW_CONCURRENT="${TEST_SH_ALLOW_CONCURRENT:-0}"
TEST_SH_LOCK_DIR="${TEST_SH_LOCK_DIR:-${TMPDIR:-/tmp}/trufflepig-test-${UID:-user}.lock}"

log() { printf '[test.sh] %s\n' "$*" >&2; }

[[ "$PER_WORKER_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    log "PER_WORKER_GB must be numeric"; exit 64;
}
[[ "$RAM_RESERVE_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
    log "RAM_RESERVE_GB must be numeric"; exit 64;
}
[[ "$TEST_SH_MIN" =~ ^[0-9]+$ ]] || { log "TEST_SH_MIN must be an integer"; exit 64; }
[[ "$TEST_SH_MAX" =~ ^[0-9]+$ ]] || { log "TEST_SH_MAX must be an integer"; exit 64; }
[[ "$TEST_SH_ALLOW_CONCURRENT" =~ ^[01]$ ]] || {
    log "TEST_SH_ALLOW_CONCURRENT must be 0 or 1"; exit 64;
}
awk -v value="$PER_WORKER_GB" 'BEGIN { exit !(value >= 12) }' || {
    log "refusing PER_WORKER_GB=${PER_WORKER_GB}: measured peak requires a minimum of 12 GB"
    exit 64
}
awk -v value="$RAM_RESERVE_GB" 'BEGIN { exit !(value >= 8) }' || {
    log "refusing RAM_RESERVE_GB=${RAM_RESERVE_GB}: safety reserve must be at least 8 GB"
    exit 64
}
if (( TEST_SH_MIN < 1 )); then
    log "TEST_SH_MIN must be at least 1"
    exit 64
fi

# Serialise local test runs.  This is deliberately a per-user lock rather than
# a lock inside the checkout: multiple Codex/terminal sessions can launch the
# same memory-heavy suite from different worktrees.  The July 2026 watchdog
# panic had two overlapping xdist pools (19 workers total), each heavy worker
# at 6.3-7.4 GB RSS; a subsequent serial full run reached ~9.6 GB.
# Free-RAM sizing alone cannot prevent that race.
LOCK_HELD=0
release_lock() {
    if (( LOCK_HELD == 0 )); then return; fi
    local owner=""
    if [[ -r "$TEST_SH_LOCK_DIR/pid" ]]; then
        owner=$(<"$TEST_SH_LOCK_DIR/pid")
    fi
    if [[ "$owner" == "$$" ]]; then
        rm -f "$TEST_SH_LOCK_DIR/pid"
        rmdir "$TEST_SH_LOCK_DIR" 2>/dev/null || true
    fi
    LOCK_HELD=0
}

acquire_lock() {
    if [[ "$TEST_SH_ALLOW_CONCURRENT" == "1" ]]; then
        log "concurrency lock disabled by TEST_SH_ALLOW_CONCURRENT=1"
        export TRUFFLEPIG_TEST_LOCK_BYPASS=1
        return
    fi

    if ! mkdir "$TEST_SH_LOCK_DIR" 2>/dev/null; then
        local owner=""
        if [[ -r "$TEST_SH_LOCK_DIR/pid" ]]; then
            owner=$(<"$TEST_SH_LOCK_DIR/pid")
        fi
        if [[ "$owner" =~ ^[0-9]+$ ]]; then
            log "refusing concurrent run: existing test lock records owner pid=${owner} at $TEST_SH_LOCK_DIR"
        else
            log "refusing test lock with absent or incomplete owner metadata: $TEST_SH_LOCK_DIR"
        fi
        # Never reclaim an existing lock automatically.  An ownerless directory
        # may belong to a launcher between mkdir and PID publication, and a
        # stale-lock check/delete sequence can race with a new owner (ABA).
        log "wait for the active run, or remove $TEST_SH_LOCK_DIR only after confirming no pytest/test.sh process is active"
        exit 75
    fi

    if ! printf '%s\n' "$$" > "$TEST_SH_LOCK_DIR/pid"; then
        rmdir "$TEST_SH_LOCK_DIR" 2>/dev/null || true
        log "cannot publish test lock owner in $TEST_SH_LOCK_DIR"
        exit 75
    fi
    LOCK_HELD=1
    export TRUFFLEPIG_TEST_LOCK_OWNER="$$"
    trap release_lock EXIT INT TERM HUP
}

export TRUFFLEPIG_TEST_LOCK_DIR="$TEST_SH_LOCK_DIR"
unset TRUFFLEPIG_TEST_LOCK_OWNER
unset TRUFFLEPIG_TEST_LOCK_BYPASS
unset TRUFFLEPIG_XDIST_APPROVED_WORKERS
acquire_lock

# Pin BLAS / OpenMP threading to 1 per pytest-xdist worker. Without
# this, every worker silently parallelises numpy/sklearn ops across
# ALL cores — and with N xdist workers each using K BLAS threads we
# get N×K concurrent compute threads on N cores. On a 10-core Mac
# (Apple Accelerate BLAS) that's 80–100 threads competing for 10
# cores, which manifests as a frozen machine even though it isn't
# technically a fork bomb. xdist already provides the test-level
# parallelism we want; intra-test BLAS parallelism on top is pure
# oversubscription.
#
# Override at the call site if a specific test legitimately needs
# numpy/sklearn multi-threading (rare):
#   OMP_NUM_THREADS=4 ./test.sh tests/test_perf_regression.py
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"  # macOS Accelerate
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
# sklearn / joblib spawns its own loky pool per estimator; cap it.
export LOKY_MAX_CPU_COUNT="${LOKY_MAX_CPU_COUNT:-1}"

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
    local page_size=""
    # Managed shells may permit vm_stat but deny sysctl. getconf is portable
    # and works in that environment; retain sysctl only as a fallback.
    page_size=$(getconf PAGESIZE 2>/dev/null || true)
    if [[ -z "$page_size" ]]; then
        page_size=$(sysctl -n hw.pagesize 2>/dev/null || true)
    fi
    [[ "$page_size" =~ ^[0-9]+$ ]] || return 1
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
    MEM_CAP=$(awk -v b="$avail" -v g="$PER_WORKER_GB" -v r="$RAM_RESERVE_GB" 'BEGIN {
        usable = b / 1024^3 - r
        n = int(usable / g)
        if (n < 1) n = 1
        print n
    }')
    AVAIL_GB=$(awk -v b="$avail" 'BEGIN { printf "%.1f", b / 1024^3 }')
    mem_note="ram_available=${AVAIL_GB}GB reserve=${RAM_RESERVE_GB}GB mem_cap=${MEM_CAP}"
else
    # A failed probe must fail safe. Falling back to CPU count recreated the
    # very xdist explosion this wrapper exists to prevent when sysctl was
    # blocked inside a managed shell.
    MEM_CAP=1
    mem_note="ram_available=? (probe unavailable) mem_cap=1-safe-fallback"
fi

if (( CPU_CAP < MEM_CAP )); then WORKERS=$CPU_CAP; else WORKERS=$MEM_CAP; fi
if (( TEST_SH_MAX > 0 && WORKERS > TEST_SH_MAX )); then WORKERS=$TEST_SH_MAX; fi
if (( TEST_SH_MIN > WORKERS )); then
    log "refusing TEST_SH_MIN=${TEST_SH_MIN}: current CPU/RAM safety cap is ${WORKERS}"
    exit 64
fi

export TRUFFLEPIG_XDIST_APPROVED_WORKERS="$WORKERS"

log "platform=${OS} cpus=${CPUS} cpu_cap=${CPU_CAP} ${mem_note} per_worker=${PER_WORKER_GB}GB"
log "workers=${WORKERS} → pytest -n ${WORKERS} tests $*"

# Keep this shell alive so its EXIT trap releases the cross-invocation lock.
pytest -n "$WORKERS" tests "$@"
