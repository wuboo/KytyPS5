#!/bin/sh
# Runs the opt-in GPU tests (*_gpu_tests) against the MoltenVK of a variant, but only when the game
# agent does not hold the GPU: kbench-results/GPU_LOCK must not exist (else exit 75 = BUSY, nothing run).
# The lock is taken for the duration of the tests (seconds) and removed on exit.
#   tools/gpu_tests.sh [-v <variant-dir>] [-b <build-dir>] [-t <seconds>] [test ...]
# Defaults: variant = newest _Build/variants/* that has libMoltenVK.dylib; build = <root>/_Build/macos;
# tests = per_vertex_vsstore_gpu_tests dcc_gpu_resolve_gpu_tests gpu_pad_gpu_tests device_address_lifetime_gpu_tests
# (a test given as "name:arg" gets the argument). Env: GPU_LOCK overrides the lock path;
# GPU_TESTS_SKIP_PGREP=1 skips the kyty_emulator process check (only to self-test this script with dummy tests).
# Output: one "PASS|FAIL|MISSING <test> <seconds>s" line per test, then "RESULT PASS|FAIL n_pass/n_total".
# Exit: 0 all passed, 1 a test failed, 2 usage/missing build, 75 BUSY.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
variant=
build=$root/_Build/macos
limit=120
while getopts v:b:t: opt; do
	case $opt in
		v) variant=$OPTARG ;;
		b) build=$OPTARG ;;
		t) limit=$OPTARG ;;
		*) echo "usage: $0 [-v variant-dir] [-b build-dir] [-t seconds] [test ...]" >&2; exit 2 ;;
	esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || set -- per_vertex_vsstore_gpu_tests dcc_gpu_resolve_gpu_tests gpu_pad_gpu_tests device_address_lifetime_gpu_tests

if [ -z "$variant" ]; then
	for d in $(ls -dt "$root"/_Build/variants/*/ 2>/dev/null); do
		if [ -f "$d/libMoltenVK.dylib" ]; then variant=${d%/}; break; fi
	done
fi
if [ -z "$variant" ] || [ ! -f "$variant/libMoltenVK.dylib" ]; then
	echo "no variant with libMoltenVK.dylib (use -v)" >&2
	exit 2
fi

lock=${GPU_LOCK:-$root/../kbench-results/GPU_LOCK}
if [ -e "$lock" ]; then
	echo "BUSY: $lock exists ($(head -1 "$lock" 2>/dev/null))"
	exit 75
fi
if [ -z "${GPU_TESTS_SKIP_PGREP:-}" ] && pgrep -f kyty_emulator >/dev/null 2>&1; then
	echo "BUSY: a kyty_emulator process is running"
	exit 75
fi
# noclobber makes the create atomic: if the game agent wins the race we see BUSY, not a shared lock.
if ! (set -C; echo "gpu-tests $$ $(date '+%Y-%m-%d %H:%M')" >"$lock") 2>/dev/null; then
	echo "BUSY: $lock appeared"
	exit 75
fi
trap 'if grep -q " $$ " "$lock" 2>/dev/null; then rm -f "$lock"; fi' EXIT INT TERM HUP

export DYLD_LIBRARY_PATH="$variant${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export SDL_VULKAN_LIBRARY="$variant/libMoltenVK.dylib"
now() { perl -MTime::HiRes=time -e 'printf "%.1f", time'; }

total=0
passed=0
for spec in "$@"; do
	name=${spec%%:*}
	arg=
	[ "$name" = "$spec" ] || arg=${spec#*:}
	total=$((total + 1))
	bin=$build/$name
	if [ ! -x "$bin" ]; then
		echo "MISSING $name (build: cmake --build $build --target $name)"
		continue
	fi
	t0=$(now)
	log=$(mktemp -t gpu_tests)
	# shellcheck disable=SC2086
	"$bin" $arg >"$log" 2>&1 &
	pid=$!
	( sleep "$limit"; kill -9 "$pid" 2>/dev/null ) &
	watchdog=$!
	wait "$pid"
	rc=$?
	kill "$watchdog" 2>/dev/null
	wait "$watchdog" 2>/dev/null
	t1=$(now)
	secs=$(perl -e "printf '%.1f', $t1 - $t0")
	if [ "$rc" -eq 0 ]; then
		passed=$((passed + 1))
		echo "PASS $name ${secs}s"
	else
		echo "FAIL $name ${secs}s rc=$rc"
		tail -5 "$log" | sed 's/^/    /'
	fi
	rm -f "$log"
done
if [ "$passed" -eq "$total" ]; then
	echo "RESULT PASS $passed/$total (variant $(basename "$variant"))"
	exit 0
fi
echo "RESULT FAIL $passed/$total (variant $(basename "$variant"))"
exit 1
