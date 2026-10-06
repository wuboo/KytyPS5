#!/bin/sh
# Runs the uma_probe battery. Usage: run_probe.sh <libMoltenVK.dylib> <out-dir>
# Takes no GPU_LOCK itself: the caller must hold it (rt-agent <pid> <date>) and no kyty_emulator may be running.
set -u
here=$(cd "$(dirname "$0")" && pwd)
dylib=$1; out=$2; mkdir -p "$out"
run() { # label, binary, args...
	label=$1; shift
	"$@" > "$out/$label.raw" 2>&1; rc=$?
	grep -vE '^[[:space:]]|^\[mvk-' "$out/$label.raw" > "$out/$label.txt"
	echo "$label rc=$rc fails=$(grep -c ' FAIL$' "$out/$label.txt")"
}
run arm64-1g-shm   "$here/uma_probe.arm64"  --dylib "$dylib" --size-mb 1024
run x64-1g-shm     "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 1024
run x64-1g-anon    "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 1024 --backing anon
run x64-1g-mutant  "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 1024 --mutant --no-canary
run x64-4g-shm     "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 4096 --no-canary
run x64-16g-import "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 16384 --no-canary --import-only
# repeats (system_wired is host-wide and noisy: compare deltas across runs)
run x64-1g-shm-r2  "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 1024 --no-canary
run x64-1g-shm-r3  "$here/uma_probe.x86_64" --dylib "$dylib" --size-mb 1024 --no-canary
