#!/bin/sh
# Mutants of tools/run_slot.py: each must make tools/test_run_slot.py fail (rc != 0). Prints "<name> rc=<n>".
cd "$(dirname "$0")" || exit 2
tmp=$(mktemp -d)
status=0
run_mutant() { # name, python one-liner replacing text in the copy
	mkdir "$tmp/$1"
	python3 -I -c "import sys;s=open('run_slot.py').read();a,b=sys.argv[1],sys.argv[2];assert a in s;open('$tmp/$1/run_slot.py','w').write(s.replace(a,b,1))" "$2" "$3" || { echo "$1 PATCH-FAILED"; status=1; return; }
	RUN_SLOT_MODULE_DIR="$tmp/$1" python3 -I test_run_slot.py >/dev/null 2>&1
	rc=$?
	echo "$1 rc=$rc"
	[ "$rc" -ne 0 ] || status=1
}
run_mutant tempo_lock_ok 'and words[-1] == "image"' 'and True'
run_mutant no_lock_ok 'if lock_text is None:' 'if False:'
run_mutant dead_ok 'or not pid_alive(lock["pid"])' 'or False'
run_mutant second_slot_ok 'if share_text is not None:' 'if False:'
run_mutant shared_dump_dir 'if dump_dir is not None:' 'if False:'
run_mutant slot_dir_same 'variant.name + "-slot2"' 'variant.name'
rm -rf "$tmp"
exit $status
