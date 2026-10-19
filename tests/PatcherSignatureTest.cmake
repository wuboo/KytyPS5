# S437: KYTY_PATCHER_SIGNATURE (src/patcher_signature.cmake). Run: cmake -DSIG_DIR=<src> [-DKYTY_SIG_MUTANT=<name>] -P PatcherSignatureTest.cmake
# A change in a patcher source, a new or renamed patcher file, a changed Zydis / xbyak pin or a bumped epoch changes the signature; a change in
# other loader files, the recompiler or the tests does not. Mutants (WILL_FAIL): IGNORE_EMULATOR, IGNORE_PATCHER, HASH_WHOLE_LOADER, PATH_IGNORED, IGNORE_PINS.
cmake_minimum_required(VERSION 3.19)
include("${SIG_DIR}/patcher_signature.cmake")

# check(<message> <if() condition...>): the condition is passed as separate arguments, not as one quoted string.
function(check text)
	if(NOT (${ARGN}))
		message(SEND_ERROR "FAIL: ${text}")
	endif()
endfunction()

string(RANDOM LENGTH 8 suffix)
set(root "${CMAKE_CURRENT_LIST_DIR}/../_psigtest_${suffix}")
get_filename_component(root "${root}" ABSOLUTE)
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/src/loader" "${root}/src/graphics/shader" "${root}/3rdparty" "${root}/tests")
file(WRITE "${root}/src/loader/guestInstructionPatcher.cpp" "int patch();\n")
file(WRITE "${root}/src/loader/guestInstructionPatcher.h" "int patch();\n")
file(WRITE "${root}/src/loader/x64InstructionEmulator.cpp" "int split();\n")
file(WRITE "${root}/src/loader/patchCache.cpp" "int cache();\n")
file(WRITE "${root}/src/loader/runtimeLinker.cpp" "int link();\n")
file(WRITE "${root}/src/graphics/shader/shader.cpp" "int shader();\n")
file(WRITE "${root}/tests/SomeTests.cpp" "int main();\n")
set(pins "FetchContent_Declare(\n\txbyak\n\tGIT_REPOSITORY https://x/xbyak.git\n\tGIT_TAG 44a72f369268f7d552650891b296693e91db86bb\n)\nFetchContent_Declare(\n\tzydis\n\tGIT_REPOSITORY https://x/zydis.git\n\tGIT_TAG 120e0e705f8e3b507dc49377ac2879979f0d545c\n\tGIT_SHALLOW FALSE\n)\n")
file(WRITE "${root}/3rdparty/CMakeLists.txt" "${pins}")

kyty_patcher_signature("${root}" base)
kyty_patcher_signature("${root}" base_again)
check("the signature must be deterministic" "${base}" STREQUAL "${base_again}")
string(LENGTH "${base}" base_length)
check("the signature is a SHA256 hex string" ${base_length} EQUAL 64)

# 1. Files outside the list: another loader file, the recompiler, tests, a new file.
file(WRITE "${root}/src/loader/runtimeLinker.cpp" "int link(); // changed\n")
file(WRITE "${root}/src/graphics/shader/shader.cpp" "int shader(); // changed\n")
file(WRITE "${root}/tests/SomeTests.cpp" "int main(); // changed\n")
file(WRITE "${root}/src/loader/newLoaderFile.cpp" "int added();\n")
kyty_patcher_signature("${root}" outside)
check("a change outside the patchers must keep the signature" "${outside}" STREQUAL "${base}")

# 2. A change in the AMD / red-zone patcher (source and header).
file(WRITE "${root}/src/loader/guestInstructionPatcher.cpp" "int patch(); // changed\n")
kyty_patcher_signature("${root}" patcher_cpp)
check("a change in guestInstructionPatcher.cpp must change the signature" NOT "${patcher_cpp}" STREQUAL "${base}")
file(WRITE "${root}/src/loader/guestInstructionPatcher.cpp" "int patch();\n")
file(WRITE "${root}/src/loader/guestInstructionPatcher.h" "int patch(); // changed\n")
kyty_patcher_signature("${root}" patcher_h)
check("a change in guestInstructionPatcher.h must change the signature" NOT "${patcher_h}" STREQUAL "${base}")
file(WRITE "${root}/src/loader/guestInstructionPatcher.h" "int patch();\n")

# 3. A change in the wide store split and in the cache itself.
file(WRITE "${root}/src/loader/x64InstructionEmulator.cpp" "int split(); // changed\n")
kyty_patcher_signature("${root}" emulator)
check("a change in x64InstructionEmulator.cpp must change the signature" NOT "${emulator}" STREQUAL "${base}")
file(WRITE "${root}/src/loader/x64InstructionEmulator.cpp" "int split();\n")
file(WRITE "${root}/src/loader/patchCache.cpp" "int cache(); // changed\n")
kyty_patcher_signature("${root}" cache)
check("a change in patchCache.cpp must change the signature" NOT "${cache}" STREQUAL "${base}")
file(WRITE "${root}/src/loader/patchCache.cpp" "int cache();\n")

# 4. A rename with the same content (inside the patcher file set).
file(RENAME "${root}/src/loader/guestInstructionPatcher.h" "${root}/src/loader/guestInstructionPatcher.hh")
kyty_patcher_signature("${root}" renamed)
check("a renamed patcher file must change the signature" NOT "${renamed}" STREQUAL "${base}")
file(RENAME "${root}/src/loader/guestInstructionPatcher.hh" "${root}/src/loader/guestInstructionPatcher.h")

# 5. Undoing the edits restores it; the epoch changes it.
kyty_patcher_signature("${root}" restored)
check("undoing the edits must restore the signature" "${restored}" STREQUAL "${base}")
set(KYTY_PATCHER_EPOCH 2)
kyty_patcher_signature("${root}" epoch)
check("bumping the epoch must change the signature" NOT "${epoch}" STREQUAL "${base}")
set(KYTY_PATCHER_EPOCH 1)

# 6. The pins of Zydis and xbyak.
string(REPLACE "120e0e705f8e3b507dc49377ac2879979f0d545c" "220e0e705f8e3b507dc49377ac2879979f0d545c" zydis_bumped "${pins}")
file(WRITE "${root}/3rdparty/CMakeLists.txt" "${zydis_bumped}")
kyty_patcher_signature("${root}" zydis_pin)
check("a changed Zydis pin must change the signature" NOT "${zydis_pin}" STREQUAL "${base}")
string(REPLACE "44a72f369268f7d552650891b296693e91db86bb" "54a72f369268f7d552650891b296693e91db86bb" xbyak_bumped "${pins}")
file(WRITE "${root}/3rdparty/CMakeLists.txt" "${xbyak_bumped}")
kyty_patcher_signature("${root}" xbyak_pin)
check("a changed xbyak pin must change the signature" NOT "${xbyak_pin}" STREQUAL "${base}")

file(REMOVE_RECURSE "${root}")
message(STATUS "patcher_signature: done")
