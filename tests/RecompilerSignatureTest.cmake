# S433: KYTY_RECOMPILER_SIGNATURE (src/recompiler_signature.cmake). Run: cmake -DSIG_DIR=<src> [-DKYTY_SIG_MUTANT=<name>] -P RecompilerSignatureTest.cmake
# A file outside the listed directories changes nothing; a change in the recompiler, in a per-vertex generator, a new or renamed file, a changed
# SPIR-V library pin or a bumped epoch changes the signature. Mutants (WILL_FAIL): IGNORE_SHADER_DIR, IGNORE_PV, HASH_WHOLE_SRC, PATH_IGNORED, IGNORE_PINS.
cmake_minimum_required(VERSION 3.19)
include("${SIG_DIR}/recompiler_signature.cmake")
find_package(Git QUIET)

set(failures 0)
# check(<message> <if() condition...>): the condition is passed as separate arguments, not as one quoted string.
function(check text)
	if(NOT (${ARGN}))
		message(SEND_ERROR "FAIL: ${text}")
	endif()
endfunction()

string(RANDOM LENGTH 8 suffix)
set(root "${CMAKE_CURRENT_LIST_DIR}/../_sigtest_${suffix}")
get_filename_component(root "${root}" ABSOLUTE)
file(REMOVE_RECURSE "${root}")
file(MAKE_DIRECTORY "${root}/src/graphics/shader/recompiler/backend" "${root}/src/graphics/host_gpu/renderer" "${root}/src/loader" "${root}/tests")
file(WRITE "${root}/src/graphics/shader/recompiler/ShaderRecompiler.cpp" "int translate();\n")
file(WRITE "${root}/src/graphics/shader/recompiler/backend/SpirvEmitter.h" "struct Emitter {};\n")
file(WRITE "${root}/src/graphics/host_gpu/renderer/perVertexTransform.cpp" "int pv();\n")
file(WRITE "${root}/src/graphics/host_gpu/renderer/render.cpp" "int render();\n")
file(WRITE "${root}/src/loader/runtimeLinker.cpp" "int link();\n")
file(WRITE "${root}/tests/SomeTests.cpp" "int main();\n")

kyty_recompiler_signature("${root}" base)
kyty_recompiler_signature("${root}" base_again)
check("the signature must be deterministic" "${base}" STREQUAL "${base_again}")
string(LENGTH "${base}" base_length)
check("the signature is a SHA256 hex string" ${base_length} EQUAL 64)

# 1. Files outside the list: loader, tests, renderer files that are not per-vertex generators, new top-level files.
file(WRITE "${root}/src/loader/runtimeLinker.cpp" "int link(); // changed\n")
file(WRITE "${root}/tests/SomeTests.cpp" "int main(); // changed\n")
file(WRITE "${root}/src/graphics/host_gpu/renderer/render.cpp" "int render(); // changed\n")
file(WRITE "${root}/src/new_file.cpp" "int added();\n")
kyty_recompiler_signature("${root}" outside)
check("a change outside the recompiler and the generators must keep the signature" "${outside}" STREQUAL "${base}")

# 2. A change in the recompiler.
file(WRITE "${root}/src/graphics/shader/recompiler/ShaderRecompiler.cpp" "int translate(); // changed\n")
kyty_recompiler_signature("${root}" recompiler)
check("a change in the recompiler must change the signature" NOT "${recompiler}" STREQUAL "${base}")

# 3. A change in the SPIR-V backend (nested directory) on top of the original tree.
file(WRITE "${root}/src/graphics/shader/recompiler/ShaderRecompiler.cpp" "int translate();\n")
file(WRITE "${root}/src/graphics/shader/recompiler/backend/SpirvEmitter.h" "struct Emitter { int x; };\n")
kyty_recompiler_signature("${root}" backend)
check("a change in the SPIR-V backend must change the signature" NOT "${backend}" STREQUAL "${base}")
file(WRITE "${root}/src/graphics/shader/recompiler/backend/SpirvEmitter.h" "struct Emitter {};\n")

# 4. A change in a per-vertex generator.
file(WRITE "${root}/src/graphics/host_gpu/renderer/perVertexTransform.cpp" "int pv(); // changed\n")
kyty_recompiler_signature("${root}" pv)
check("a change in a per-vertex generator must change the signature" NOT "${pv}" STREQUAL "${base}")
file(WRITE "${root}/src/graphics/host_gpu/renderer/perVertexTransform.cpp" "int pv();\n")

# 5. A new file in the recompiler.
file(WRITE "${root}/src/graphics/shader/recompiler/NewPass.cpp" "int pass();\n")
kyty_recompiler_signature("${root}" added)
check("a new recompiler file must change the signature" NOT "${added}" STREQUAL "${base}")
file(REMOVE "${root}/src/graphics/shader/recompiler/NewPass.cpp")

# 6. A rename with the same content.
file(RENAME "${root}/src/graphics/shader/recompiler/backend/SpirvEmitter.h" "${root}/src/graphics/shader/recompiler/backend/Emitter.h")
kyty_recompiler_signature("${root}" renamed)
check("a renamed recompiler file must change the signature" NOT "${renamed}" STREQUAL "${base}")
file(RENAME "${root}/src/graphics/shader/recompiler/backend/Emitter.h" "${root}/src/graphics/shader/recompiler/backend/SpirvEmitter.h")

# 7. The epoch.
kyty_recompiler_signature("${root}" restored)
check("undoing the edits must restore the signature" "${restored}" STREQUAL "${base}")
set(KYTY_RECOMPILER_EPOCH 2)
kyty_recompiler_signature("${root}" epoch)
check("bumping the epoch must change the signature" NOT "${epoch}" STREQUAL "${base}")
set(KYTY_RECOMPILER_EPOCH 1)

# 8. The pins of the SPIR-V libraries (needs git).
if(GIT_FOUND)
	set(git "${GIT_EXECUTABLE}" -c user.name=t -c user.email=t@example.invalid -c commit.gpgsign=false)
	execute_process(COMMAND ${git} init -q WORKING_DIRECTORY "${root}" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ${git} update-index --add --cacheinfo 160000,1111111111111111111111111111111111111111,3rdparty/SPIRV-Tools
	                WORKING_DIRECTORY "${root}" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ${git} commit -q -m pins WORKING_DIRECTORY "${root}" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
	kyty_recompiler_signature("${root}" pin_one)
	execute_process(COMMAND ${git} update-index --cacheinfo 160000,2222222222222222222222222222222222222222,3rdparty/SPIRV-Tools
	                WORKING_DIRECTORY "${root}" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
	execute_process(COMMAND ${git} commit -q -m pins2 WORKING_DIRECTORY "${root}" RESULT_VARIABLE rc OUTPUT_QUIET ERROR_QUIET)
	kyty_recompiler_signature("${root}" pin_two)
	check("a changed SPIR-V library pin must change the signature" NOT "${pin_one}" STREQUAL "${pin_two}")
else()
	message(STATUS "git not found: the pin check is skipped")
endif()

file(REMOVE_RECURSE "${root}")
message(STATUS "recompiler_signature: done")
