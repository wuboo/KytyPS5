# S433: KYTY_RECOMPILER_SIGNATURE, a SHA256 over the sources that decide what the shader recompiler and the per-vertex generators emit.
# It replaces KYTY_GIT_REVISION in the key of the _ShaderCache/<TITLE>/pv_*.bin and pvu_*.bin files, so a commit that changes nothing here (loader,
# command processor, UMA, audio, tests, build files) keeps the cache warm, and a change here (or a dependency pin) invalidates it.
#
# What goes in (paths relative to the repository root, each as "<path>:<sha256 of the content>"):
#   src/graphics/shader/**                              the recompiler: frontend (decode, CFG), IR and passes, SPIR-V backend, shader.cpp, bindings
#   src/graphics/host_gpu/renderer/perVertex*           per-vertex transform, prototype, VS-store (+ embedded SPIR-V) and their headers
#   gitlinks 3rdparty/SPIRV-Tools, 3rdparty/SPIRV-Headers   the SPIR-V libraries the transform and the validator use
#   KYTY_RECOMPILER_EPOCH (below)                       a manual bump for a change this list cannot see
# Why it is safe: pv_*.bin holds SPIR-V and a vertex layout produced by those sources alone, from the SPIR-V the recompiler emits for a (vs, ps) pair
# of guest shader hashes. Nothing else writes into it. The working-tree content is hashed (not the commit), so uncommitted edits count too.
# Inputs that are not code - the environment (KYTY_SPV_NOCONTRACT) and the format version - stay in GetPerVertexTransformSignature.
# What it does not see: a compiler or standard library change that alters codegen without a source change (none known to change SPIR-V),
# files outside the list that the generators start to depend on (extend the list), build flags. Bump KYTY_RECOMPILER_EPOCH in those cases.
set(KYTY_RECOMPILER_EPOCH 1)

# kyty_recompiler_signature(<repo root> <out var>) ; with KYTY_SIG_MUTANT set (tests only) the list is deliberately wrong.
function(kyty_recompiler_signature ROOT OUT_VAR)
	set(dirs_recursive "src/graphics/shader")
	if(KYTY_SIG_MUTANT STREQUAL "IGNORE_SHADER_DIR")
		set(dirs_recursive "")
	elseif(KYTY_SIG_MUTANT STREQUAL "HASH_WHOLE_SRC")
		set(dirs_recursive "src")
	endif()
	set(files "")
	foreach(dir IN LISTS dirs_recursive)
		file(GLOB_RECURSE found RELATIVE "${ROOT}" "${ROOT}/${dir}/*.h" "${ROOT}/${dir}/*.hpp" "${ROOT}/${dir}/*.cpp" "${ROOT}/${dir}/*.inc")
		list(APPEND files ${found})
	endforeach()
	if(NOT KYTY_SIG_MUTANT STREQUAL "IGNORE_PV")
		file(GLOB found RELATIVE "${ROOT}" "${ROOT}/src/graphics/host_gpu/renderer/perVertex*.h" "${ROOT}/src/graphics/host_gpu/renderer/perVertex*.cpp"
		     "${ROOT}/src/graphics/host_gpu/renderer/perVertex*.inc")
		list(APPEND files ${found})
	endif()
	list(REMOVE_DUPLICATES files)
	list(SORT files)
	set(combined "epoch=${KYTY_RECOMPILER_EPOCH};")
	foreach(file IN LISTS files)
		file(SHA256 "${ROOT}/${file}" content_hash)
		if(KYTY_SIG_MUTANT STREQUAL "PATH_IGNORED")
			string(APPEND combined "${content_hash};")
		else()
			string(APPEND combined "${file}:${content_hash};")
		endif()
	endforeach()
	# The pins of the SPIR-V libraries: the gitlinks of the superproject (a plain file tree in a source archive has none).
	if(GIT_EXECUTABLE AND NOT KYTY_SIG_MUTANT STREQUAL "IGNORE_PINS")
		execute_process(
			COMMAND "${GIT_EXECUTABLE}" ls-tree HEAD 3rdparty/SPIRV-Tools 3rdparty/SPIRV-Headers
			WORKING_DIRECTORY "${ROOT}"
			OUTPUT_VARIABLE pins
			OUTPUT_STRIP_TRAILING_WHITESPACE
			RESULT_VARIABLE pins_result
			ERROR_QUIET
		)
		if(pins_result EQUAL 0)
			string(APPEND combined "pins=${pins};")
		endif()
	endif()
	string(SHA256 signature "${combined}")
	set(${OUT_VAR} "${signature}" PARENT_SCOPE)
endfunction()
