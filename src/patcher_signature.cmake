# S437: KYTY_PATCHER_SIGNATURE, a SHA256 over the sources that decide what the guest patchers write into the executable segments. It goes into the
# key of the KYTY_PATCH_CACHE entries (_PatchCache), so a build whose patchers differ never applies an entry written by another build, and a
# commit that changes nothing here (shader recompiler, command processor, audio, tests) keeps the cache warm (unlike a binary UUID).
#
# What goes in (paths relative to the repository root, each as "<path>:<sha256 of the content>"):
#   src/loader/guestInstructionPatcher.*  AMD CPU patcher (SSE4a, RDPID, CLWB, rosettacmp), red-zone patcher, trampoline generator
#   src/loader/x64InstructionEmulator.*   256-bit store split (wide-store traps)
#   src/loader/patchCache.*               what is stored and how it is applied
#   the Zydis and xbyak pins (GIT_TAG of 3rdparty/CMakeLists.txt): the decoder and the assembler the patchers use
#   KYTY_PATCHER_EPOCH                    a manual bump for a change this list cannot see
# The working-tree content is hashed (not the commit), so uncommitted edits count too. What it does not see: a compiler change that alters codegen
# of the patchers, files the patchers start to depend on (extend the list), build flags. Bump KYTY_PATCHER_EPOCH in those cases
# (loader/patchCache.h PatchVersion stays the manual bump for a change of the entry format).
set(KYTY_PATCHER_EPOCH 1)

# kyty_patcher_signature(<repo root> <out var>) ; with KYTY_SIG_MUTANT set (tests only) the list is deliberately wrong.
function(kyty_patcher_signature ROOT OUT_VAR)
	set(globs "${ROOT}/src/loader/guestInstructionPatcher.*" "${ROOT}/src/loader/x64InstructionEmulator.*" "${ROOT}/src/loader/patchCache.*")
	if(KYTY_SIG_MUTANT STREQUAL "IGNORE_EMULATOR")
		set(globs "${ROOT}/src/loader/guestInstructionPatcher.*" "${ROOT}/src/loader/patchCache.*")
	elseif(KYTY_SIG_MUTANT STREQUAL "IGNORE_PATCHER")
		set(globs "${ROOT}/src/loader/x64InstructionEmulator.*" "${ROOT}/src/loader/patchCache.*")
	elseif(KYTY_SIG_MUTANT STREQUAL "HASH_WHOLE_LOADER")
		set(globs "${ROOT}/src/loader/*")
	endif()
	file(GLOB files RELATIVE "${ROOT}" ${globs})
	list(REMOVE_DUPLICATES files)
	list(SORT files)
	set(combined "epoch=${KYTY_PATCHER_EPOCH};")
	foreach(file IN LISTS files)
		file(SHA256 "${ROOT}/${file}" content_hash)
		if(KYTY_SIG_MUTANT STREQUAL "PATH_IGNORED")
			string(APPEND combined "${content_hash};")
		else()
			string(APPEND combined "${file}:${content_hash};")
		endif()
	endforeach()
	# The pins of the libraries (FetchContent_Declare ... GIT_TAG <sha>).
	if(EXISTS "${ROOT}/3rdparty/CMakeLists.txt" AND NOT KYTY_SIG_MUTANT STREQUAL "IGNORE_PINS")
		file(READ "${ROOT}/3rdparty/CMakeLists.txt" deps)
		foreach(lib zydis xbyak)
			if(deps MATCHES "FetchContent_Declare\\([ \t\r\n]*${lib}[^)]*GIT_TAG[ \t]+([0-9a-fA-F]+)")
				string(APPEND combined "${lib}=${CMAKE_MATCH_1};")
			endif()
		endforeach()
	endif()
	string(SHA256 signature "${combined}")
	set(${OUT_VAR} "${signature}" PARENT_SCOPE)
endfunction()
