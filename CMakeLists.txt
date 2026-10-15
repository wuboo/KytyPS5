cmake_minimum_required(VERSION 3.19...4.3)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
	set(CMAKE_POLICY_VERSION_MINIMUM 3.10 CACHE STRING "Minimum policy version for bundled dependencies")
endif()

get_property(KYTY_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT KYTY_GENERATOR_IS_MULTI_CONFIG AND NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build" FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

project(Kyty)

find_program(SCCACHE_PROGRAM sccache)
find_program(CCACHE_PROGRAM ccache)

foreach(lang IN ITEMS C CXX)
    # An explicitly empty launcher disables caching for this language.
    if(NOT DEFINED CMAKE_${lang}_COMPILER_LAUNCHER)
        if(SCCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}")
            message(STATUS "sccache enabled for ${lang}: ${SCCACHE_PROGRAM}")
        elseif(CCACHE_PROGRAM)
            set(CMAKE_${lang}_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
            message(STATUS "ccache enabled for ${lang}: ${CCACHE_PROGRAM}")
        endif()
    endif()
endforeach()

set(KYTY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(KYTY_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests")
set(KYTY_THIRD_PARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")

if(CMAKE_SYSTEM_NAME MATCHES ".*Linux")
	set(LINUX TRUE)
endif()

if (NOT (WIN32 OR LINUX OR APPLE))
    message(FATAL_ERROR "only Windows, Linux, and macOS builds are supported")
endif()

if(WIN32)
	# Prevent Windows headers from defining min/max macros in all targets.
	add_compile_definitions(NOMINMAX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

include("${KYTY_SOURCE_DIR}/utils.cmake")
include(CTest)

option(KYTY_ENABLE_CLANG_TIDY "Run clang-tidy checks during builds" OFF)

set(KYTY_BUILD_ORIGIN "Source" CACHE STRING "Build provenance shown in diagnostics")
set_property(CACHE KYTY_BUILD_ORIGIN PROPERTY STRINGS Source Official Fork)
if(NOT KYTY_BUILD_ORIGIN STREQUAL "Source" AND NOT KYTY_BUILD_ORIGIN STREQUAL "Official" AND
   NOT KYTY_BUILD_ORIGIN STREQUAL "Fork")
	message(FATAL_ERROR "KYTY_BUILD_ORIGIN must be Source, Official, or Fork")
endif()

set(KYTY_RELEASE_TAG "" CACHE STRING "Release tag embedded in official builds")
set(KYTY_BUILD_REPOSITORY "" CACHE STRING "GitHub repository embedded in fork builds")
set(KYTY_OFFICIAL_BUILD OFF)
set(KYTY_FORK_BUILD OFF)
if(KYTY_BUILD_ORIGIN STREQUAL "Official")
	if(KYTY_RELEASE_TAG STREQUAL "")
		message(FATAL_ERROR "Official builds require KYTY_RELEASE_TAG")
	endif()
	set(KYTY_OFFICIAL_BUILD ON)
elseif(KYTY_BUILD_ORIGIN STREQUAL "Fork")
	if(KYTY_BUILD_REPOSITORY STREQUAL "")
		message(FATAL_ERROR "Fork builds require KYTY_BUILD_REPOSITORY")
	endif()
	set(KYTY_FORK_BUILD ON)
endif()

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

include(TestBigEndian)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
	message(FATAL_ERROR "only 64-bit builds are supported")
endif()

test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
	set(KYTY_ENDIAN KYTY_ENDIAN_BIG)
else()
	set(KYTY_ENDIAN KYTY_ENDIAN_LITTLE)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
	set(KYTY_BUILD KYTY_BUILD_DEBUG)
else()
	set(KYTY_BUILD KYTY_BUILD_RELEASE)
endif()

if(LINUX OR APPLE)
	# macOS rides the POSIX/Linux code paths until it gets a dedicated platform
	set(KYTY_PLATFORM KYTY_PLATFORM_LINUX)
else()
	set(KYTY_PLATFORM KYTY_PLATFORM_WINDOWS)
endif()

set(CLANG FALSE)
set(GCC FALSE)
set(KYTY_CLANG_CL FALSE)
get_filename_component(KYTY_CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME_WE)
string(TOLOWER "${KYTY_CXX_COMPILER_NAME}" KYTY_CXX_COMPILER_NAME)

if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
	set(CLANG TRUE)
	set(KYTY_COMPILER CLANG)
	if(WIN32 AND KYTY_CXX_COMPILER_NAME STREQUAL "clang-cl")
		set(KYTY_CLANG_CL TRUE)
		set(KYTY_LINKER LLD_LINK)
	elseif(APPLE)
		set(KYTY_LINKER LD64) # Apple's default linker; lld flags don't apply
	else()
		set(KYTY_LINKER LLD)
	endif()
	if ((CMAKE_AR MATCHES "llvm") AND (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12.0.0) AND (CMAKE_CXX_COMPILER MATCHES "winlibs"))
		string(REPLACE "llvm-ar.exe" "ar.exe" CMAKE_AR ${CMAKE_AR}) # llvm-ar.exe 11.0.0 from winlibs is broken
	endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(GCC TRUE)
	set(KYTY_COMPILER GCC)
	set(KYTY_LINKER LD)
else()
	message(FATAL_ERROR "only Clang and GCC are supported; found ${CMAKE_CXX_COMPILER_ID}")
endif()

set(KYTY_PROJECT_NAME "Emulator" CACHE STRING "Project name suffix")

string(TOLOWER ${KYTY_COMPILER} KYTY_COMPILER_ID)
string(TOLOWER ${KYTY_LINKER} KYTY_LINKER_ID)

if (CLANG AND (KYTY_LINKER STREQUAL LLD))
	set(KYTY_LD_OPTIONS "-fuse-ld=lld")
endif()

if (KYTY_LINKER STREQUAL LD AND NOT LINUX)
	set(KYTY_LD_OPTIONS "-Wl,--image-base=0x100000000000")
endif()

project(Kyty${KYTY_PROJECT_NAME}${CMAKE_BUILD_TYPE}${KYTY_COMPILER} VERSION 0.3.0)

include_directories(
	${KYTY_THIRD_PARTY_DIR}/gtest/include
	${KYTY_THIRD_PARTY_DIR}/gtest
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/magic_enum/include/magic_enum
	${PROJECT_BINARY_DIR}
	${KYTY_SOURCE_DIR}
)

set(KYTY_VERSION "${PROJECT_VERSION}")

configure_file(
  ${KYTY_SOURCE_DIR}/cmake_config.h.in
  ${PROJECT_BINARY_DIR}/cmake_config.h
  )

find_package(Git)

add_custom_target(KytyGitVersion
	COMMAND ${CMAKE_COMMAND}
		-D INPUT_FILE=${KYTY_SOURCE_DIR}/kytyGitVersion.h.in
		-D OUTPUT_FILE=${PROJECT_BINARY_DIR}/kytyGitVersion.h
		-D GIT_EXECUTABLE=${GIT_EXECUTABLE}
		-D GIT_WORKING_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR}
		-P ${KYTY_SOURCE_DIR}/generate_version.cmake
	BYPRODUCTS ${PROJECT_BINARY_DIR}/kytyGitVersion.h
	COMMENT "Generate kytyGitVersion.h"
)

add_library(kyty_git_version INTERFACE)
add_dependencies(kyty_git_version KytyGitVersion)

if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0.0)
	list(APPEND KYTY_IWYU
		kyty_emulator
		common
		#launcher
	)
	list(APPEND KYTY_CLANG_TIDY
		kyty_emulator
		#common
		launcher
	)
endif()

option(KYTY_BUILD_LAUNCHER "Build Qt launcher" ON)

config_compiler_and_linker()

add_subdirectory("${KYTY_THIRD_PARTY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/3rdparty")
add_subdirectory("${KYTY_SOURCE_DIR}/common" "${CMAKE_CURRENT_BINARY_DIR}/common")

file(GLOB kyty_emulator_src CONFIGURE_DEPENDS
	"${KYTY_SOURCE_DIR}/libs/*.cpp"
	"${KYTY_SOURCE_DIR}/libs/*.h"
	"${KYTY_SOURCE_DIR}/graphics/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/command_processor/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/cache/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/image/*.h"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/pipeline/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/cfg/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/decode/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/frontend/translate/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/backend/spirv/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/opcodes/*.inc"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/*.h"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/window/*.h"
	"${KYTY_SOURCE_DIR}/kernel/*.cpp"
	"${KYTY_SOURCE_DIR}/kernel/*.h"
	"${KYTY_SOURCE_DIR}/loader/*.cpp"
	"${KYTY_SOURCE_DIR}/loader/*.h"
)

find_program(KYTY_GLSLANG_VALIDATOR NAMES glslangValidator glslang REQUIRED)
set(gpu_tiler_shader_dir "${KYTY_SOURCE_DIR}/graphics/host_gpu/shaders")
set(gpu_tiler_generated_dir "${PROJECT_BINARY_DIR}/gpu_tiler_shaders")
set(gpu_tiler_shader_names
	standard256
	standard4
	standard4_3d
	standard64
	standard64_3d
	prt
	prt_3d
	render_target
	depth
	promote_d16
	demote_d16
	color_transform
)
file(GLOB gpu_tiler_shader_includes CONFIGURE_DEPENDS "${gpu_tiler_shader_dir}/gpu_tiler_*.inc")
foreach(shader_name IN LISTS gpu_tiler_shader_names)
	set(shader_source "${gpu_tiler_shader_dir}/gpu_tiler_${shader_name}.comp")
	set(shader_spv "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}.spv")
	set(shader_header "${gpu_tiler_generated_dir}/gpu_tiler_${shader_name}_spv.h")
	string(TOUPPER "GPU_TILER_${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}" ${gpu_tiler_shader_includes}
		VERBATIM
	)
	list(APPEND gpu_tiler_shader_headers "${shader_header}")
endforeach()

set(per_vertex_shader_source "${gpu_tiler_shader_dir}/per_vertex_unpack.comp")
set(per_vertex_shader_generated_dir "${PROJECT_BINARY_DIR}/per_vertex_shaders")
set(per_vertex_shader_spv "${per_vertex_shader_generated_dir}/per_vertex_unpack.spv")
set(per_vertex_shader_header "${per_vertex_shader_generated_dir}/per_vertex_unpack_spv.h")
add_custom_command(
	OUTPUT "${per_vertex_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${per_vertex_shader_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.2 -Os
		-o "${per_vertex_shader_spv}" "${per_vertex_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${per_vertex_shader_spv} -DOUTPUT=${per_vertex_shader_header}
		-DSYMBOL=PERVERTEX_UNPACK_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${per_vertex_shader_source}"
	VERBATIM
)

set(fault_buffer_shader_source "${gpu_tiler_shader_dir}/fault_buffer_process.comp")
set(fault_buffer_shader_spv "${gpu_tiler_generated_dir}/fault_buffer_process.spv")
set(fault_buffer_shader_header "${gpu_tiler_generated_dir}/fault_buffer_process_spv.h")
add_custom_command(
	OUTPUT "${fault_buffer_shader_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_tiler_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -DCACHING_PAGEBITS=14 -DMAX_PAGE_FAULTS=1024
	        -o "${fault_buffer_shader_spv}" "${fault_buffer_shader_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${fault_buffer_shader_spv} -DOUTPUT=${fault_buffer_shader_header}
	        -DSYMBOL=FAULT_BUFFER_PROCESS_SPV -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${fault_buffer_shader_source}"
	VERBATIM
)
list(APPEND kyty_emulator_src "${fault_buffer_shader_header}")
list(APPEND kyty_emulator_src ${gpu_tiler_shader_headers})

set(gpu_blit_generated_dir "${PROJECT_BINARY_DIR}/gpu_blit_shaders")
function(add_gpu_blit_shader shader_name shader_file)
	set(shader_source "${gpu_tiler_shader_dir}/${shader_file}")
	set(shader_spv "${gpu_blit_generated_dir}/${shader_name}.spv")
	set(shader_header "${gpu_blit_generated_dir}/${shader_name}_spv.h")
	string(TOUPPER "${shader_name}_SPV" shader_symbol)
	add_custom_command(
		OUTPUT "${shader_header}"
		COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_blit_generated_dir}"
		COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os ${ARGN}
		        "-I${gpu_tiler_shader_dir}" -o "${shader_spv}" "${shader_source}"
		COMMAND ${CMAKE_COMMAND} -DINPUT=${shader_spv} -DOUTPUT=${shader_header}
		        -DSYMBOL=${shader_symbol} -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
		DEPENDS "${shader_source}"
		VERBATIM
	)
	set(gpu_blit_shader_headers ${gpu_blit_shader_headers} "${shader_header}" PARENT_SCOPE)
endfunction()
add_gpu_blit_shader(gpu_blit_fs_triangle gpu_blit_fs_triangle.vert)
add_gpu_blit_shader(gpu_blit_color_to_ms_depth gpu_blit_color_to_ms_depth.frag)
add_gpu_blit_shader(gpu_video_out_overlay gpu_video_out_overlay.frag)
add_gpu_blit_shader(gpu_imgui_hdr gpu_video_out_overlay.frag -DIMGUI_HDR=1)
list(APPEND kyty_emulator_src ${gpu_blit_shader_headers})
list(APPEND kyty_emulator_src "${per_vertex_shader_header}")

list(APPEND kyty_emulator_src
	"${KYTY_SOURCE_DIR}/emulator.h"
	"${KYTY_SOURCE_DIR}/emulator.cpp"
)
list(REMOVE_DUPLICATES kyty_emulator_src)

# KYTY_GPU_KEEPWARM: native Metal (Objective-C++), macOS only.
if(APPLE)
	set(KYTY_KEEPWARM_MM "${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/gpuKeepWarm.mm")
	# Compiled as C++ with the Objective-C++ dialect forced (enable_language(OBJCXX) after the second project() breaks the toolchain files).
	set_source_files_properties("${KYTY_KEEPWARM_MM}" PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
	list(APPEND kyty_emulator_src "${KYTY_KEEPWARM_MM}")
endif()

if (KYTY_CLANG_CL)
	set_source_files_properties(${kyty_emulator_src} PROPERTIES COMPILE_FLAGS "-Wno-pragma-pack -Wno-deprecated-declarations -D_TIMESPEC_DEFINED")
endif()

set(kyty_emulator_link_libraries common Vulkan::Headers spirv-tools-opt spirv-tools SDL3::SDL3-static imgui xxhash FFmpeg::ffmpeg fmt::fmt nlohmann_json::nlohmann_json LibAtrac9 kyty_git_version Zydis::Zydis xbyak::xbyak opus)

# Linux system libraries.
if(LINUX)
	find_package(Threads REQUIRED)
	list(APPEND kyty_emulator_link_libraries m ${CMAKE_DL_LIBS} Threads::Threads)
endif()

set(inc_headers
	${KYTY_SOURCE_DIR}
	${per_vertex_shader_generated_dir}
	${KYTY_THIRD_PARTY_DIR}/VulkanMemoryAllocator/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Tools/include
	${KYTY_THIRD_PARTY_DIR}/SPIRV-Headers/include
	${KYTY_THIRD_PARTY_DIR}/fmt/include
	${KYTY_THIRD_PARTY_DIR}/xxHash
	${KYTY_THIRD_PARTY_DIR}/cpuinfo/include
	${KYTY_THIRD_PARTY_DIR}/renderdoc
	${KYTY_THIRD_PARTY_DIR}/stb
)

if (KYTY_CLANG_CL)
	list(APPEND kyty_emulator_link_libraries winpthread)
	list(APPEND inc_headers
		${KYTY_THIRD_PARTY_DIR}/winpthread/include
	)
endif()

list(APPEND check_headers
	${KYTY_SOURCE_DIR}
)

function(add_kyty_full_emulator_test target source)
	add_executable(${target} EXCLUDE_FROM_ALL ${source} ${kyty_emulator_src})
	target_link_libraries(${target} ${kyty_emulator_link_libraries})
	target_include_directories(${target} PRIVATE ${inc_headers})
	if (WIN32)
		target_link_libraries(${target} iphlpapi)
	endif()
	if (KYTY_CLANG_CL)
		target_link_libraries(${target} onecore)
		add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:${target}>/libwinpthread-1.dll)
	endif()
	# The macOS x86_64 guest address space needs its .zerofill segments anchored
	# by linker flags, or the kernel kills the binary on load (posix_spawn EIO).
	configure_macos_guest_address_space(${target})
endfunction()

function(configure_macos_guest_address_space target)
	if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
			(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
		target_sources(${target} PRIVATE "${KYTY_SOURCE_DIR}/kernel/macosGuestAddressSpace.cpp")
		target_compile_definitions(${target} PRIVATE KYTY_LINKED_GUEST_ADDRESS_SPACE=1)
		target_link_options(${target} PRIVATE
			-Wl,-ld_classic,-no_pie,-no_fixup_chains,-no_huge,-pagezero_size,0x40000,-segaddr,SYSTEM_MANAGED,0x40000,-segaddr,SYSTEM_RESERVED,0x7ffffc000,-segaddr,USER_AREA,0x7000000000,-image_base,0x700000000000)
	endif()
endfunction()

add_kyty_full_emulator_test(shader_cfg_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")
# S149: emitter mutant of the VS-store PS (corner 0 and 2 swapped); TestPvVsStoreEmission must catch it (WILL_FAIL).
add_kyty_full_emulator_test(shader_cfg_pv_vsstore_mutant_first_third_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")
target_compile_definitions(shader_cfg_pv_vsstore_mutant_first_third_tests PRIVATE KYTY_PVVS_MUTANT_FIRST_THIRD=1)
# S150: the dynamic record stride must come from push dword 7.
add_kyty_full_emulator_test(shader_cfg_pv_vsstore_mutant_stride_dword_tests "${KYTY_TESTS_DIR}/shaderCfgTests.cpp")
target_compile_definitions(shader_cfg_pv_vsstore_mutant_stride_dword_tests PRIVATE KYTY_PVVS_MUTANT_STRIDE_DWORD=1)

add_executable(per_vertex_transform_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PerVertexTransformTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/perVertexTransform.cpp"
)
target_include_directories(per_vertex_transform_tests PRIVATE ${inc_headers})
target_link_libraries(per_vertex_transform_tests common fmt::fmt xxhash)

# S134: KYTY_PV_UNIQUE planner, record model and range flag (CPU). Mutants (each must be caught, WILL_FAIL): the instance term and the "- first" of the record dropped; the 0.7 share, the writes fallback and the dropped fallback removed; the range never reports dropped corners.
add_executable(per_vertex_unique_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PerVertexUniqueTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/perVertexTransform.cpp"
)
target_include_directories(per_vertex_unique_tests PRIVATE ${inc_headers})
target_link_libraries(per_vertex_unique_tests common fmt::fmt xxhash)
foreach(mutant IN ITEMS NO_INSTANCE NO_FIRST NO_RATIO WRITES_OK DROPPED_OK NO_DROPPED)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(per_vertex_unique_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/PerVertexUniqueTests.cpp"
		"${KYTY_SOURCE_DIR}/graphics/host_gpu/renderer/perVertexTransform.cpp"
	)
	target_include_directories(per_vertex_unique_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_link_libraries(per_vertex_unique_mutant_${mutant_lower}_tests common fmt::fmt xxhash)
	target_compile_definitions(per_vertex_unique_mutant_${mutant_lower}_tests PRIVATE KYTY_PVU_MUTANT_${mutant}=1)
endforeach()

# S149: KYTY_PV_VSSTORE planner, record model and pass ring (CPU). Mutants (each must be caught, WILL_FAIL): corner 0/2 swapped, instance term and "- first" dropped, 16-bit halves swapped, first_index and vertex_offset ignored, dropped corners accepted, ring region reused.
add_executable(per_vertex_vsstore_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PerVertexVsStoreTests.cpp"
)
target_include_directories(per_vertex_vsstore_tests PRIVATE ${inc_headers})
foreach(mutant IN ITEMS FIRST_THIRD NO_INSTANCE NO_FIRST HALF_SWAP NO_FIRST_INDEX NO_VERTEX_OFFSET DROPPED_OK REUSE_REGION PARAM31_OK PUSH_NO_FIRST_INSTANCE INDEX_NO_ADJUST POOL_REUSE_BUSY NO_INSTANCED_CLASS NO_TOUCH ALIGN_IGNORED RING_IGNORED PUSH_ALWAYS INDEX_RANGE_NO_ROUND)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(per_vertex_vsstore_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/PerVertexVsStoreTests.cpp"
	)
	target_include_directories(per_vertex_vsstore_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(per_vertex_vsstore_mutant_${mutant_lower}_tests PRIVATE KYTY_PVVS_MUTANT_${mutant}=1)
endforeach()

# S151: seam mutants of the index descriptor (clamp / tail check switched off in the test file), each must be caught.
foreach(mutant IN ITEMS NO_CLAMP NO_TAIL_CHECK)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(per_vertex_vsstore_test_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/PerVertexVsStoreTests.cpp"
	)
	target_include_directories(per_vertex_vsstore_test_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(per_vertex_vsstore_test_mutant_${mutant_lower}_tests PRIVATE KYTY_PVVS_TEST_MUTANT_${mutant}=1)
endforeach()

add_kyty_full_emulator_test(per_vertex_gpu_execution_tests "${KYTY_TESTS_DIR}/PerVertexGpuExecutionTests.cpp")
# S150: GPU harness of KYTY_PV_VSSTORE (needs MoltenVK through SDL_VULKAN_LIBRARY, opt-in like the one above).
# `per_vertex_vsstore_gpu_tests --dump` prints the emitted SPIR-V without a GPU. The mutant (corner 0/2 swapped in the PS emitter) must fail it.
add_kyty_full_emulator_test(per_vertex_vsstore_gpu_tests "${KYTY_TESTS_DIR}/PerVertexVsStoreGpuTests.cpp")
add_kyty_full_emulator_test(per_vertex_vsstore_gpu_mutant_first_third_tests "${KYTY_TESTS_DIR}/PerVertexVsStoreGpuTests.cpp")
target_compile_definitions(per_vertex_vsstore_gpu_mutant_first_third_tests PRIVATE KYTY_PVVS_MUTANT_FIRST_THIRD=1)
add_kyty_full_emulator_test(per_vertex_vsstore_gpu_mutant_range_tests "${KYTY_TESTS_DIR}/PerVertexVsStoreGpuTests.cpp")
target_compile_definitions(per_vertex_vsstore_gpu_mutant_range_tests PRIVATE KYTY_PVVS_MUTANT_INDEX_RANGE_NO_ROUND=1)
# Keeps device-address buffers alive until queued MoltenVK work completes (the game lost the device
# otherwise). Needs a real Vulkan driver through SDL_VULKAN_LIBRARY, so it stays opt-in.
add_executable(device_address_lifetime_gpu_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/DeviceAddressLifetimeGpuTests.cpp"
)
target_include_directories(device_address_lifetime_gpu_tests PRIVATE ${inc_headers})
target_link_libraries(device_address_lifetime_gpu_tests Vulkan::Headers)
add_executable(push_descriptor_policy_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PushDescriptorPolicyTests.cpp"
)
target_include_directories(push_descriptor_policy_tests PRIVATE ${inc_headers})
target_link_libraries(push_descriptor_policy_tests Vulkan::Headers)

add_executable(scalar_provenance_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ScalarProvenanceTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ReadLaneElimination.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(scalar_provenance_tests fmt::fmt xbyak::xbyak)
target_include_directories(scalar_provenance_tests PRIVATE ${inc_headers})

add_executable(page_manager_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PageManagerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
)
target_link_libraries(page_manager_tests common)
target_include_directories(page_manager_tests PRIVATE ${inc_headers})

add_executable(archive_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ArchiveFileTests.cpp"
)
target_link_libraries(archive_file_tests common ZArchive::zarchive)
target_include_directories(archive_file_tests PRIVATE ${inc_headers})

add_executable(trophy_system_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/TrophySystemTests.cpp"
)
target_link_libraries(trophy_system_tests common)
target_include_directories(trophy_system_tests PRIVATE ${inc_headers})

add_executable(avplayer_file_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AvPlayerFileTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(avplayer_file_tests common FFmpeg::ffmpeg ZArchive::zarchive)
target_include_directories(avplayer_file_tests PRIVATE ${inc_headers})

add_executable(vdecsw_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/VdecswTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/videoDec2Decoder.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(vdecsw_tests common FFmpeg::ffmpeg)
target_include_directories(vdecsw_tests PRIVATE ${inc_headers})

add_executable(bit_array_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BitArrayTests.cpp"
)
target_include_directories(bit_array_tests PRIVATE ${inc_headers})

add_executable(lru_cache_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/LruCacheTests.cpp"
)
target_include_directories(lru_cache_tests PRIVATE ${inc_headers})
add_executable(eager_submit_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_tests PRIVATE ${inc_headers})
# S326 mutants: split with an open pass; split while the GPU is busy; no age threshold. All must be caught (WILL_FAIL).
add_executable(eager_submit_mutant_pass_open_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_pass_open_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_pass_open_tests PRIVATE KYTY_EAGER_MUTANT_PASS_OPEN=1)
add_executable(eager_submit_mutant_busy_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_busy_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_busy_tests PRIVATE KYTY_EAGER_MUTANT_BUSY=1)
add_executable(eager_submit_mutant_no_age_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_no_age_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_no_age_tests PRIVATE KYTY_EAGER_MUTANT_NO_AGE=1)
# S327 mutant: split off an open pass that is not being closed. Must be caught (WILL_FAIL).
add_executable(eager_submit_mutant_no_closing_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_no_closing_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_no_closing_tests PRIVATE KYTY_EAGER_MUTANT_NO_CLOSING=1)
# S328 mutants: mid-pass split in a later buffer; no per-segment limit; clear repeated at the reopen; clock keeps the largest offset. All must be caught (WILL_FAIL).
add_executable(eager_submit_mutant_mid_any_cb_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_mid_any_cb_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_mid_any_cb_tests PRIVATE KYTY_EAGER_MUTANT_MID_ANY_CB=1)
add_executable(eager_submit_mutant_mid_no_limit_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_mid_no_limit_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_mid_no_limit_tests PRIVATE KYTY_EAGER_MUTANT_MID_NO_LIMIT=1)
add_executable(eager_submit_mutant_clear_reopen_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EagerSubmitTests.cpp"
)
target_include_directories(eager_submit_mutant_clear_reopen_tests PRIVATE ${inc_headers})
target_compile_definitions(eager_submit_mutant_clear_reopen_tests PRIVATE KYTY_EAGER_MUTANT_CLEAR_REOPEN=1)
add_executable(gpu_gap_mutant_clock_max_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GpuGapTests.cpp"
)
target_include_directories(gpu_gap_mutant_clock_max_tests PRIVATE ${inc_headers})
target_compile_definitions(gpu_gap_mutant_clock_max_tests PRIVATE KYTY_GAP_MUTANT_CLOCK_MAX=1)
add_executable(gpu_gap_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GpuGapTests.cpp"
)
target_include_directories(gpu_gap_tests PRIVATE ${inc_headers})
# S327 mutant: the new gap is armed before the pending buffers are harvested (loses the gap that is still armed). Must be caught (WILL_FAIL).
add_executable(gpu_gap_mutant_arm_first_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/GpuGapTests.cpp"
)
target_include_directories(gpu_gap_mutant_arm_first_tests PRIVATE ${inc_headers})
target_compile_definitions(gpu_gap_mutant_arm_first_tests PRIVATE KYTY_GAP_MUTANT_ARM_FIRST=1)
add_executable(pad_script_parser_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PadScriptParserTests.cpp"
)
target_include_directories(pad_script_parser_tests PRIVATE ${inc_headers})

add_executable(ime_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImeDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeCommon.cpp"
	"${KYTY_SOURCE_DIR}/libs/imeDialog.cpp"
)
target_include_directories(ime_dialog_tests PRIVATE ${inc_headers})

add_executable(memory_tracker_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/MemoryTrackerTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/pageManager.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/memoryTracker.cpp"
)
target_link_libraries(memory_tracker_tests fmt::fmt common)
target_include_directories(memory_tracker_tests PRIVATE ${inc_headers})

add_executable(shader_vertex_metadata_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ShaderVertexMetadataTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderVertexMetadata.cpp"
)
target_include_directories(shader_vertex_metadata_tests PRIVATE ${inc_headers})

add_executable(resource_materialization_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceMaterializationTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
)
target_link_libraries(resource_materialization_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_materialization_tests PRIVATE ${inc_headers})

add_executable(env_audit_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EnvAuditTests.cpp"
)
target_include_directories(env_audit_tests PRIVATE ${inc_headers})

add_executable(amd_cpu_mode_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AmdCpuModeTests.cpp")
target_include_directories(amd_cpu_mode_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): the header is copied with one edit into the build tree, so the sources carry no test hooks.
# SSE4A_IGNORED / RDPID_IGNORED / CLWB_IGNORED: auto does not look at that CPUID bit; CLI_IGNORED: the flag loses to the variable;
# ZERO_UNPARSED: KYTY_AMD_CPU=0 not understood; OFF_PATCHES: --no-amd-cpu still patches; BAD_ENV_OFF: unparsable variable disables;
# ENV_UNKNOWN: KYTY_AMD_CPU missing from the known variables.
function(add_amd_cpu_mode_mutant name header from to)
	string(TOLOWER ${name} name_lower)
	set(mutant_dir "${CMAKE_CURRENT_BINARY_DIR}/amd_cpu_mutants/${name_lower}")
	file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/${header}" text)
	string(REPLACE "${from}" "${to}" edited "${text}")
	if(edited STREQUAL text)
		message(FATAL_ERROR "amd_cpu_mode mutant ${name}: pattern not found in ${header}")
	endif()
	file(WRITE "${mutant_dir}/${header}" "${edited}")
	add_executable(amd_cpu_mode_mutant_${name_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AmdCpuModeTests.cpp")
	target_include_directories(amd_cpu_mode_mutant_${name_lower}_tests PRIVATE ${inc_headers})
	target_include_directories(amd_cpu_mode_mutant_${name_lower}_tests BEFORE PRIVATE "${mutant_dir}")
endfunction()
add_amd_cpu_mode_mutant(SSE4A_IGNORED common/amdCpuMode.h "add(host.sse4a, \"SSE4a\");" "add(true, \"SSE4a\");")
add_amd_cpu_mode_mutant(RDPID_IGNORED common/amdCpuMode.h "add(host.rdpid, \"RDPID\");" "add(true, \"RDPID\");")
add_amd_cpu_mode_mutant(CLWB_IGNORED common/amdCpuMode.h "add(host.clwb, \"CLWB\");" "add(true, \"CLWB\");")
add_amd_cpu_mode_mutant(CLI_IGNORED common/amdCpuMode.h "if (command_line) return *command_line;" "")
add_amd_cpu_mode_mutant(ZERO_UNPARSED common/amdCpuMode.h "text == \"off\" || text == \"0\"" "text == \"off\"")
add_amd_cpu_mode_mutant(OFF_PATCHES common/amdCpuMode.h "return {false, \"forced off" "return {true, \"forced off")
add_amd_cpu_mode_mutant(BAD_ENV_OFF common/amdCpuMode.h "	return AmdCpuMode::Auto;\n}\n\ninline AmdCpuDecision" "	return AmdCpuMode::Off;\n}\n\ninline AmdCpuDecision")
add_amd_cpu_mode_mutant(ENV_UNKNOWN common/knownEnv.h "\"KYTY_AMD_CPU\", " "")

add_executable(bench_window_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BenchWindowTests.cpp")
target_include_directories(bench_window_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): FULLSCREEN_ZERO_ON: "0" turns it on; OCCLUSION_IGNORED: NSWindow occlusion not used;
# ENV_UNKNOWN: KYTY_BENCH_FULLSCREEN missing from the known variables.
function(add_bench_window_mutant name header from to)
	string(TOLOWER ${name} name_lower)
	set(mutant_dir "${CMAKE_CURRENT_BINARY_DIR}/bench_window_mutants/${name_lower}")
	file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/${header}" text)
	string(REPLACE "${from}" "${to}" edited "${text}")
	if(edited STREQUAL text)
		message(FATAL_ERROR "bench_window mutant ${name}: pattern not found in ${header}")
	endif()
	file(WRITE "${mutant_dir}/${header}" "${edited}")
	add_executable(bench_window_mutant_${name_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BenchWindowTests.cpp")
	target_include_directories(bench_window_mutant_${name_lower}_tests PRIVATE ${inc_headers})
	target_include_directories(bench_window_mutant_${name_lower}_tests BEFORE PRIVATE "${mutant_dir}")
endfunction()
add_bench_window_mutant(FULLSCREEN_ZERO_ON common/benchWindow.h "std::string_view(value) == \"1\"" "std::string_view(value) != \"\"")
add_bench_window_mutant(OCCLUSION_IGNORED common/benchWindow.h " && occlusion_visible != 0" "")
add_bench_window_mutant(ENV_UNKNOWN common/knownEnv.h "\"KYTY_BENCH_FULLSCREEN\"," "")

add_executable(control_file_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/ControlFileTests.cpp")
target_include_directories(control_file_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): numbers are read leniently; `until` is ignored.
foreach(mutant IN ITEMS LAX_NUMBERS NO_UNTIL FORCE_ANY_SLOT DIAG_ANY_SLOT FILTER_IGNORED DIAG_ALL_IGNORED DUMP_ANY_SLOT WATCH_ANY_SLOT)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(control_file_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/ControlFileTests.cpp")
	target_include_directories(control_file_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(control_file_mutant_${mutant_lower}_tests PRIVATE KYTY_CONTROL_MUTANT_${mutant}=1)
endforeach()

add_executable(cp_phases_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CpPhasesTests.cpp")
target_include_directories(cp_phases_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): a phase is not reduced by its children; the Parse phase is never counted.
foreach(mutant IN ITEMS KEEP_CHILDREN DROP_PARSE)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(cp_phases_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CpPhasesTests.cpp")
	target_include_directories(cp_phases_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(cp_phases_mutant_${mutant_lower}_tests PRIVATE KYTY_CPPHASES_MUTANT_${mutant}=1)
endforeach()

add_executable(mem_events_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/MemEventsTests.cpp")
target_include_directories(mem_events_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): an inclusive range end; repeats merge whatever the address.
foreach(mutant IN ITEMS INCLUSIVE_END MERGE_ANY_ADDR)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(mem_events_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/MemEventsTests.cpp")
	target_include_directories(mem_events_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(mem_events_mutant_${mutant_lower}_tests PRIVATE KYTY_MEVENTS_MUTANT_${mutant}=1)
endforeach()

add_executable(rt_dump_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/RtDumpTests.cpp")
target_include_directories(rt_dump_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): dump_rt matches every flip / every pixel shader.
foreach(mutant IN ITEMS IGNORE_FLIP IGNORE_PS NO_ADDR)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(rt_dump_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/RtDumpTests.cpp")
	target_include_directories(rt_dump_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(rt_dump_mutant_${mutant_lower}_tests PRIVATE KYTY_RTDUMP_MUTANT_${mutant}=1)
endforeach()

add_executable(bc7_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/Bc7Tests.cpp")
target_include_directories(bc7_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): the mode 6 endpoints lose their p-bit; every pixel of a partitioned block is in
# subset 0; the anchor index keeps its top bit; mode 1 loses the shared p-bit of the second endpoint.
foreach(mutant IN ITEMS NO_PBIT SUBSET ANCHOR SHARED_PBIT)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(bc7_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/Bc7Tests.cpp")
	target_include_directories(bc7_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(bc7_mutant_${mutant_lower}_tests PRIVATE KYTY_BC7_MUTANT_${mutant}=1)
endforeach()

# BcDetile against the real Standard64KB layout (tile.cpp), forward-tiled with TileGetBlockOffset.
add_executable(bc_dump_tile_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BcDumpTileTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/tile.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_include_directories(bc_dump_tile_tests PRIVATE ${inc_headers})
target_link_libraries(bc_dump_tile_tests common fmt::fmt)
# Mutant (must be caught, WILL_FAIL): the block index swaps row and column.
add_executable(bc_dump_tile_mutant_block_row_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BcDumpTileTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/tile.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_include_directories(bc_dump_tile_mutant_block_row_tests PRIVATE ${inc_headers})
target_link_libraries(bc_dump_tile_mutant_block_row_tests common fmt::fmt)
target_compile_definitions(bc_dump_tile_mutant_block_row_tests PRIVATE KYTY_BCDUMP_MUTANT_BLOCK_ROW=1)
# Mutant (must be caught, WILL_FAIL): every single-tile spec starts at tile 0.
add_executable(bc_dump_tile_mutant_tile0_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/BcDumpTileTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/tile.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
)
target_include_directories(bc_dump_tile_mutant_tile0_tests PRIVATE ${inc_headers})
target_link_libraries(bc_dump_tile_mutant_tile0_tests common fmt::fmt)
target_compile_definitions(bc_dump_tile_mutant_tile0_tests PRIVATE KYTY_BCDUMP_MUTANT_TILE0=1)

add_executable(bc_dump_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BcDumpTests.cpp")
target_include_directories(bc_dump_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): the block index swaps row and column; the diff looks at the first byte only.
foreach(mutant IN ITEMS BLOCK_ROW DIFF_FIRST_BYTE)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(bc_dump_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BcDumpTests.cpp")
	target_include_directories(bc_dump_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(bc_dump_mutant_${mutant_lower}_tests PRIVATE KYTY_BCDUMP_MUTANT_${mutant}=1)
endforeach()

add_executable(diag_tex_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/DiagTexTests.cpp")
target_include_directories(diag_tex_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): zero fraction counts the non-zero texels / the once-per-block guard is gone.
foreach(mutant IN ITEMS NONZERO NO_ONCE HIST_NO_FMT HIST_NO_INDICES)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(diag_tex_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/DiagTexTests.cpp")
	target_include_directories(diag_tex_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(diag_tex_mutant_${mutant_lower}_tests PRIVATE KYTY_DIAGTEX_MUTANT_${mutant}=1)
endforeach()

add_executable(min_lod_view_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/MinLodViewTests.cpp")
target_include_directories(min_lod_view_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): MinLod floors instead of rounding up; the shift is not capped at the last level; RESINFO reports the shifted view.
foreach(mutant IN ITEMS FLOOR NO_CAP RESINFO)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(min_lod_view_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/MinLodViewTests.cpp")
	target_include_directories(min_lod_view_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(min_lod_view_mutant_${mutant_lower}_tests PRIVATE KYTY_MINLOD_MUTANT_${mutant}=1)
endforeach()

add_executable(apple_drop_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AppleDropTests.cpp")
target_include_directories(apple_drop_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): Z clip inverted; depth bounds counted without the guest enabling it; color write counted on a host that has it; LOD bias clamp only on the positive side.
foreach(mutant IN ITEMS INVERT_CLIP ALWAYS_BOUNDS NATIVE_COUNTS BIAS_ABS)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(apple_drop_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AppleDropTests.cpp")
	target_include_directories(apple_drop_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(apple_drop_mutant_${mutant_lower}_tests PRIVATE KYTY_APPLEDROP_MUTANT_${mutant}=1)
endforeach()

add_executable(cp_counters_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CpCountersTests.cpp")
target_include_directories(cp_counters_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): native/non-native capture swapped; clock step back not clamped; any KYTY_DIAG_PROTO_SKIP value turns it on; the switch is ignored.
foreach(mutant IN ITEMS NATIVE_SWAP NO_CLAMP FLAG_ANY SKIP_IGNORED)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(cp_counters_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CpCountersTests.cpp")
	target_include_directories(cp_counters_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(cp_counters_mutant_${mutant_lower}_tests PRIVATE KYTY_CPCOUNTERS_MUTANT_${mutant}=1)
endforeach()

add_executable(depth_ro_union_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/DepthRoUnionTests.cpp")
target_include_directories(depth_ro_union_tests PRIVATE ${inc_headers})
# S316 mutants (each must be caught, WILL_FAIL): dirty ignored; GENERAL / feedback loop counted as read-only.
foreach(mutant IN ITEMS IGNORE_DIRTY ALLOW_GENERAL)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(depth_ro_union_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/DepthRoUnionTests.cpp")
	target_include_directories(depth_ro_union_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(depth_ro_union_mutant_${mutant_lower}_tests PRIVATE KYTY_DEPTH_RO_UNION_MUTANT_${mutant}=1)
endforeach()

add_executable(frame_dump_flips_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/FrameDumpFlipsTests.cpp")
target_include_directories(frame_dump_flips_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): zero kept as a flip; match off by one.
foreach(mutant IN ITEMS KEEP_ZERO OFF_BY_ONE)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(frame_dump_flips_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/FrameDumpFlipsTests.cpp")
	target_include_directories(frame_dump_flips_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(frame_dump_flips_mutant_${mutant_lower}_tests PRIVATE KYTY_FRAMEDUMP_MUTANT_${mutant}=1)
endforeach()

add_executable(signal_census_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/SignalCensusTests.cpp")
target_include_directories(signal_census_tests PRIVATE ${inc_headers})
# S276 mutants (each must be caught, WILL_FAIL): SIGBUS counted as SIGSEGV; every thread counted as main; top list unsorted; repeat hits not accumulated; opcode copy ignores the page end; csv row shifted by one column.
foreach(mutant IN ITEMS SIG_SWAP MAIN_ALWAYS TOP_UNSORTED HIST_COUNT_ONCE NO_PAGE_CLAMP CSV_SHIFTED)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(signal_census_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/SignalCensusTests.cpp")
	target_include_directories(signal_census_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(signal_census_mutant_${mutant_lower}_tests PRIVATE KYTY_SIGCENSUS_MUTANT_${mutant}=1)
endforeach()

add_executable(cs_trace_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CsTraceTests.cpp")
target_include_directories(cs_trace_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): read resources dropped from the trace; traced flip off by one.
foreach(mutant IN ITEMS DROP_READS OFF_BY_ONE)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(cs_trace_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CsTraceTests.cpp")
	target_include_directories(cs_trace_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(cs_trace_mutant_${mutant_lower}_tests PRIVATE KYTY_CSTRACE_MUTANT_${mutant}=1)
endforeach()

add_executable(bvh_node_parser_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BvhNodeParserTests.cpp")
target_include_directories(bvh_node_parser_tests PRIVATE ${inc_headers})
# Mutants (each must be caught, WILL_FAIL): T# base shifted by 6 instead of 8; empty child slots listed.
foreach(mutant IN ITEMS BASE_SHIFT KEEP_EMPTY)
	string(TOLOWER ${mutant} mutant_lower)
	add_executable(bvh_node_parser_mutant_${mutant_lower}_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/BvhNodeParserTests.cpp")
	target_include_directories(bvh_node_parser_mutant_${mutant_lower}_tests PRIVATE ${inc_headers})
	target_compile_definitions(bvh_node_parser_mutant_${mutant_lower}_tests PRIVATE KYTY_BVHPARSER_MUTANT_${mutant}=1)
endforeach()

add_executable(rt_miss_switch_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/RtMissSwitchTests.cpp")
target_include_directories(rt_miss_switch_tests PRIVATE ${inc_headers})
# Mutant (must be caught, WILL_FAIL): the switch is never seen.
add_executable(rt_miss_switch_mutant_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/RtMissSwitchTests.cpp")
target_include_directories(rt_miss_switch_mutant_tests PRIVATE ${inc_headers})
target_compile_definitions(rt_miss_switch_mutant_tests PRIVATE KYTY_RTMISS_MUTANT_IGNORED=1)

# Mutant (must be caught, WILL_FAIL): the diagnostic switches are missing from the known list.
add_executable(env_audit_mutant_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/EnvAuditTests.cpp")
target_include_directories(env_audit_mutant_tests PRIVATE ${inc_headers})
target_compile_definitions(env_audit_mutant_tests PRIVATE KYTY_KNOWNENV_MUTANT_NO_DIAG=1)

add_executable(resource_tracking_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
)
target_link_libraries(resource_tracking_tests fmt::fmt xbyak::xbyak)
target_include_directories(resource_tracking_tests PRIVATE ${inc_headers})

# Mutant (must be caught, WILL_FAIL): BvhIntersect no longer sets ShaderInfo::uses_bvh.
add_executable(resource_tracking_mutant_bvh_flag_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ResourceTrackingTests.cpp"
	"${KYTY_SOURCE_DIR}/graphics/guest_gpu/gpu_format.cpp"
	"${KYTY_SOURCE_DIR}/graphics/host_gpu/hostMemory.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/shaderPixelParameter.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/SrtWalker.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceTracking.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ResourceMaterialization.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/ShaderInfoCollection.cpp"
	"${KYTY_SOURCE_DIR}/graphics/shader/recompiler/ir/passes/BindingLayout.cpp"
)
target_link_libraries(resource_tracking_mutant_bvh_flag_tests fmt::fmt)
target_include_directories(resource_tracking_mutant_bvh_flag_tests PRIVATE ${inc_headers})
target_compile_definitions(resource_tracking_mutant_bvh_flag_tests PRIVATE KYTY_BVHFLAG_MUTANT_NOT_SET=1)

add_executable(audio_out2_port_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/AudioOut2PortTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/libAudio2.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(audio_out2_port_tests common fmt::fmt)
target_include_directories(audio_out2_port_tests PRIVATE ${inc_headers})

add_executable(audio_out_timing_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/AudioOutTimingTests.cpp")
target_link_libraries(audio_out_timing_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(audio_out_timing_tests PRIVATE ${inc_headers})

add_executable(ngs2_sampler_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/Ngs2SamplerTests.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(ngs2_sampler_tests common fmt::fmt LibAtrac9)
target_include_directories(ngs2_sampler_tests PRIVATE ${inc_headers})

add_executable(pad_haptics_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/PadHapticsTests.cpp"
)
target_link_libraries(pad_haptics_tests common fmt::fmt SDL3::SDL3-static opus)
target_include_directories(pad_haptics_tests PRIVATE ${inc_headers})

add_executable(controller_settings_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ControllerSettingsTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/padScript.cpp"
	"${KYTY_SOURCE_DIR}/graphics/presentation/frameTiming.cpp"
)
target_link_libraries(controller_settings_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(controller_settings_tests PRIVATE ${inc_headers})

add_executable(host_input_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/HostInputTests.cpp")
target_link_libraries(host_input_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(host_input_tests PRIVATE ${inc_headers})

add_executable(save_data_memory_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataMemoryTests.cpp"
)
target_link_libraries(save_data_memory_tests common fmt::fmt)
target_include_directories(save_data_memory_tests PRIVATE ${inc_headers})

add_executable(png_enc_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/PngEncTests.cpp")
target_link_libraries(png_enc_tests common fmt::fmt)
target_include_directories(png_enc_tests PRIVATE ${inc_headers})

add_executable(ces_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/CesTests.cpp")
target_link_libraries(ces_tests common fmt::fmt SDL3::SDL3-static)
target_include_directories(ces_tests PRIVATE ${inc_headers})

add_executable(save_data_dialog_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SaveDataDialogTests.cpp"
	"${KYTY_SOURCE_DIR}/libs/dialog.cpp"
)
target_link_libraries(save_data_dialog_tests common fmt::fmt)
target_include_directories(save_data_dialog_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(http_uri_parse_tests
	"${KYTY_TESTS_DIR}/HttpUriParseTests.cpp")

add_executable(event_queue_lifetime_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/EventQueueLifetimeTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/eventQueue.cpp"
	"${KYTY_SOURCE_DIR}/loader/timer.cpp"
)
target_link_libraries(event_queue_lifetime_tests common fmt::fmt)
target_include_directories(event_queue_lifetime_tests PRIVATE ${inc_headers})

add_kyty_full_emulator_test(kernel_file_system_tests
	"${KYTY_TESTS_DIR}/KernelFileSystemTests.cpp")
target_link_libraries(kernel_file_system_tests ZArchive::zarchive)

add_executable(sync_on_address_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/SyncOnAddressTests.cpp"
	"${KYTY_SOURCE_DIR}/kernel/syncOnAddress.cpp"
)
target_link_libraries(sync_on_address_tests common)
target_include_directories(sync_on_address_tests PRIVATE ${inc_headers})

add_executable(image_page_table_tests EXCLUDE_FROM_ALL
	"${KYTY_TESTS_DIR}/ImagePageTableTests.cpp"
)
target_link_libraries(image_page_table_tests common fmt::fmt)
target_include_directories(image_page_table_tests PRIVATE ${inc_headers})

add_executable(gpu_keepwarm_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/GpuKeepWarmTests.cpp")
target_include_directories(gpu_keepwarm_tests PRIVATE ${inc_headers})
include("${KYTY_TESTS_DIR}/GpuKeepWarmMutants.cmake")
if(APPLE)
	# Smoke on the real MTLDevice (opt-in, needs the GPU): spins 3 s, expects cb_n > 0, and that the spin pauses when idle and resumes.
	add_executable(gpu_keepwarm_gpu_tests EXCLUDE_FROM_ALL "${KYTY_TESTS_DIR}/GpuKeepWarmGpuTests.cpp" "${KYTY_KEEPWARM_MM}")
	target_include_directories(gpu_keepwarm_gpu_tests PRIVATE ${inc_headers})
	target_link_libraries(gpu_keepwarm_gpu_tests "-framework Metal" "-framework Foundation")
endif()

add_kyty_full_emulator_test(shader_recompiler_compute_tests "${KYTY_TESTS_DIR}/ShaderRecompilerComputeTests.cpp")

set(gpu_test_generated_dir "${PROJECT_BINARY_DIR}/gpu_test_shaders")
set(gpu_test_ms_depth_source
	"${gpu_tiler_shader_dir}/gpu_test_ms_depth.comp")
set(gpu_test_ms_depth_spv
	"${gpu_test_generated_dir}/gpu_test_ms_depth.spv")
set(gpu_test_ms_depth_header
	"${gpu_test_generated_dir}/gpu_test_ms_depth_spv.h")
add_custom_command(
	OUTPUT "${gpu_test_ms_depth_header}"
	COMMAND ${CMAKE_COMMAND} -E make_directory "${gpu_test_generated_dir}"
	COMMAND "${KYTY_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 -Os
	        -o "${gpu_test_ms_depth_spv}" "${gpu_test_ms_depth_source}"
	COMMAND ${CMAKE_COMMAND} -DINPUT=${gpu_test_ms_depth_spv}
	        -DOUTPUT=${gpu_test_ms_depth_header} -DSYMBOL=GPU_TEST_MS_DEPTH_SPV
	        -P "${KYTY_SOURCE_DIR}/embed_spirv.cmake"
	DEPENDS "${gpu_test_ms_depth_source}"
	VERBATIM
)
target_sources(shader_recompiler_compute_tests PRIVATE
	"${gpu_test_ms_depth_header}")

add_kyty_full_emulator_test(virtual_memory_allocation_tests "${KYTY_TESTS_DIR}/VirtualMemoryAllocationTests.cpp")
target_compile_definitions(virtual_memory_allocation_tests PRIVATE
	KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS=1)

# These tests use exceptions.
if(NOT KYTY_CLANG_CL)
	foreach(kyty_exception_test scalar_provenance_tests resource_tracking_tests
			resource_tracking_mutant_bvh_flag_tests
			virtual_memory_allocation_tests)
		target_compile_options(${kyty_exception_test} PRIVATE -fexceptions)
	endforeach()
endif()

if(BUILD_TESTING)
	add_test(NAME emulator_user_name_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 1234567890123456)
	add_test(NAME emulator_user_name_too_long
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --user-name 12345678901234567)
	set_tests_properties(emulator_user_name_too_long PROPERTIES WILL_FAIL TRUE)
	add_test(NAME emulator_present_mode_cli
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Immediate)
	add_test(NAME emulator_present_mode_invalid
	         COMMAND $<TARGET_FILE:kyty_emulator> --help --present-mode Invalid)
	set_tests_properties(emulator_present_mode_invalid PROPERTIES WILL_FAIL TRUE)
	add_test(NAME ime_dialog COMMAND $<TARGET_FILE:ime_dialog_tests>)
	add_test(NAME shader_cfg COMMAND $<TARGET_FILE:shader_cfg_tests>)
	# The mutant aborts (Check -> abort), which ctest counts as a crash even with WILL_FAIL: require its failure message instead.
	add_test(NAME shader_cfg_pv_vsstore_mutant_first_third
	         COMMAND sh -c "$<TARGET_FILE:shader_cfg_pv_vsstore_mutant_first_third_tests> 2>&1 | grep -q 'first/third swapped'")
	add_test(NAME shader_cfg_pv_vsstore_mutant_stride_dword
	         COMMAND sh -c "$<TARGET_FILE:shader_cfg_pv_vsstore_mutant_stride_dword_tests> 2>&1 | grep -q 'not read from push dword 7'")
	add_test(NAME per_vertex_transform COMMAND $<TARGET_FILE:per_vertex_transform_tests>)
	add_test(NAME per_vertex_unique COMMAND $<TARGET_FILE:per_vertex_unique_tests>)
	add_test(NAME per_vertex_vsstore COMMAND $<TARGET_FILE:per_vertex_vsstore_tests>)
	add_test(NAME push_descriptor_policy COMMAND $<TARGET_FILE:push_descriptor_policy_tests>)
	add_test(NAME scalar_provenance COMMAND $<TARGET_FILE:scalar_provenance_tests>)
	add_test(NAME gpu_keepwarm COMMAND $<TARGET_FILE:gpu_keepwarm_tests>)
	foreach(mutant IN LISTS KYTY_KW_MUTANTS)
		add_test(NAME gpu_keepwarm_mutant_${mutant} COMMAND $<TARGET_FILE:gpu_keepwarm_mutant_${mutant}_tests>)
		set_tests_properties(gpu_keepwarm_mutant_${mutant} PROPERTIES WILL_FAIL TRUE)
	endforeach()
	add_test(NAME image_page_table COMMAND $<TARGET_FILE:image_page_table_tests>)
	add_test(NAME memory_tracker COMMAND $<TARGET_FILE:memory_tracker_tests>)
	add_test(NAME page_manager COMMAND $<TARGET_FILE:page_manager_tests>)
	add_test(NAME archive_file COMMAND $<TARGET_FILE:archive_file_tests>)
	add_test(NAME trophy_system COMMAND $<TARGET_FILE:trophy_system_tests>)
	add_test(NAME avplayer_file COMMAND $<TARGET_FILE:avplayer_file_tests>)
	add_test(NAME vdecsw COMMAND $<TARGET_FILE:vdecsw_tests>)
	add_test(NAME bit_array COMMAND $<TARGET_FILE:bit_array_tests>)
	add_test(NAME lru_cache COMMAND $<TARGET_FILE:lru_cache_tests>)
	add_test(NAME eager_submit COMMAND $<TARGET_FILE:eager_submit_tests>)
	add_test(NAME eager_submit_mutant_pass_open COMMAND $<TARGET_FILE:eager_submit_mutant_pass_open_tests>)
	add_test(NAME eager_submit_mutant_busy COMMAND $<TARGET_FILE:eager_submit_mutant_busy_tests>)
	add_test(NAME eager_submit_mutant_no_age COMMAND $<TARGET_FILE:eager_submit_mutant_no_age_tests>)
	set_tests_properties(eager_submit_mutant_pass_open eager_submit_mutant_busy eager_submit_mutant_no_age PROPERTIES WILL_FAIL TRUE)
	add_test(NAME eager_submit_mutant_no_closing COMMAND $<TARGET_FILE:eager_submit_mutant_no_closing_tests>)
	add_test(NAME gpu_gap COMMAND $<TARGET_FILE:gpu_gap_tests>)
	add_test(NAME gpu_gap_mutant_arm_first COMMAND $<TARGET_FILE:gpu_gap_mutant_arm_first_tests>)
	set_tests_properties(eager_submit_mutant_pass_open eager_submit_mutant_busy eager_submit_mutant_no_age eager_submit_mutant_no_closing gpu_gap_mutant_arm_first PROPERTIES WILL_FAIL TRUE)
	add_test(NAME eager_submit_mutant_mid_any_cb COMMAND $<TARGET_FILE:eager_submit_mutant_mid_any_cb_tests>)
	add_test(NAME eager_submit_mutant_mid_no_limit COMMAND $<TARGET_FILE:eager_submit_mutant_mid_no_limit_tests>)
	add_test(NAME eager_submit_mutant_clear_reopen COMMAND $<TARGET_FILE:eager_submit_mutant_clear_reopen_tests>)
	add_test(NAME gpu_gap_mutant_clock_max COMMAND $<TARGET_FILE:gpu_gap_mutant_clock_max_tests>)
	set_tests_properties(eager_submit_mutant_pass_open eager_submit_mutant_busy eager_submit_mutant_no_age eager_submit_mutant_no_closing gpu_gap_mutant_arm_first eager_submit_mutant_mid_any_cb eager_submit_mutant_mid_no_limit eager_submit_mutant_clear_reopen gpu_gap_mutant_clock_max PROPERTIES WILL_FAIL TRUE)
	add_test(NAME pad_script_parser COMMAND $<TARGET_FILE:pad_script_parser_tests>)
	add_test(NAME shader_vertex_metadata COMMAND $<TARGET_FILE:shader_vertex_metadata_tests>)
	add_test(NAME resource_materialization COMMAND $<TARGET_FILE:resource_materialization_tests>)
	add_test(NAME env_audit COMMAND $<TARGET_FILE:env_audit_tests>)
	add_test(NAME env_audit_mutant COMMAND $<TARGET_FILE:env_audit_mutant_tests>)
	set_tests_properties(env_audit_mutant PROPERTIES WILL_FAIL TRUE)
	add_test(NAME bench_window COMMAND $<TARGET_FILE:bench_window_tests>)
	foreach(mutant IN ITEMS fullscreen_zero_on occlusion_ignored env_unknown)
		add_test(NAME bench_window_mutant_${mutant} COMMAND $<TARGET_FILE:bench_window_mutant_${mutant}_tests>)
		set_tests_properties(bench_window_mutant_${mutant} PROPERTIES WILL_FAIL TRUE)
	endforeach()
	add_test(NAME amd_cpu_mode COMMAND $<TARGET_FILE:amd_cpu_mode_tests>)
	add_test(NAME amd_cpu_mode_mutant_sse4a_ignored COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_sse4a_ignored_tests>)
	set_tests_properties(amd_cpu_mode_mutant_sse4a_ignored PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_rdpid_ignored COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_rdpid_ignored_tests>)
	set_tests_properties(amd_cpu_mode_mutant_rdpid_ignored PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_clwb_ignored COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_clwb_ignored_tests>)
	set_tests_properties(amd_cpu_mode_mutant_clwb_ignored PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_cli_ignored COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_cli_ignored_tests>)
	set_tests_properties(amd_cpu_mode_mutant_cli_ignored PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_zero_unparsed COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_zero_unparsed_tests>)
	set_tests_properties(amd_cpu_mode_mutant_zero_unparsed PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_off_patches COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_off_patches_tests>)
	set_tests_properties(amd_cpu_mode_mutant_off_patches PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_bad_env_off COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_bad_env_off_tests>)
	set_tests_properties(amd_cpu_mode_mutant_bad_env_off PROPERTIES WILL_FAIL TRUE)
	add_test(NAME amd_cpu_mode_mutant_env_unknown COMMAND $<TARGET_FILE:amd_cpu_mode_mutant_env_unknown_tests>)
	set_tests_properties(amd_cpu_mode_mutant_env_unknown PROPERTIES WILL_FAIL TRUE)
	add_test(NAME control_file COMMAND $<TARGET_FILE:control_file_tests>)
	add_test(NAME cs_trace COMMAND $<TARGET_FILE:cs_trace_tests>)
	add_test(NAME rt_dump COMMAND $<TARGET_FILE:rt_dump_tests>)
	add_test(NAME mem_events COMMAND $<TARGET_FILE:mem_events_tests>)
	add_test(NAME cp_phases COMMAND $<TARGET_FILE:cp_phases_tests>)
	add_test(NAME diag_tex COMMAND $<TARGET_FILE:diag_tex_tests>)
	add_test(NAME min_lod_view COMMAND $<TARGET_FILE:min_lod_view_tests>)
	add_test(NAME apple_drop COMMAND $<TARGET_FILE:apple_drop_tests>)
	add_test(NAME cp_counters COMMAND $<TARGET_FILE:cp_counters_tests>)
	add_test(NAME depth_ro_union COMMAND $<TARGET_FILE:depth_ro_union_tests>)
	add_test(NAME frame_dump_flips COMMAND $<TARGET_FILE:frame_dump_flips_tests>)
	add_test(NAME signal_census COMMAND $<TARGET_FILE:signal_census_tests>)
	add_test(NAME bc7 COMMAND $<TARGET_FILE:bc7_tests>)
	add_test(NAME bc_dump COMMAND $<TARGET_FILE:bc_dump_tests>)
	add_test(NAME bc_dump_tile COMMAND $<TARGET_FILE:bc_dump_tile_tests>)
	add_test(NAME bvh_node_parser COMMAND $<TARGET_FILE:bvh_node_parser_tests>)
	add_test(NAME rt_miss_switch COMMAND $<TARGET_FILE:rt_miss_switch_tests>)
	add_test(NAME rt_miss_switch_mutant COMMAND $<TARGET_FILE:rt_miss_switch_mutant_tests>)
	set_tests_properties(rt_miss_switch_mutant PROPERTIES WILL_FAIL TRUE)
	add_test(NAME resource_tracking_mutant_bvh_flag COMMAND $<TARGET_FILE:resource_tracking_mutant_bvh_flag_tests>)
	set_tests_properties(resource_tracking_mutant_bvh_flag PROPERTIES WILL_FAIL TRUE)
	foreach(mutant_test control_file_mutant_lax_numbers control_file_mutant_no_until
			control_file_mutant_force_any_slot control_file_mutant_diag_any_slot
			control_file_mutant_filter_ignored control_file_mutant_diag_all_ignored control_file_mutant_dump_any_slot control_file_mutant_watch_any_slot
			bc7_mutant_no_pbit bc7_mutant_subset bc7_mutant_anchor bc7_mutant_shared_pbit
			bc_dump_mutant_block_row bc_dump_mutant_diff_first_byte bc_dump_tile_mutant_block_row bc_dump_tile_mutant_tile0
			diag_tex_mutant_hist_no_fmt diag_tex_mutant_hist_no_indices rt_dump_mutant_no_addr
			diag_tex_mutant_nonzero diag_tex_mutant_no_once
			cs_trace_mutant_drop_reads cs_trace_mutant_off_by_one
			rt_dump_mutant_ignore_flip rt_dump_mutant_ignore_ps
			bvh_node_parser_mutant_base_shift bvh_node_parser_mutant_keep_empty
			mem_events_mutant_inclusive_end mem_events_mutant_merge_any_addr
			cp_phases_mutant_keep_children cp_phases_mutant_drop_parse
			min_lod_view_mutant_floor min_lod_view_mutant_no_cap min_lod_view_mutant_resinfo
			apple_drop_mutant_invert_clip apple_drop_mutant_always_bounds apple_drop_mutant_native_counts apple_drop_mutant_bias_abs
			cp_counters_mutant_native_swap cp_counters_mutant_no_clamp cp_counters_mutant_flag_any cp_counters_mutant_skip_ignored
			frame_dump_flips_mutant_keep_zero frame_dump_flips_mutant_off_by_one
			gpu_truth_mutant_sum gpu_truth_mutant_no_extend gpu_truth_mutant_no_clamp gpu_truth_mutant_bucket_edge gpu_truth_mutant_site_swap gpu_truth_mutant_cause_swap gpu_truth_mutant_reopen_any gpu_truth_mutant_load_cleared gpu_truth_mutant_every_run gpu_truth_mutant_site_line gpu_truth_mutant_site_no_bytes transit_hoist_mutant_write_elided transit_hoist_mutant_unseen_access depth_layout_mutant_feedback depth_layout_mutant_skip_any submit_causes_mutant_bucket submit_causes_mutant_nowait submit_causes_mutant_lifo submit_causes_mutant_sticky barrier_causes_mutant_order barrier_causes_mutant_waw_attach barrier_causes_mutant_relation depth_ro_union_mutant_ignore_dirty depth_ro_union_mutant_allow_general
			signal_census_mutant_sig_swap signal_census_mutant_main_always signal_census_mutant_top_unsorted signal_census_mutant_hist_count_once signal_census_mutant_no_page_clamp signal_census_mutant_csv_shifted
			per_vertex_unique_mutant_no_instance per_vertex_unique_mutant_no_first per_vertex_unique_mutant_no_ratio
			per_vertex_unique_mutant_writes_ok per_vertex_unique_mutant_dropped_ok per_vertex_unique_mutant_no_dropped
			per_vertex_vsstore_mutant_first_third per_vertex_vsstore_mutant_no_instance per_vertex_vsstore_mutant_no_first per_vertex_vsstore_mutant_half_swap per_vertex_vsstore_mutant_no_first_index per_vertex_vsstore_mutant_no_vertex_offset per_vertex_vsstore_mutant_dropped_ok per_vertex_vsstore_mutant_reuse_region per_vertex_vsstore_mutant_param31_ok per_vertex_vsstore_mutant_push_no_first_instance per_vertex_vsstore_mutant_index_no_adjust per_vertex_vsstore_mutant_pool_reuse_busy per_vertex_vsstore_mutant_no_instanced_class per_vertex_vsstore_mutant_no_touch per_vertex_vsstore_mutant_align_ignored per_vertex_vsstore_mutant_ring_ignored per_vertex_vsstore_mutant_push_always per_vertex_vsstore_test_mutant_no_clamp per_vertex_vsstore_test_mutant_no_tail_check)
		add_test(NAME ${mutant_test} COMMAND $<TARGET_FILE:${mutant_test}_tests>)
		set_tests_properties(${mutant_test} PROPERTIES WILL_FAIL TRUE)
	endforeach()
	add_test(NAME resource_tracking COMMAND $<TARGET_FILE:resource_tracking_tests>)
	add_test(NAME event_queue_lifetime COMMAND $<TARGET_FILE:event_queue_lifetime_tests>)
	add_test(NAME kernel_file_system COMMAND $<TARGET_FILE:kernel_file_system_tests>)
	add_test(NAME sync_on_address COMMAND $<TARGET_FILE:sync_on_address_tests>)
	add_test(NAME audio_out2_port COMMAND $<TARGET_FILE:audio_out2_port_tests>)
	add_test(NAME audio_out_timing COMMAND $<TARGET_FILE:audio_out_timing_tests>)
	add_test(NAME ngs2_sampler COMMAND $<TARGET_FILE:ngs2_sampler_tests>)
	add_test(NAME pad_haptics COMMAND $<TARGET_FILE:pad_haptics_tests>)
	add_test(NAME controller_settings COMMAND $<TARGET_FILE:controller_settings_tests>)
	add_test(NAME host_input COMMAND $<TARGET_FILE:host_input_tests>)
	add_test(NAME save_data_memory COMMAND $<TARGET_FILE:save_data_memory_tests>)
	add_test(NAME png_enc COMMAND $<TARGET_FILE:png_enc_tests>)
	add_test(NAME ces COMMAND $<TARGET_FILE:ces_tests>)
	add_test(NAME save_data_dialog COMMAND $<TARGET_FILE:save_data_dialog_tests>)
	add_test(NAME http_uri_parse COMMAND $<TARGET_FILE:http_uri_parse_tests>)
	add_test(NAME shader_recompiler_compute COMMAND $<TARGET_FILE:shader_recompiler_compute_tests>)
	add_test(NAME shader_recompiler_bvh_spirv_unchanged
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --bvh-spirv-hash-only)
	add_test(NAME shader_recompiler_alignbyte
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --alignbyte-only)
	add_test(NAME graphics_pipeline_rasterization
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --polygon-mode-only)
	add_test(NAME graphics_draw_offsets
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --draw-offset-only)
	add_test(NAME virtual_memory_allocation
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests>)
	add_test(NAME guest_red_zone_patcher
		COMMAND $<TARGET_FILE:virtual_memory_allocation_tests> --red-zone-patcher-only)
	add_test(NAME command_scheduler_timeline
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --scheduler-only)
	add_test(NAME stream_buffer_ring
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --stream-buffer-only)
	add_test(NAME gpu_command_lane
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-command-lane-only)
	add_test(NAME pm4_context_state
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --context-state-only)
	add_test(NAME gpu_tiler
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --gpu-tiler-only)
	add_test(NAME image_transfer_layout
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-layout-only)

	add_test(NAME texture_cache_layered_image
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --layered-image-only)
	add_test(NAME host_image_allocation
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --host-image-allocation-only)
	add_test(NAME texture_cache_image_views
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-view-cache-only)
	add_test(NAME texture_cache_storage_sampled
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --storage-sampled-only)
	add_test(NAME texture_cache_depth_readback
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --depth-readback-only)
	add_test(NAME compute_meta_clear_classification
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --compute-meta-clear-only)
	add_test(NAME buffer_cache_dirty_gc
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-gc-only)
	add_test(NAME color_render_target_1d
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --rt1d-only)

	add_test(NAME texture_cache_image_overlap
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --image-overlap-only)
	add_test(NAME texture_cache_htile_clear
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --htile-clear-only)
	add_test(NAME buffer_cache_ranges
		COMMAND $<TARGET_FILE:shader_recompiler_compute_tests> --buffer-cache-range-only)

	add_custom_target(kyty_tests DEPENDS
		ime_dialog_tests
		shader_cfg_tests
		shader_cfg_pv_vsstore_mutant_first_third_tests
		shader_cfg_pv_vsstore_mutant_stride_dword_tests
		scalar_provenance_tests
		image_page_table_tests
		memory_tracker_tests
		page_manager_tests
		archive_file_tests
		trophy_system_tests
		avplayer_file_tests
		vdecsw_tests
		bit_array_tests
		lru_cache_tests
		eager_submit_tests
		eager_submit_mutant_pass_open_tests
		eager_submit_mutant_busy_tests
		eager_submit_mutant_no_age_tests
		eager_submit_mutant_no_closing_tests
		gpu_gap_tests
		gpu_gap_mutant_arm_first_tests
		eager_submit_mutant_mid_any_cb_tests
		eager_submit_mutant_mid_no_limit_tests
		eager_submit_mutant_clear_reopen_tests
		gpu_gap_mutant_clock_max_tests
		push_descriptor_policy_tests
		pad_script_parser_tests
		shader_vertex_metadata_tests
		resource_materialization_tests
		env_audit_tests
		env_audit_mutant_tests
		amd_cpu_mode_tests
		amd_cpu_mode_mutant_sse4a_ignored_tests
		amd_cpu_mode_mutant_rdpid_ignored_tests
		amd_cpu_mode_mutant_clwb_ignored_tests
		amd_cpu_mode_mutant_cli_ignored_tests
		amd_cpu_mode_mutant_zero_unparsed_tests
		amd_cpu_mode_mutant_off_patches_tests
		amd_cpu_mode_mutant_bad_env_off_tests
		amd_cpu_mode_mutant_env_unknown_tests
		control_file_tests
		control_file_mutant_lax_numbers_tests
		control_file_mutant_no_until_tests
		control_file_mutant_force_any_slot_tests
		control_file_mutant_diag_any_slot_tests
		control_file_mutant_filter_ignored_tests
		control_file_mutant_diag_all_ignored_tests
		bc7_tests
		bc7_mutant_no_pbit_tests
		bc7_mutant_subset_tests
		bc7_mutant_anchor_tests
		bc7_mutant_shared_pbit_tests
		bc_dump_tests
		bc_dump_tile_tests
		bc_dump_tile_mutant_block_row_tests
		bc_dump_tile_mutant_tile0_tests
		bc_dump_mutant_block_row_tests
		bc_dump_mutant_diff_first_byte_tests
		control_file_mutant_dump_any_slot_tests
		control_file_mutant_watch_any_slot_tests
		diag_tex_mutant_hist_no_fmt_tests
		diag_tex_mutant_hist_no_indices_tests
		rt_dump_mutant_no_addr_tests
		diag_tex_tests
		min_lod_view_tests
		min_lod_view_mutant_floor_tests
		min_lod_view_mutant_no_cap_tests
		min_lod_view_mutant_resinfo_tests
		apple_drop_tests
		apple_drop_mutant_invert_clip_tests
		apple_drop_mutant_always_bounds_tests
		apple_drop_mutant_native_counts_tests
		apple_drop_mutant_bias_abs_tests
		per_vertex_unique_tests
		per_vertex_unique_mutant_no_instance_tests
		per_vertex_unique_mutant_no_first_tests
		per_vertex_unique_mutant_no_ratio_tests
		per_vertex_unique_mutant_writes_ok_tests
		per_vertex_unique_mutant_dropped_ok_tests
		per_vertex_unique_mutant_no_dropped_tests
		per_vertex_vsstore_tests
		per_vertex_vsstore_mutant_first_third_tests
		per_vertex_vsstore_mutant_no_instance_tests
		per_vertex_vsstore_mutant_no_first_tests
		per_vertex_vsstore_mutant_half_swap_tests
		per_vertex_vsstore_mutant_no_first_index_tests
		per_vertex_vsstore_mutant_no_vertex_offset_tests
		per_vertex_vsstore_mutant_dropped_ok_tests
		per_vertex_vsstore_mutant_reuse_region_tests
		per_vertex_vsstore_mutant_param31_ok_tests
		per_vertex_vsstore_mutant_push_no_first_instance_tests
		per_vertex_vsstore_mutant_index_no_adjust_tests
		per_vertex_vsstore_mutant_pool_reuse_busy_tests
		per_vertex_vsstore_mutant_no_instanced_class_tests
		per_vertex_vsstore_mutant_no_touch_tests
		per_vertex_vsstore_mutant_align_ignored_tests
		per_vertex_vsstore_mutant_ring_ignored_tests
		per_vertex_vsstore_mutant_push_always_tests
		per_vertex_vsstore_test_mutant_no_clamp_tests
		per_vertex_vsstore_test_mutant_no_tail_check_tests
		cp_counters_tests
		depth_ro_union_tests
		depth_ro_union_mutant_ignore_dirty_tests
		depth_ro_union_mutant_allow_general_tests
		cp_counters_mutant_native_swap_tests
		cp_counters_mutant_no_clamp_tests
		cp_counters_mutant_flag_any_tests
		cp_counters_mutant_skip_ignored_tests
		frame_dump_flips_tests
		frame_dump_flips_mutant_keep_zero_tests
		frame_dump_flips_mutant_off_by_one_tests
		signal_census_tests
		signal_census_mutant_sig_swap_tests
		signal_census_mutant_main_always_tests
		signal_census_mutant_top_unsorted_tests
		signal_census_mutant_hist_count_once_tests
		signal_census_mutant_no_page_clamp_tests
		signal_census_mutant_csv_shifted_tests
		diag_tex_mutant_nonzero_tests
		diag_tex_mutant_no_once_tests
		cp_phases_tests
		cp_phases_mutant_keep_children_tests
		cp_phases_mutant_drop_parse_tests
		mem_events_tests
		mem_events_mutant_inclusive_end_tests
		mem_events_mutant_merge_any_addr_tests
		rt_dump_tests
		rt_dump_mutant_ignore_flip_tests
		rt_dump_mutant_ignore_ps_tests
		cs_trace_tests
		cs_trace_mutant_drop_reads_tests
		cs_trace_mutant_off_by_one_tests
		bvh_node_parser_tests
		bvh_node_parser_mutant_base_shift_tests
		bvh_node_parser_mutant_keep_empty_tests
		rt_miss_switch_tests
		rt_miss_switch_mutant_tests
		resource_tracking_mutant_bvh_flag_tests
		resource_tracking_tests
		event_queue_lifetime_tests
		kernel_file_system_tests
		sync_on_address_tests
		shader_recompiler_compute_tests
		virtual_memory_allocation_tests
		audio_out2_port_tests
		audio_out_timing_tests
		ngs2_sampler_tests
		pad_haptics_tests
		controller_settings_tests
		host_input_tests
		save_data_memory_tests
		png_enc_tests
		ces_tests
		save_data_dialog_tests
		http_uri_parse_tests
	)
	foreach(mutant IN LISTS KYTY_KW_MUTANTS)
		add_dependencies(kyty_tests gpu_keepwarm_mutant_${mutant}_tests)
	endforeach()
endif()

# Test Darwin signal-context restoration for SSE4a instructions.
if(APPLE AND (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64" OR
		(NOT CMAKE_OSX_ARCHITECTURES AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")))
	add_executable(macos_sse4a_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosSse4aTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(macos_sse4a_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(macos_sse4a_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME macos_sse4a COMMAND $<TARGET_FILE:macos_sse4a_tests>)
		set_tests_properties(macos_sse4a PROPERTIES TIMEOUT 15)
		add_dependencies(kyty_tests macos_sse4a_tests)
	endif()
	# S276 mutant (WILL_FAIL): the SSE4a emulator is counted as the generic CPU-extensions one.
	add_executable(macos_sse4a_mutant_sse4a_as_cpuext_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosSse4aTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(macos_sse4a_mutant_sse4a_as_cpuext_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(macos_sse4a_mutant_sse4a_as_cpuext_tests PRIVATE ${inc_headers})
	target_compile_definitions(macos_sse4a_mutant_sse4a_as_cpuext_tests PRIVATE KYTY_X64EMU_MUTANT_SSE4A_AS_CPUEXT=1)
	if(BUILD_TESTING)
		add_test(NAME macos_sse4a_mutant_sse4a_as_cpuext COMMAND $<TARGET_FILE:macos_sse4a_mutant_sse4a_as_cpuext_tests>)
		set_tests_properties(macos_sse4a_mutant_sse4a_as_cpuext PROPERTIES TIMEOUT 15 WILL_FAIL TRUE)
		add_dependencies(kyty_tests macos_sse4a_mutant_sse4a_as_cpuext_tests)
	endif()
	# S280: bit-exact check of the AMD patcher's EXTRQ/INSERTQ/VRSQRTPS replacements, executed on the CPU.
	add_executable(amd_trampoline_bitexact_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/amdTrampolineBitExactTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(amd_trampoline_bitexact_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(amd_trampoline_bitexact_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME amd_trampoline_bitexact COMMAND $<TARGET_FILE:amd_trampoline_bitexact_tests>)
		set_tests_properties(amd_trampoline_bitexact PROPERTIES TIMEOUT 300)
		add_dependencies(kyty_tests amd_trampoline_bitexact_tests)
	endif()
	# S280 mutants of the AMD patcher / SSE4a emulator for amd_trampoline_bitexact (each must fail, WILL_FAIL; the test
	# exits with 1 on a failed check, a signal exit would not be inverted). One or two edits are applied to a copy of
	# src/loader/<file> in the build tree:
	#   EXTRQ_LEN0 (len 0 not 64), EXTRQ_IDX_DROPPED (no shift by idx), EXTRQ_REG_NO_NEG / EXTRQ_REG_SHIFT9 (register form
	#   decodes the controls wrongly), EXTRQ_RDX_CLOBBER / EXTRQ_FLAGS_LOST / EXTRQ_NO_RED_ZONE (trampoline leaks state),
	#   EXTRQ_UPPER_KEPT (upper quadword not cleared), RSQRT_NO_DAZ / RSQRT_MXCSR_LEAK / RSQRT_MUL (VRSQRTPS replacement),
	#   KIND_*_IGNORED (KYTY_AMD_PATCH selection ignored), EMU_* (signal emulator used for INSERTQ).
	function(add_bitexact_mutant name file from to from2 to2)
		string(TOLOWER ${name} name_lower)
		set(mutant_dir "${CMAKE_CURRENT_BINARY_DIR}/amd_bitexact_mutants/${name_lower}")
		file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/loader/${file}" text)
		string(REPLACE "${from}" "${to}" edited "${text}")
		if(edited STREQUAL text)
			message(FATAL_ERROR "bitexact mutant ${name}: pattern not found in ${file}")
		endif()
		if(NOT from2 STREQUAL "")
			set(before "${edited}")
			string(REPLACE "${from2}" "${to2}" edited "${before}")
			if(edited STREQUAL before)
				message(FATAL_ERROR "bitexact mutant ${name}: second pattern not found in ${file}")
			endif()
		endif()
		file(WRITE "${mutant_dir}/${file}" "${edited}")
		if(file STREQUAL "guestInstructionPatcher.cpp")
			set(patcher_src "${mutant_dir}/${file}")
			set(emulator_src "${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp")
		else()
			set(patcher_src "${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp")
			set(emulator_src "${mutant_dir}/${file}")
		endif()
		add_executable(amd_trampoline_bitexact_mutant_${name_lower}_tests EXCLUDE_FROM_ALL
			"${KYTY_TESTS_DIR}/amdTrampolineBitExactTests.cpp" "${emulator_src}" "${patcher_src}")
		target_link_libraries(amd_trampoline_bitexact_mutant_${name_lower}_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
		target_include_directories(amd_trampoline_bitexact_mutant_${name_lower}_tests PRIVATE ${inc_headers})
		if(BUILD_TESTING)
			add_test(NAME amd_trampoline_bitexact_mutant_${name_lower} COMMAND $<TARGET_FILE:amd_trampoline_bitexact_mutant_${name_lower}_tests>)
			set_tests_properties(amd_trampoline_bitexact_mutant_${name_lower} PROPERTIES TIMEOUT 300 WILL_FAIL TRUE)
			add_dependencies(kyty_tests amd_trampoline_bitexact_mutant_${name_lower}_tests)
		endif()
	endfunction()
	add_bitexact_mutant(EXTRQ_LEN0 guestInstructionPatcher.cpp "const u64 mask   = length == 0 ? UINT64_MAX : (u64 {1} << length) - 1;" "const u64 mask   = length == 0 ? 0 : (u64 {1} << length) - 1;" "" "")
	add_bitexact_mutant(EXTRQ_IDX_DROPPED guestInstructionPatcher.cpp "if (index != 0) {" "if (false) {" "" "")
	add_bitexact_mutant(EXTRQ_REG_NO_NEG guestInstructionPatcher.cpp "generator.neg(ecx);" "" "" "")
	add_bitexact_mutant(EXTRQ_REG_SHIFT9 guestInstructionPatcher.cpp "generator.shr(rax, 8);" "generator.shr(rax, 9);" "" "")
	add_bitexact_mutant(EXTRQ_RDX_CLOBBER guestInstructionPatcher.cpp "generator.pop(rdx);" "generator.pop(r11);" "" "")
	add_bitexact_mutant(EXTRQ_FLAGS_LOST guestInstructionPatcher.cpp "generator.popfq();\n\tgenerator.lea(rsp, ptr[rsp + GuestRedZoneSize]);" "generator.add(rsp, 8);\n\tgenerator.lea(rsp, ptr[rsp + GuestRedZoneSize]);" "" "")
	add_bitexact_mutant(EXTRQ_NO_RED_ZONE guestInstructionPatcher.cpp "generator.lea(rsp, ptr[rsp - GuestRedZoneSize]);\n\tgenerator.pushfq();" "generator.pushfq();" "generator.popfq();\n\tgenerator.lea(rsp, ptr[rsp + GuestRedZoneSize]);" "generator.popfq();")
	add_bitexact_mutant(EXTRQ_UPPER_KEPT guestInstructionPatcher.cpp "generator.movq(destination, rax);" "generator.pinsrq(destination, rax, 0);" "" "")
	add_bitexact_mutant(RSQRT_NO_DAZ guestInstructionPatcher.cpp "0x1fc0" "0x1f80" "" "")
	add_bitexact_mutant(RSQRT_MXCSR_LEAK guestInstructionPatcher.cpp "generator.ldmxcsr(ptr[rsp + 32]);" "" "" "")
	add_bitexact_mutant(RSQRT_MUL guestInstructionPatcher.cpp "generator.vdivpd(" "generator.vmulpd(" "" "")
	add_bitexact_mutant(KIND_RSQRT_IGNORED guestInstructionPatcher.cpp "(amd_kinds & Common::AmdPatchRsqrt) != 0 &&" "" "" "")
	add_bitexact_mutant(KIND_SSE4A_IGNORED guestInstructionPatcher.cpp "(amd_kinds & Common::AmdPatchSse4a) != 0 && !host_features.sse4a &&" "!host_features.sse4a &&" "" "")
	add_bitexact_mutant(KIND_RDPID_IGNORED guestInstructionPatcher.cpp "(amd_kinds & Common::AmdPatchCpuExt) != 0 && !host_features.rdpid &&" "!host_features.rdpid &&" "" "")
	add_bitexact_mutant(KIND_CLWB_IGNORED guestInstructionPatcher.cpp "(amd_kinds & Common::AmdPatchCpuExt) != 0 && !host_features.clwb &&" "!host_features.clwb &&" "" "")
	add_bitexact_mutant(EMU_LENGTH_MASK x64InstructionEmulator.cpp "length &= 0x3fu;" "length &= 0x7fu;" "" "")
	add_bitexact_mutant(EMU_INSERTQ_CONTROLS_LOW x64InstructionEmulator.cpp "source[prefix == 0x66 ? 0 : 1]" "source[0]" "" "")
	add_bitexact_mutant(EMU_INDEX_SHIFT9 x64InstructionEmulator.cpp "static_cast<uint8_t>(controls >> 8u)" "static_cast<uint8_t>(controls >> 9u)" "" "")
	add_bitexact_mutant(EMU_INSERT_SWAPPED x64InstructionEmulator.cpp "dest[0] = InsertBitField(dest[0], source[0], length, index);" "dest[0] = InsertBitField(source[0], dest[0], length, index);" "" "")
	# S318: the patcher kind rosettacmp (Rosetta 2 VCMPSS/CMPSS bug), executed on the CPU and checked structurally.
	add_executable(rosetta_cmp_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/rosettaCmpTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
		"${KYTY_SOURCE_DIR}/loader/guestInstructionPatcher.cpp"
	)
	target_link_libraries(rosetta_cmp_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
	target_include_directories(rosetta_cmp_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME rosetta_cmp COMMAND $<TARGET_FILE:rosetta_cmp_tests>)
		set_tests_properties(rosetta_cmp PROPERTIES TIMEOUT 60)
		add_dependencies(kyty_tests rosetta_cmp_tests)
	endif()
	# S318 mutants of the patcher for rosetta_cmp (each must fail, WILL_FAIL; the test exits with 1 on a failed check):
	#   NO_SELF_MOVE (the fix is dropped), ALL_PREDICATES (every VCMPSS is rewritten), LEGACY_FROM_0 (CMPSS 0..2 too),
	#   KIND_IGNORED (KYTY_AMD_PATCH selection ignored), LEGACY_VEX_MOVE (a VEX move after the legacy CMPSS).
	function(add_rosettacmp_mutant name from to)
		string(TOLOWER ${name} name_lower)
		set(mutant_dir "${CMAKE_CURRENT_BINARY_DIR}/rosettacmp_mutants/${name_lower}")
		file(READ "${CMAKE_CURRENT_SOURCE_DIR}/src/loader/guestInstructionPatcher.cpp" text)
		string(REPLACE "${from}" "${to}" edited "${text}")
		if(edited STREQUAL text)
			message(FATAL_ERROR "rosettacmp mutant ${name}: pattern not found")
		endif()
		file(WRITE "${mutant_dir}/guestInstructionPatcher.cpp" "${edited}")
		add_executable(rosetta_cmp_mutant_${name_lower}_tests EXCLUDE_FROM_ALL
			"${KYTY_TESTS_DIR}/rosettaCmpTests.cpp"
			"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
			"${mutant_dir}/guestInstructionPatcher.cpp")
		target_link_libraries(rosetta_cmp_mutant_${name_lower}_tests common fmt::fmt Zydis::Zydis xbyak::xbyak)
		target_include_directories(rosetta_cmp_mutant_${name_lower}_tests PRIVATE ${inc_headers})
		if(BUILD_TESTING)
			add_test(NAME rosetta_cmp_mutant_${name_lower} COMMAND $<TARGET_FILE:rosetta_cmp_mutant_${name_lower}_tests>)
			set_tests_properties(rosetta_cmp_mutant_${name_lower} PROPERTIES TIMEOUT 60 WILL_FAIL TRUE)
			add_dependencies(kyty_tests rosetta_cmp_mutant_${name_lower}_tests)
		endif()
	endfunction()
	add_rosettacmp_mutant(NO_SELF_MOVE "generator.vmovaps(destination, destination);" "")
	add_rosettacmp_mutant(ALL_PREDICATES "return low != 0 && low != 1 && low != 2 && low != 11 && low != 27;" "return true;")
	add_rosettacmp_mutant(LEGACY_FROM_0 "return predicate >= 3 && predicate <= 7;" "return predicate <= 7;")
	add_rosettacmp_mutant(KIND_IGNORED "(amd_kinds & Common::AmdPatchRosettaCmp) != 0 && IsRosettaCompareSite(decoded)" "IsRosettaCompareSite(decoded)")
	add_rosettacmp_mutant(LEGACY_VEX_MOVE "generator.movaps(destination, destination);" "generator.vmovaps(destination, destination);")
	add_executable(macos_wide_store_tests EXCLUDE_FROM_ALL
		"${KYTY_TESTS_DIR}/macosWideStoreTests.cpp"
		"${KYTY_SOURCE_DIR}/loader/x64InstructionEmulator.cpp"
	)
	target_link_libraries(macos_wide_store_tests common fmt::fmt Zydis::Zydis)
	target_include_directories(macos_wide_store_tests PRIVATE ${inc_headers})
	if(BUILD_TESTING)
		add_test(NAME macos_wide_store COMMAND $<TARGET_FILE:macos_wide_store_tests>)
		set_tests_properties(macos_wide_store PROPERTIES TIMEOUT 15)
		add_dependencies(kyty_tests macos_wide_store_tests)
	endif()
endif()

add_executable(kyty_emulator "${KYTY_SOURCE_DIR}/main.cpp" ${kyty_emulator_src})
configure_macos_guest_address_space(kyty_emulator)

set(KYTY_TROPHY_UNLOCK_SOUND "${CMAKE_CURRENT_SOURCE_DIR}/assets/sounds/trophy-unlock.wav")
add_custom_target(kyty_trophy_sound
	COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_TROPHY_UNLOCK_SOUND}"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/sounds/trophy-unlock.wav"
	VERBATIM)
add_dependencies(kyty_emulator kyty_trophy_sound)
install(FILES "${KYTY_TROPHY_UNLOCK_SOUND}" DESTINATION assets/sounds)

add_custom_target(kyty_touch_cursors
	COMMAND ${CMAKE_COMMAND} -E copy_directory
		"${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors"
		"$<TARGET_FILE_DIR:kyty_emulator>/assets/cursors"
	VERBATIM)
add_dependencies(kyty_emulator kyty_touch_cursors)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/assets/cursors" DESTINATION assets)

target_link_libraries(kyty_emulator ${kyty_emulator_link_libraries})
if (WIN32)
	target_link_libraries(kyty_emulator opengl32)
	target_link_libraries(kyty_emulator iphlpapi)
	target_link_libraries(kyty_emulator setupapi)
	target_link_libraries(kyty_emulator ws2_32)
endif()
if (KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator onecore)
endif()
if (CLANG AND NOT KYTY_CLANG_CL)
	target_link_libraries(kyty_emulator pthread)
endif()
# dlopen/dlsym/dladdr for RenderDoc.
target_link_libraries(kyty_emulator ${CMAKE_DL_LIBS})
target_include_directories(kyty_emulator PRIVATE ${inc_headers})

clang_tidy_check(kyty_emulator "" "${check_headers}" "${inc_headers}")

include_what_you_use(kyty_emulator "${inc_headers}")

set(KYTY_EMULATOR_MAP_NAME "kyty_emulator_${KYTY_COMPILER_ID}_${KYTY_LINKER_ID}.map")

set(KYTY_EMULATOR_MAP_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/${KYTY_EMULATOR_MAP_NAME}")
set(KYTY_EMULATOR_PDB_LINK_PATH "${CMAKE_CURRENT_BINARY_DIR}/kyty_emulator.pdb")

if(KYTY_CLANG_CL)
	target_link_options(kyty_emulator PRIVATE
		"/DYNAMICBASE:NO"
		"/DEBUG:FULL"
		"/PDB:${KYTY_EMULATOR_PDB_LINK_PATH}"
		"/lldmap:${KYTY_EMULATOR_MAP_LINK_PATH}"
	)
	add_custom_command(TARGET kyty_emulator POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" $<TARGET_FILE_DIR:kyty_emulator>/libwinpthread-1.dll)
elseif(WIN32 OR LINUX)
	set_target_properties(kyty_emulator PROPERTIES LINK_FLAGS "${KYTY_LD_OPTIONS}")
	# LINK_FLAGS is a space-delimited string: a -Map path containing spaces would
	# be split into bogus arguments, so pass it as a discrete link option instead.
	target_link_options(kyty_emulator PRIVATE "-Wl,-Map=${KYTY_EMULATOR_MAP_LINK_PATH}")
endif()

if(LINUX)
    target_link_options(kyty_emulator PRIVATE
        $<$<CONFIG:Release>:-Wl,--strip-all>
    )
endif()

if(APPLE)
	# The emulator writes x86-64 trampolines/PLT stubs into memory and executes them, and
	# runs guest code under Rosetta. That requires the JIT / unsigned-executable-memory
	# entitlements, so re-sign the binary after every link (an unsigned/relinked binary
	# reverts to the hardened defaults and aborts when it first executes written code).
	add_custom_command(TARGET kyty_emulator POST_BUILD
		COMMAND codesign -s - --force --options runtime
			--entitlements "${KYTY_SOURCE_DIR}/macos_jit.entitlements"
			$<TARGET_FILE:kyty_emulator>
		COMMENT "Codesign kyty_emulator with JIT entitlements (macOS)")
endif()

install(TARGETS kyty_emulator DESTINATION .)
if(KYTY_BUILD_LAUNCHER)
	add_subdirectory("${KYTY_SOURCE_DIR}/launcher" "${CMAKE_CURRENT_BINARY_DIR}/launcher")

	if(LINUX)
		target_link_options(launcher PRIVATE $<$<CONFIG:Release>:-Wl,--strip-all>)
	endif()
endif()
if(KYTY_CLANG_CL)
	install(FILES "${KYTY_THIRD_PARTY_DIR}/winpthread/bin/libwinpthread-1.dll" DESTINATION .)
endif()
