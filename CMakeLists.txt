cmake_minimum_required(VERSION 3.14)

project(fortran-pytorch-lib-benchmark Fortran)

# Set a custom build type called Profile that is based on RelWithDebInfo
# and populate it with flags to use instrumented profiling (gprof).
# Also check that build type is set, is one of the permitted types,
# and set it to Debug if it's not set.

# Oh CMake.  Somehow these never-before-seen variables are getting set to
# an empty string so this set(...CACHE) doesn't do anything.
if(NOT CMAKE_Fortran_FLAGS_PROFILE)
set(CMAKE_Fortran_FLAGS_PROFILE
    "${CMAKE_Fortran_FLAGS_RELWITHDEBINFO} -pg -fno-omit-frame-pointer"
    CACHE STRING "" FORCE
)
endif()
if(NOT CMAKE_EXE_LINKER_FLAGS_PROFILE)
set(CMAKE_EXE_LINKER_FLAGS_PROFILE
    "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -pg"
    CACHE STRING "" FORCE
)
endif()
if(NOT CMAKE_SHARED_LINKER_FLAGS_PROFILE)
set(CMAKE_SHARED_LINKER_FLAGS_PROFILE
    "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -pg"
    CACHE STRING "" FORCE
)
endif()
if(NOT CMAKE_STATIC_LINKER_FLAGS_PROFILE)
set(CMAKE_STATIC_LINKER_FLAGS_PROFILE
    "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO}"
    CACHE STRING "" FORCE
)
endif()
if(NOT CMAKE_MODULE_LINKER_FLAGS_PROFILE)
set(CMAKE_MODULE_LINKER_FLAGS_PROFILE
    "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} -pg"
    CACHE STRING "" FORCE
)
endif()

get_property(isMultiConfig GLOBAL
	PROPERTY GENERATOR_IS_MULTI_CONFIG
)
if(isMultiConfig)
	if(NOT "Profile" IN_LIST CMAKE_CONFIGURATION_TYPES)
	list(APPEND CMAKE_CONFIGURATION_TYPES Profile)
	endif()
else()
	set(allowedBuildTypes Debug Release RelWithDebInfo Profile)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
		STRINGS "${allowedBuildTypes}"
	)
	if(NOT CMAKE_BUILD_TYPE)
		set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
	elseif(NOT CMAKE_BUILD_TYPE IN_LIST allowedBuildTypes)
		message(FATAL_ERROR "Unknown build type: ${CMAKE_BUILD_TYPE}")
	endif()
endif()

# Benchmarking Torch
find_package(FTorch)
message(STATUS "Building with the Fortran PyTorch coupling")

add_executable(benchmarker_torch benchmarker_torch.f90)
target_link_libraries(benchmarker_torch PRIVATE FTorch::ftorch )
target_sources(benchmarker_torch PRIVATE cg_drag_torch.f90)

# Large stride test
add_executable(benchmarker_large_stride_torch benchmarker_large_stride_torch.f90)
target_link_libraries(benchmarker_large_stride_torch PRIVATE FTorch::ftorch )

# Make sure python present
find_package(Python REQUIRED COMPONENTS Development)


# Benchmarking Forpy
# Get forpy module from github
include(FetchContent)
FetchContent_Declare(
    forpy
    GIT_REPOSITORY https://github.com/ylikx/forpy.git
    GIT_TAG        b4fc550c7282ed0751d6559c4d867a60af3718a7
#    SOURCE_DIR     ${FETCHCONTENT_BASE_DIR}/
    GIT_PROGRESS   ON
    # UPDATE_DISCONNECTED   ON
)
FetchContent_MakeAvailable(forpy)
message(STATUS "Building with the Fortran Forpy coupling")

add_executable(benchmarker_forpy benchmarker_forpy.f90)
target_sources(benchmarker_forpy PRIVATE ${forpy_SOURCE_DIR}/forpy_mod.F90)
target_sources(benchmarker_forpy PRIVATE cg_drag_forpy.f90)
target_link_libraries(benchmarker_forpy PRIVATE Python::Python)
