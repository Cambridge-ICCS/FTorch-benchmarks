cmake_minimum_required(VERSION 3.14)
project(fortran-pytorch-lib-benchmark Fortran)

#find_package(FTorch)

find_package(Python REQUIRED COMPONENTS Development)
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

add_executable(benchmarker_forpy benchmarker_forpy.f90)
target_sources(benchmarker_forpy PRIVATE ${forpy_SOURCE_DIR}/forpy_mod.F90)
target_sources(benchmarker_forpy PRIVATE cg_drag_forpy.f90)
target_link_libraries(benchmarker_forpy PRIVATE Python::Python)
