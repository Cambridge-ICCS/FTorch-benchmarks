project(simplenet_fortran_smartsim)

cmake_minimum_required(VERSION 3.13)
enable_language(Fortran)
SET(CMAKE_FC_COMPILER gfortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

if(NOT DEFINED SMARTREDIS_INSTALL_PATH)
  set(SMARTREDIS_INSTALL_PATH "/home/joe/software/tools/smartredis/install")
endif()

# Specify all pre-processor and library dependencies
find_library(SMARTREDIS_LIBRARY smartredis PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
find_library(SMARTREDIS_FORTRAN_LIBRARY smartredis-fortran PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
# find_library(HIREDIS hiredis PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
# find_library(REDISPP redis++ PATHS ${SMARTREDIS_INSTALL_PATH}/lib NO_DEFAULT_PATH REQUIRED)
set(SMARTREDIS_LIBRARIES
        ${SMARTREDIS_LIBRARY}
        ${SMARTREDIS_FORTRAN_LIBRARY}
        # ${HIREDIS}
        # ${REDISPP}
)

include_directories(SYSTEM
    # /usr/local/include
    ${SMARTREDIS_INSTALL_PATH}/include
)

# Build executables
add_executable(simplenet_fortran_smartsim simplenet_fortran_smartsim.f90)
target_link_libraries(simplenet_fortran_smartsim ${SMARTREDIS_LIBRARIES})

# TODO: Avoid export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/joe/software/tools/smartredis/install/lib with rpath setting
