#!/bin/bash

source ~/.virtualenvs/smartsim/bin/activate

BUILD_DIR="$(pwd)/build"
rm -rf "${BUILD_DIR}"

FTORCH_BUILD_DIR="${SOFTWARE}/ftorch/build"
SMARTREDIS_BUILD_DIR="${SOFTWARE}/tools/smartredis/install"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${FTORCH_BUILD_DIR}/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${SMARTREDIS_BUILD_DIR}/lib"

cmake -S . -B "${BUILD_DIR}" -DCMAKE_PREFIX_PATH="${FTORCH_BUILD_DIR};${SMARTREDIS_BUILD_DIR}"
cmake --build "${BUILD_DIR}"
