# Benchmark the fortran to pytorch coupling library

Recent results using
[MiMA](https://github.com/DataWaveProject/MiMA-machine-learning) showed a
moderate _slowdown_ when using our Fortran to PyTorch direct coupling library.
This repository contains code to investigate that surprising result.

## Requirements
It is recommended to use Linux for this. Follow the build instructions below to set up the following requirements:


0) [Fortran-PyTorch-lib](https://github.com/Cambridge-ICCS/fortran-pytorch-lib) repository.
1) CMake >= 3.14
2) Python
3) a virtual environment with PyTorch and NumPy installed

## Build instructions
Get the [fortran-PyTorch-lib](https://github.com/Cambridge-ICCS/fortran-pytorch-lib) repository if you haven't already got it. Follow the installation instructions on the ReadMe of that repository. 

In your fortran-pytorch-lib-benchmark repository, create a build directory and run `cmake` as follows, noting the cmake options below:

```
mkdir build
cd build
cmake ..
```
You may need to specify the path to PyTorch with the option `-DCMAKE_PREFIX-PATH=<full-path-to-PyTorch>`. 

You may also need to specify the path to the CMake library files by using the option `-DFTorch_DIR=<full-path-to-cmake-lib-files>/lib/cmake/`. This is the location you specified when installing fortran-pytorch-lib, if you used `-DCMAKE_INSTALL_PREFIX`. The default location is /usr/local. 

You can specify `cmake` options like `-DCMAKE_BUILD_TYPE=RelWithDebInfo`
or `-DCMAKE_Fortran_COMPILER=ifort` if you need.

Then, run `make` and the program(s) should build.

### Profiling build type
There is now a custom build type for `cmake` you can set with
`-DCMAKE_BUILD_TYPE=Profile`.  This will add the necessary options
for instrumented profiling, in the `gprof` style.

## Running instructions
You'll need the Python virtual environment loaded.  Run the program:
```
benchmarker_forpy <path-to-model-dir> <python-module-to-load> <N>
```
Where `<path-to-model-dir>` is the path to where the PyTorch model resides,
`<python-module-to-load>` is the Python file to load.  It should export
`initialize` and `compute_reshape_drag` methods.  See the Wavenet 2 model
provided.  `<N>` is the number of times to run the inference.

### Profiling using `-DCMAKE_BUILD_TYPE=Profile`
Run `cmake` with the `-DCMAKE_BUILD_TYPE=Profile` option and make the code.
Then, after you run it, a `gmon.out` file will be created in the current
directory.  To process this file you must do:
```
gprof <path-to-benchmarker_forpy> <path-to-gmon.out>
```
By default this gets you the default flat profile followed by the call
graph.  Check the other options to `gprof`.
