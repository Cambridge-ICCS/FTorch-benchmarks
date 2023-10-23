# CSD3 specific instructions
- The Python module to use should be: `module use /usr/local/software/spack/spack-modules/icelake-python3_10/linux-rocky8-icelake; module load python/3.10.4/gcc/4gjyuriy`
- Construct a virtual environment with this Python, and `pip install torch numpy`.
- Build the code with the Python-3.10 module above loaded but _without_ the virtual environment activated (it will contain a `cmake` that doesn't work).
- Run the code with both the Python-3.10 module and the virtual environment loaded.
- The code should be run so that the input files (`uuu.txt`, `vvv.txt`, etc) are located `../cgdrag_model/uuu.txt` from the current directory; and that the `network_wst.pkl` file is _in_ the current directory.
- So for example I run it from the `pytorch` directory with the command `../build/benchmarker_forpy ../pytorch run_emulator_davenet 5`
