"""Script to test stridenet"""
import numpy as np
import run_emulator_stride as res


IMAX = 128
JMAX = 128

BigTensor = np.ones((IMAX, JMAX))

# Initialise and run the model
model = res.initialize()
Y_out = res.compute(model, BigTensor)
