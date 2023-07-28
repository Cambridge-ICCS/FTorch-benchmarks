"""Script to test stridenet"""
import numpy as np
import run_emulator_stride as res


IMAX = 128
JMAX = 128

BigTensor = np.ones((IMAX, JMAX))
Y_out = np.zeros((IMAX, JMAX))

# Initialise and run the model
model = res.initialize()
_ = res.compute(model, BigTensor, Y_out)

print("Y_out = \n", Y_out)
