"""Script to test davenet NN"""
import numpy as np
import run_emulator_davenet as red


IMAX = 128
NUM_COL = 4

wind = np.random.randn(IMAX, NUM_COL, 40)
lat = np.random.randn(NUM_COL)
ps = np.random.randn(IMAX, NUM_COL)
Y_out = np.zeros((IMAX, NUM_COL, 40))

# Initialise and run the model
model = red.initialize()
Y_out = red.compute_reshape_drag(model, wind, lat, ps, Y_out, NUM_COL)
