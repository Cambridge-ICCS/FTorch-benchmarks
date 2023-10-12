"""Script to test davenet NN"""
import numpy as np
import run_emulator_davenet as red
import run_emulator_davenet_orig as redo


IMAX = 128
NUM_COL = 4


wind = np.random.randn(IMAX, NUM_COL, 40)
lat = np.random.randn(NUM_COL)
lat_long = np.tile(lat.T, (IMAX, 1))
ps = np.random.randn(IMAX, NUM_COL)
Y_out = np.zeros((IMAX, NUM_COL, 40))

# Initialise and run the model
model = red.initialize()
Y_out = red.compute_reshape_drag(model, wind, lat_long, ps, Y_out, NUM_COL)

model_o = redo.initialize()
Y_out_o = redo.compute_reshape_drag(model_o, wind, lat, ps, Y_out, NUM_COL)

print(np.array_equal(Y_out, Y_out_o))

