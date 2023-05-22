Code from Dave Connelly and Minah Yang.
Pytorch neural net for gravity wave parameterization.

There are 3 relevant files:
1. arch_DaveNet.py : This defines the architecture.
2. network_wst.pkl : When loaded via torch, yields a dictionary that contains weights (trained model parameters), means (40-length vector mean of the output), and stds (40-length vector standard deviations of the output)
3. run_emulator_DaveNet.py : Contains python defs MiMA would call.

I guess 1 & 3 can be combined into one script. I am also attaching a test python script that I used for debugging (test_coupling.py) The two calls at the end of this file is what MiMA needs.
