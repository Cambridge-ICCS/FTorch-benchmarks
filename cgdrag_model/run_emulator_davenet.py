"""
Contains all python commands MiMA will use.

It needs in the same directory as `arch_DaveNet.py` which describes the
model architecture, and `network_wst.pkl` which contains the model weights.
"""
from torch import load, device, no_grad, tensor, float64, jit, device
import arch_davenet as m


# Initialize everything using torchscript
def initialize_ts(*args):
    """
    Initialize a WaveNet model from torchscript.

    """

    (filename,) = args
    model = jit.load(filename)

    return model


# Initialize everything
def initialize(
    path_weights_stats="../cgdrag_model/network_wst.pkl",
    device_str: str = "cpu",
):
    """
    Initialize a WaveNet model and load weights.

    Parameters
    __________
    path_weights_stats : str
        Path to a pickled object that contains weights and statistics (means and stds).
    device_str : str
        Device to load model. Must be either "cpu" or "cuda".

    """

    checkpoint = load(path_weights_stats, map_location=device(device_str))
    model = m.WaveNet(checkpoint).to(device_str)

    # Load weights and set to evaluation mode.
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    return model


# Compute drag
def compute_reshape_drag(*args):
    """
    Compute the drag from inputs using a neural net.

    Takes in input arguments passed from MiMA and outputs drag in shape desired by MiMA.
    Reshaping & porting to torch.tensor type, and applying model.forward is performed.

    Parameters
    __________
    model : nn.Module
        WaveNet model ready to be deployed.
    wind :
        U or V (128, num_col, 40)
    lat :
        latitudes (num_col)
    p_surf :
        surface pressure (128, num_col)
    Y_out :
        output prellocated in MiMA (128, num_col, 40)
    num_col :
        # of latitudes on this proc
    device : str
        Device to move input_batch to. "cpu" (default) or "cuda".

    Returns
    -------
    Y_out :
        Results to be returned to MiMA
    """
    model, wind, lat, p_surf, Y_out, num_col, input_device = args
    input_device = device(input_device)

    # Reshape and put all input variables together [wind, lat, p_surf]
    wind_T = tensor(wind).to(input_device)

    # lat_T = zeros((imax * num_col, 1), dtype=float64)
    lat_T = tensor(lat).to(input_device)

    # pressure_T = zeros((imax * num_col, 1), dtype=float64)
    pressure_T = tensor(p_surf).to(input_device)

    # Apply model.
    with no_grad():
        # Ensure evaluation mode (leave training mode and stop using current batch stats)
        # model.eval()  # Set during initialisation
        assert model.training is False
        temp = model(wind_T, pressure_T, lat_T)

    # Place in output array for MiMA.
    Y_out[:, :] = temp.cpu()
    del temp

    return Y_out
