"""
Contains all python commands MiMA will use.

It needs in the same directory as `arch_DaveNet.py` which describes the
model architecture, and `network_wst.pkl` which contains the model weights.
"""
from torch import load, device, no_grad, reshape, zeros, tensor, float64
import arch_davenet_orig as m


# Initialize everything
def initialize(path_weights_stats="network_wst.pkl"):
    """
    Initialize a WaveNet model and load weights.

    Parameters
    __________
    path_weights_stats : pickled object that contains weights and statistics (means and stds).

    """

    device_str = "cpu"
    checkpoint = load(path_weights_stats, map_location=device(device_str))
    model = m.WaveNet(checkpoint).to(device_str)

    # Load weights and set to evaluation mode.
    model.load_state_dict(checkpoint["weights"])
    model.eval()
    del checkpoint
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

    Returns
    -------
    Y_out :
        Results to be returned to MiMA
    """
    model, wind, lat, p_surf, Y_out, num_col = args
    imax = 128

    # Reshape and put all input variables together [wind, lat, p_surf]
    X = zeros((imax * num_col, 42), dtype=float64)
    X[:, :40] = reshape(
        tensor(wind), (imax * num_col, 40)
    )  # wind[i,j,:] is now at X[i*num_col+j,:40]

    for i in range(num_col):
        X[i::num_col, 40] = lat[i]  # lat[j] is at X[j::num_col,40].

    X[:, 41] = reshape(
        tensor(p_surf), (imax * num_col,)
    )  # p_surf[i,j] is now at X[i*num_col+j,41].

    # Apply model.
    with no_grad():
        # Ensure evaluation mode (leave training mode and stop using current batch stats)
        # model.eval()  # Set during initialisation
        assert model.training is False
        temp = model(X)

    # Reshape into what MiMA needs.
    # Y_out[i,j,:] was temp[i*num_col+j,:].
    Y_out[:, :, :] = reshape(temp, (imax, num_col, 40))
    del temp
    return Y_out
