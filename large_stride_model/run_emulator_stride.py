"""
Contains all python commands to run StrideNet from Forpy.

It needs in the same directory as `arch_StrideNet.py` which describes the
model architecture.
"""
from torch import no_grad, jit, from_numpy
import arch_stride as m


# Initialize everything
def initialize_ts(*args):
    """
    Initialize a StrideNet model from torchscript.

    """

    (filename,) = args
    model = jit.load(filename)

    return model


# Initialize everything
def initialize():
    """
    Initialize a StrideNet model.

    """

    device_str = "cpu"
    model = m.StrideNet().to(device_str)

    # Load weights and set to evaluation mode.
    model.eval()
    return model


# Compute drag
def compute(*args):
    """
    Run the computation from inputs.

    Takes in input arguments passed from Fortran via Forpy and outputs another Tensor
    Reshaping & porting to torch.tensor type, and applying model.forward is performed.

    Parameters
    __________
    model : nn.Module
        StrideNet model ready to be deployed.
    big_tensor : torch.Tensor
        Large 2D Tensor to operate on

    Returns
    -------
    Y_out :
        Results to be returned to MiMA
    """
    model, big_tensor, Y_out = args

    big_tensor = from_numpy(big_tensor)

    # Apply model.
    with no_grad():
        # Ensure evaluation mode (leave training mode and stop using current batch stats)
        # model.eval()  # Set during initialisation
        assert model.training is False
        temp = model(big_tensor)

    Y_out[:, :] = temp

    return Y_out
