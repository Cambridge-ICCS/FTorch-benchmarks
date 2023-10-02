"""Load and run pretrained ResNet-18 from TorchVision."""

import torch
import torch.nn.functional as F
import torchvision
from torch import load, device, no_grad, reshape, zeros, tensor, float64, jit, from_numpy


# Initialize everything
def initialize_ts(*args):
    """
    Initialize a StrideNet model from torchscript.

    """

    filename, = args
    model = jit.load(filename)

    return model


# Initialize everything
def initialize():
    """
    Download pre-trained ResNet-18 model and prepare for inference.

    Returns
    -------
    model : torch.nn.Module
    """

    # Load a pre-trained PyTorch model
    print("Loading pre-trained ResNet-18 model...", end="")
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    print("done.")

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference
    model.eval()

    return model


def run_model(model):
    """
    Run the pre-trained ResNet-18 with dummy input of ones.

    Parameters
    ----------
    model : torch.nn.Module
    """

    print("Running ResNet-18 model for ones...", end="")
    dummy_input = torch.ones(1, 3, 224, 224)
    output = model(dummy_input)
    top5 = F.softmax(output, dim=1).topk(5).indices
    print("done.")

    print(f"Top 5 results:\n  {top5}")


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
    BigTensor : torch.Tensor
        Large 2D Tensor to operate on

    Returns
    -------
    Y_out :
        Results to be returned to MiMA
    """
    model, BigTensor, Y_out = args

    BigTensor = from_numpy(BigTensor)

    # Apply model.
    with no_grad():
        # Ensure evaluation mode (leave training mode and stop using current batch stats)
        # model.eval()  # Set during initialisation
        assert model.training is False
        temp = model(BigTensor)

    Y_out[:, :] = temp

    return Y_out


if __name__ == "__main__":
    rn_model = initialize()
    run_model(rn_model)
