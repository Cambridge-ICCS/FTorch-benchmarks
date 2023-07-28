"""Module defining the pytorch WaveNet architecture for coupling to MiMA. """
import torch
from torch import nn


class StrideNet(nn.Module):
    """PyTorch Model performing a large array operation."""

    def __init__(
        self,
    ) -> None:
        """
        Initialize a StrideNet model.

        Parameters
        ----------
        """

        super().__init__()

    def forward(self, BigTensor: torch.Tensor) -> torch.Tensor:
        """
        Apply the network to a Big 2D `Tensor` of input features.

        Parameters
        ----------
        BigTensor : torch.Tensor
            Large 2D Tensor.

        Returns
        -------
        output : torch.Tensor
            Tensor of predicted outputs.

        """

        tensor_shape = BigTensor.shape

        Y = torch.zeros(tensor_shape)

        for i in range(tensor_shape[0]):
            for j in range(tensor_shape[1]):
                Y[i, j] = 2.0 * BigTensor[i, j]

        # negate first off-diagonal element
        # (this is to deliberately break the symmetry of the operation)
        Y[0, 1] = -1.0*Y[0, 1]

        return Y
