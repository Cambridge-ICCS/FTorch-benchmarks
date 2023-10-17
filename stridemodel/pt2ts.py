"""Load a pytorch model and convert it to TorchScript."""
from typing import Optional
import torch

# FPTLIB-TODO
# Add a module import with your model here:
import run_emulator_stride as res


def script_to_torchscript(
    model: torch.nn.Module, filename: Optional[str] = "scripted_model.pt"
) -> None:
    """
    Save pyTorch model to TorchScript using scripting.

    Parameters
    ----------
    model : torch.NN.Module
        a pyTorch model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    scripted_model = torch.jit.script(model)
    print(scripted_model.code)
    scripted_model.save(filename)


def trace_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filename: Optional[str] = "traced_model.pt",
) -> None:
    """
    Save pyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a pyTorch model
    dummy_input : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    traced_model = torch.jit.trace(model, dummy_input)
    # traced_model.save(filename)
    frozen_model = torch.jit.freeze(traced_model)
    ## print(frozen_model.graph)
    ## print(frozen_model.code)
    frozen_model.save(filename)


def load_torchscript(filename: Optional[str] = "saved_model.pth") -> torch.nn.Module:
    """
    Load a TorchScript from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    model = torch.jit.load(filename)

    return model


if __name__ == "__main__":
    # FPTLIB-TODO
    # Load a pre-trained PyTorch model
    # Insert code here to load your model from file as `trained_model`:
    trained_model = res.initialize()

    # Switch-off some specific layers/parts of the model that behave
    # differently during training and inference.
    # This may have been done by the user already, so just make sure here.
    trained_model.eval()

    # FPTLIB-TODO
    # Generate a dummy input Tensor `trained_model_dummy_input` to the model of appropriate size.
    # trained_model_dummy_input = torch.ones((512, 42))
    trained_model_dummy_input = torch.ones((512, 512), dtype=torch.float64)

    # FPTLIB-TODO
    # Set the name of the file you want to save the torchscript model to
    saved_ts_filename = "../stridemodel/saved_large_stride_model_cpu.pt"

    # FPTLIB-TODO
    # If you want to save for inference on GPU uncomment the following 5 lines:
    # device = torch.device('cuda')
    # trained_model = trained_model.to(device)
    # trained_model.eval()
    # trained_model_dummy_input = trained_model_dummy_input.to(device)
    # saved_ts_filename = "../stridemodel/saved_large_stride_model_gpu.pt"

    # Run model over dummy input
    # If something isn't working This will generate an error
    trained_model_dummy_output = trained_model(trained_model_dummy_input)

    # FPTLIB-TODO
    # Save the pytorch model using either scripting (recommended where possible) or tracing
    # -----------
    # Scripting
    # -----------
    script_to_torchscript(trained_model, filename=saved_ts_filename)

    # -----------
    # Tracing
    # -----------
    # trace_to_torchscript(trained_model, trained_model_dummy_input, filename=saved_ts_filename)

    # Load torchscript and run model as a test
    testing_input = 2.0 * trained_model_dummy_input
    trained_model_testing_output = trained_model(testing_input)
    ts_model = load_torchscript(filename=saved_ts_filename)
    ts_model_output = ts_model(testing_input)

    if torch.all(ts_model_output.eq(trained_model_testing_output)):
        print("Saved TorchScript model working as expected in a basic test.")
        print("Users should perform further validation as appropriate.")
    else:
        raise RuntimeError(
            "Saved Torchscript model is not performing as expected.\n"
            "Consider using scripting if you used tracing, or investigate further."
        )
