# instructions

* pt2ts.py - use this script to generate the `.pt` saved model in torchscript format
* resnet18.py - this is called from the forpy fortran code `../benchmarker_resnet_forpy.f90`
* resnet_infer_python.py - this can be used to run the model in python
* saved_resnet18_model_cpu.pt - this is the torchscript saved model that is loaded in fortran using the direct coupling method
  in `../benchmarker_resnet_torch.f90`
