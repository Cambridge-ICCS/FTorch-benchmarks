#!/usr/bin/env bash

nrun=1000
NDIM=256

python ../pytorch/pt2ts.py
python ../resnetmodel/pt2ts.py
python ../stridemodel/pt2ts.py

for n in {1,4,8};
do
    export OMP_NUM_THREADS=$n

    ./benchmarker_cgdrag_forpy ../pytorch run_emulator_davenet      $nrun 10      | tee cgdrag_forpy_$n.out
    ./benchmarker_cgdrag_torch ../pytorch saved_cgdrag_model_gpu.pt $nrun 10 True | tee cgdrag_torch_$n.out

    ./benchmarker_resnet_forpy ../resnetmodel resnet18                    $nrun 10      | tee resnet_forpy_$n.out
    ./benchmarker_resnet_torch ../resnetmodel saved_resnet18_model_gpu.pt $nrun 10 True | tee resnet_torch_$n.out

    ./benchmarker_large_stride_forpy ../stridemodel run_emulator_stride             $nrun $NDIM      | tee ls_forpy_$n.out
    ./benchmarker_large_stride_torch ../stridemodel saved_large_stride_model_gpu.pt $nrun $NDIM True | tee ls_torch_$n.out
done
