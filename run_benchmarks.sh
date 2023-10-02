#!/usr/bin/env bash

nrun=1000
NDIM=256

for n in {1,4,8};
do
    export OMP_NUM_THREADS=$n
    ./benchmarker_cgdrag_forpy ../pytorch run_emulator_davenet $nrun 10 | tee cgdrag_forpy_$n.out
    ./benchmarker_cgdrag_torch ../pytorch saved_model.pth      $nrun 10 | tee cgdrag_torch_$n.out

    ./benchmarker_resnet_forpy ../resnetmodel resnet18                    $nrun 10 | tee resnet_forpy_$n.out
    ./benchmarker_resnet_torch ../resnetmodel saved_resnet18_model_cpu.pt $nrun 10 | tee resnet_torch_$n.out

    ./benchmarker_large_stride_forpy ../stridemodel run_emulator_stride $nrun $NDIM | tee ls_forpy_$n.out
    ./benchmarker_large_stride_torch ../stridemodel saved_model.pth     $nrun $NDIM | tee ls_torch_$n.out
done
