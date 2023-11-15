#!/usr/bin/env bash

nrun=10000

for n in {1,8};
do
    export OMP_NUM_THREADS=$n

    date;/usr/bin/time -v ./benchmarker_cgdrag_forpy ../cgdrag_model run_emulator_davenet      $nrun 10 --alloc_in_loop                               | tee cgdrag_forpy_cpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_forpy ../cgdrag_model run_emulator_davenet      $nrun 10 --alloc_in_loop --use_cuda                    | tee cgdrag_forpy_gpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_torch ../cgdrag_model saved_cgdrag_model_cpu.pt $nrun 10                                               | tee cgdrag_torch_implicit_cpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_torch ../cgdrag_model saved_cgdrag_model_gpu.pt $nrun 10 --alloc_in_loop --explicit_reshape --use_cuda | tee cgdrag_torch_explicit_gpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_torch ../cgdrag_model saved_cgdrag_model_gpu.pt $nrun 10 --use_cuda                                    | tee cgdrag_torch_implicit_gpu_$n.out;date

    date;/usr/bin/time -v ./benchmarker_resnet_forpy ../resnet_model resnet18                    $nrun 10            | tee resnet_forpy_cpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_resnet_forpy ../resnet_model resnet18                    $nrun 10 --use_cuda | tee resnet_forpy_gpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_resnet_torch ../resnet_model saved_resnet18_model_cpu.pt $nrun 10            | tee resnet_torch_cpu_$n.out;date
    date;/usr/bin/time -v ./benchmarker_resnet_torch ../resnet_model saved_resnet18_model_gpu.pt $nrun 10 --use_cuda | tee resnet_torch_gpu_$n.out;date

done
