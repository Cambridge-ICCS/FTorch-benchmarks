#!/usr/bin/env bash

nrun=1000

for n in {1,4,8};
do
    export OMP_NUM_THREADS=$n

    date;/usr/bin/time -v ./benchmarker_cgdrag_forpy ../cgdrag_model run_emulator_davenet      $nrun 10                    | tee cgdrag_forpy_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_torch ../cgdrag_model saved_cgdrag_model_cpu.pt $nrun 10 --explicit_reshape | tee cgdrag_torch_explicit_$n.out;date
    date;/usr/bin/time -v ./benchmarker_cgdrag_torch ../cgdrag_model saved_cgdrag_model_cpu.pt $nrun 10                    | tee cgdrag_torch_implicit_$n.out;date

    date;/usr/bin/time -v ./benchmarker_resnet_forpy ../resnet_model resnet18                    $nrun 10 | tee resnet_forpy_$n.out;date
    date;/usr/bin/time -v ./benchmarker_resnet_torch ../resnet_model saved_resnet18_model_cpu.pt $nrun 10 | tee resnet_torch_$n.out;date

done
