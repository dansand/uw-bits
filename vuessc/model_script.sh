#!/bin/bash
for i in $(seq 3 0.2 5.2)
do
    mpirun -np 4 python models-A.py $i
#    echo $i
done
