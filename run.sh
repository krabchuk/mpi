#!/bin/bash

for ((i = 2; i < 20; i++))
do 
    for ((j = 1; j <= i; j++))
    do
        for ((p = 1; p <= 3; p++))
        do
            echo "$i $j $p" && mpirun -n $p ./a.out $i $j
        done
    done
done		
