#!/bin/bash

for chunk in *.smi; do
    sbatch dock.sh "${chunk}.db" $chunk
done