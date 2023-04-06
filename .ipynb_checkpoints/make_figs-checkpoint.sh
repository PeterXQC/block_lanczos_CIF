#!/bin/bash

for i in {1..8}; do
    filename="figure_3.$i.ipynb"
    echo "Running $filename..."
    jupyter nbconvert --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 --execute "$filename"
    
jupyter nbconvert --to notebook --inplace --allow-errors --ExecutePreprocessor.timeout=-1 --execute "figure_A.1.ipynb"
done