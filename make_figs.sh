#!/bin/bash

for i in {1..8}; do
    if [ $i != 3 ]
    then
        filename="figure_3.$i.ipynb"
        echo "Running $filename..."
        jupyter nbconvert --execute --inplace --allow-errors --ExecutePreprocessor.timeout=-1 "$filename"
    fi
done

echo "Running figure_A.1.ipynb..."
jupyter nbconvert --execute --inplace --allow-errors --ExecutePreprocessor.timeout=-1 "figure_A.1.ipynb"