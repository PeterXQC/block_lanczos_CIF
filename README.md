# A posteriori error bounds for the block-Lanczos method for matrix function approximation

This repo contains all that is necessary to reproduce numerical experiments in the paper. Note that figure 3.3 is a visualization of the Pac-Man contour and is not related to any numerical experiments. Thus, there is no code related to figure 3.3 in this repo. 

## List of Contents

- `block_methods/`: functions used in figure generation
- `data/`: data generated from experiments to create each figure
- `imgs/`: generated figures
- `matrices/` matrices used to create each figure

## `matrices/`

There is only one matrix in `matrices/`, which is used to generate figure 3.6. This matrix is obtained from [Matrix Market](https://math.nist.gov/MatrixMarket/) under the directory `by collection/Independent Sets and Generators/QCD/`.

## Usage

To generate figures:
- Run `make_figs.py` (this script runs each notebook in series, and can be very time consuming).
- Run each notebook manually (possibly very time consuming).
- Run each notebook manually from loaded data.

## Special Thanks

Many thanks to [Tyler Chen](https://chen.pw/)(NYU) for allowing this project to happen, and for the immense support along the way.
