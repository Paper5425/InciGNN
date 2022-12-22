# InciGNN: Incidence Matrix-based Graph Neural Network for Directed Graph Representation Learning
a novel directed graph representation learning framework that could produce node and edge representations concurrently in an unsupervised manner, and generate the graph representations according to the incidence matrices.

Source code for WWW2023 paper submission #5425.

## Installation
* Tested with Python 3.8, PyTorch 1.10.0, and PyTorch Geometric 2.0.3
<br/>(set the CUDA variables in the script)
* Alternatively, install the above and the packages listed in [requirements.txt](requirements.txt)

## Overview
* 'code' <br/> Some of the source code is inherented from [D-VAE](https://github.com/muhanzhang/D-VAE/).

* '/data' <br/> Training datasets for the experiments, including ENAS, BN, and NAS-Bench-101.

## Train InciGNN

* Run `python ./train.py`

Please leave an issue if you have any trouble running the code or suggestions for improvements.