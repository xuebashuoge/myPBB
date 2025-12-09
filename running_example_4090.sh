#!/bin/bash

python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt