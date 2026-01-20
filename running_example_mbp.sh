#!/bin/bash
MAX_JOBS=4

cat << EOF | xargs -P $MAX_JOBS -I CMD bash -c "CMD"
python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 10
python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 10
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 10
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 10
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 10
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 10
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 10
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 10

python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 20
python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 20
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 20
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 20
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 20
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 20
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 20
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 20

python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 50
python running_example_4090.py --norm_type frob --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 50
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 50
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 50
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 50
python running_example_4090.py --norm_type frob --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 50
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 1.0 --objective vanilla --epoch 50
python running_example_4090.py --norm_type spec --prior_type rand --channel_type rayleigh_zf --noise_var 0.1 --objective vanilla --epoch 50

EOF
echo "GPU 3: All jobs finished."