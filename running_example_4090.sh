#!/bin/bash








#!/bin/bash

# ====================================================
# CONFIGURATION
# ====================================================
# How many scripts to run AT THE SAME TIME on a single GPU?
# Set this based on your VRAM.
# e.g., If each script takes 4GB, a 24GB 4090 can handle ~5-6 jobs safely.
MAX_JOBS=7 

echo "Starting execution with a concurrency limit of $MAX_JOBS jobs per GPU."

# ====================================================
# GPU 2 QUEUE
# Processing all 'BEC' Channel tasks (Mixed Frob & Spec)
# ====================================================
(
    echo "GPU 2: Queuing jobs..."
    # We pipe this list of commands into xargs
    # -P $MAX_JOBS tells it to run that many lines in parallel
    cat << EOF | xargs -P $MAX_JOBS -I CMD bash -c "CMD"
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.5
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.5
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.5 --epochs 20
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.5 --epochs 20
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.5 --epochs 50
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.5 --epochs 50
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.5
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.5
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.5 --epochs 20
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.5 --epochs 20
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.5 --epochs 50
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.5 --epochs 50
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.1
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.1
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.1 --epochs 20
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.1 --epochs 20
python running_example_4090.py --gpu 2 --norm_type frob --prior_type learnt --channel_type bec --outage 0.1 --epochs 50
python running_example_4090.py --gpu 2 --norm_type spec --prior_type learnt --channel_type bec --outage 0.1 --epochs 50
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.1
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.1
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.1 --epochs 20
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.1 --epochs 20
python running_example_4090.py --gpu 2 --norm_type frob --prior_type rand --channel_type bec --outage 0.1 --epochs 50
python running_example_4090.py --gpu 2 --norm_type spec --prior_type rand --channel_type bec --outage 0.1 --epochs 50
EOF
    echo "GPU 2: All jobs finished."
) &
PID_GPU2=$!

# ====================================================
# GPU 3 QUEUE
# Processing all 'Rayleigh' Channel tasks (Mixed Frob & Spec)
# ====================================================
(
    echo "GPU 3: Queuing jobs..."
    cat << EOF | xargs -P $MAX_JOBS -I CMD bash -c "CMD"
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 0.1
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 0.1
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 0.1 --epochs 20
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 0.1 --epochs 20
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 0.1 --epochs 50
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 0.1 --epochs 50
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 0.1
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 0.1
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 0.1 --epochs 20
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 0.1 --epochs 20
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 0.1 --epochs 50
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 0.1 --epochs 50
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 1.0
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 1.0
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 1.0 --epochs 20
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 1.0 --epochs 20
python running_example_4090.py --gpu 3 --norm_type frob --prior_type learnt --channel_type rayleigh --noise_var 1.0 --epochs 50
python running_example_4090.py --gpu 3 --norm_type spec --prior_type learnt --channel_type rayleigh --noise_var 1.0 --epochs 50
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 1.0
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 1.0
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 1.0 --epochs 20
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 1.0 --epochs 20
python running_example_4090.py --gpu 3 --norm_type frob --prior_type rand --channel_type rayleigh --noise_var 1.0 --epochs 50
python running_example_4090.py --gpu 3 --norm_type spec --prior_type rand --channel_type rayleigh --noise_var 1.0 --epochs 50
EOF
    echo "GPU 3: All jobs finished."
) &
PID_GPU3=$!

# ====================================================
# MAIN WAIT
# ====================================================
wait $PID_GPU2
wait $PID_GPU3
echo "All parallel processing complete."