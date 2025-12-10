#!/bin/bash

# ====================================================
# CONFIGURATION
# ====================================================
# How many scripts to run AT THE SAME TIME on a single GPU?
# Set this based on your VRAM.
# e.g., If each script takes 4GB, a 24GB 4090 can handle ~5-6 jobs safely.
MAX_JOBS=4 

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
python lip_compute_4090.py --gpu 1 --norm_type frob --channel_type bec --outage 0.1
python lip_compute_4090.py --gpu 1 --norm_type frob --channel_type bec --outage 0.2
python lip_compute_4090.py --gpu 1 --norm_type frob --channel_type bec --outage 0.3
python lip_compute_4090.py --gpu 1 --norm_type frob --channel_type bec --outage 0.4
python lip_compute_4090.py --gpu 1 --norm_type frob --channel_type bec --outage 0.5
python lip_compute_4090.py --gpu 1 --norm_type spec --channel_type bec --outage 0.1
python lip_compute_4090.py --gpu 1 --norm_type spec --channel_type bec --outage 0.2
python lip_compute_4090.py --gpu 1 --norm_type spec --channel_type bec --outage 0.3
python lip_compute_4090.py --gpu 1 --norm_type spec --channel_type bec --outage 0.4
python lip_compute_4090.py --gpu 1 --norm_type spec --channel_type bec --outage 0.5
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
python lip_compute_4090.py --gpu 3 --norm_type frob --channel_type rayleigh --noise_var 0.1
python lip_compute_4090.py --gpu 3 --norm_type frob --channel_type rayleigh --noise_var 0.1778
python lip_compute_4090.py --gpu 3 --norm_type frob --channel_type rayleigh --noise_var 0.3162
python lip_compute_4090.py --gpu 3 --norm_type frob --channel_type rayleigh --noise_var 0.5623
python lip_compute_4090.py --gpu 3 --norm_type frob --channel_type rayleigh --noise_var 1.0
python lip_compute_4090.py --gpu 3 --norm_type spec --channel_type rayleigh --noise_var 0.1
python lip_compute_4090.py --gpu 3 --norm_type spec --channel_type rayleigh --noise_var 0.1778
python lip_compute_4090.py --gpu 3 --norm_type spec --channel_type rayleigh --noise_var 0.3162
python lip_compute_4090.py --gpu 3 --norm_type spec --channel_type rayleigh --noise_var 0.5623
python lip_compute_4090.py --gpu 3 --norm_type spec --channel_type rayleigh --noise_var 1.0
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