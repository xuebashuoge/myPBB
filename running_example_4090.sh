#!/bin/bash

# ====================================================
# CONFIGURATION
# ====================================================

# ====================================================
# TASK LIST
# ====================================================
read -r -d '' COMMAND_LIST << EOM




# spec norm

# SNR -10dB
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.005 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.001 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.01 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.1 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 0.5 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective channel_gradient --kl_penalty 1.0 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 10 --objective vanilla

# SNR -10dB
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.005 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.001 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.01 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.1 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 0.5 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective channel_gradient --kl_penalty 1.0 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 20 --objective vanilla

# SNR -10dB
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.005 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.001 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.01 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.05 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.1 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 0.5 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective channel_gradient --kl_penalty 1.0 --channel_penalty 1.0
python running_example_4090.py --norm_type spec --prior_type learnt --channel_type rayleigh_zf --noise_var 10.0 --epochs 50 --objective vanilla









EOM

# ====================================================
# PYTHON DISPATCHER (修复版)
# ====================================================
export COMMAND_LIST

python3 -c "
import os
import subprocess
import threading
import queue
import time
import sys

# 【修改点 1】直接在这里指定，不再依赖 os.environ 避免 KeyError
# 如果你想改 GPU，直接改下面这个列表即可
gpus = ['0', '1', '2', '3']

# 【修改点 2】同理，直接指定并发数，避免环境变量读取失败
max_jobs = 6

# 读取任务列表
try:
    raw_commands = os.environ['COMMAND_LIST'].strip().split('\n')
except KeyError:
    print('Error: COMMAND_LIST not found. Make sure to export it in bash.')
    sys.exit(1)

commands = [cmd.strip() for cmd in raw_commands if cmd.strip() and not cmd.strip().startswith('#')]

if not commands:
    print('No commands found to run.')
    sys.exit(0)

print(f'Detected {len(commands)} tasks to run across GPUs: {gpus}')

# 资源池初始化
gpu_resource_pool = queue.Queue()
for gpu_id in gpus:
    for _ in range(max_jobs):
        gpu_resource_pool.put(gpu_id)

def worker(cmd):
    gpu_id = gpu_resource_pool.get() 
    try:
        full_cmd = f'{cmd} --gpu {gpu_id}'
        # print(f'[Starting on GPU {gpu_id}] {cmd[:40]}...')
        subprocess.run(full_cmd, shell=True)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        gpu_resource_pool.put(gpu_id)

threads = []
start_time = time.time()

for cmd in commands:
    t = threading.Thread(target=worker, args=(cmd,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

end_time = time.time()
print(f'All jobs finished in {end_time - start_time:.2f} seconds.')
"

echo "Done."