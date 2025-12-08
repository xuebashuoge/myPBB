import torch
import argparse
from pbb.utils import compute_lipschitz_parallel



# 1. Initialize the parser
parser = argparse.ArgumentParser(description="PAC-Bayes Bounds Configuration")

# 2. Add the requested arguments
# Setting default=3 preserves your original logic, but allows overrides (e.g., --gpu 0)
parser.add_argument('--gpu', type=int, default=3, help='The index of the CUDA device to use (default: 3)')

# Adding norm_type. Default is set to 2 (L2 norm), but can be changed to 'inf' or others.
parser.add_argument('--norm_type', type=str, default='frob', help='The type of norm to use (e.g., "frob", "spec")')

# 3. Parse the arguments
args = parser.parse_args()



DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 250
PRIOR = 'rand'

SIGMAPRIOR = 0.03
PMIN = 1e-5
CLAMPING = True
LEARNING_RATE_PRIOR = 0.005
MOMENTUM_PRIOR = 0.99

# note the number of MC samples used in the paper is 150.000, which usually takes a several hours to compute
MC_SAMPLES = 500
CHUNK_SIZE = 250

compute_lipschitz_parallel('mnist', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=4, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='bec', outage=0.5, l_0=2, seed=7, norm_type=args.norm_type)

compute_lipschitz_parallel('mnist', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=4, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=70, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='bec', outage=0.5, l_0=2, seed=7, norm_type=args.norm_type)

compute_lipschitz_parallel('cifar10', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=9, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='bec', outage=0.5, l_0=4, seed=7, norm_type=args.norm_type)

compute_lipschitz_parallel('mnist', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=4, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='rayleigh', tx_power=1.0, noise_var=0.1, l_0=2, seed=7, norm_type=args.norm_type)

compute_lipschitz_parallel('mnist', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=4, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=70, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='rayleigh', tx_power=1.0, noise_var=0.1, l_0=2, seed=7, norm_type=args.norm_type)

compute_lipschitz_parallel('cifar10', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=9, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='rayleigh', tx_power=1.0, noise_var=0.1, l_0=4, seed=7, norm_type=args.norm_type)

print("Lipschitz computation finished.")