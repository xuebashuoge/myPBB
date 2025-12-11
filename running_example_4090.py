import torch
import argparse
from pbb.utils import runexp

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="PAC-Bayes Bounds Configuration")

# 2. Add the requested arguments
# Setting default=3 preserves your original logic, but allows overrides (e.g., --gpu 0)
parser.add_argument('--gpu', type=int, default=3, help='The index of the CUDA device to use (default: 3)')

# Adding norm_type. Default is set to 2 (L2 norm), but can be changed to 'inf' or others.
parser.add_argument('--norm_type', type=str, default='frob', help='The type of norm to use (e.g., "frob", "spec")')

parser.add_argument('--prior_type', type=str, default='rand', help='The type of prior to use (e.g., "rand", "learnt")')

parser.add_argument('--channel_type', type=str, default='bec', help='The type of channel to use (e.g., "bec", "rayleigh")')

parser.add_argument('--outage', type=float, default=0.5, help='The outage probability for the bec channel (default: 0.5)')

parser.add_argument('--noise_var', type=float, default=1.0, help='The noise variance for the rayleigh channel (default: 1.0)')

parser.add_argument('--epochs', type=int, default=10, help='The number of training epochs (default: 10)')

# 3. Parse the arguments
args = parser.parse_args()


DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 250
TRAIN_EPOCHS = args.epochs
DELTA = 0.025
DELTA_TEST = 0.01
PRIOR = args.prior_type

SIGMAPRIOR = 0.03
PMIN = 1e-5
CLAMPING = True
KL_PENALTY = 0.1
LEARNING_RATE = 0.001
MOMENTUM = 0.95
LEARNING_RATE_PRIOR = 0.005
MOMENTUM_PRIOR = 0.99

# note the number of MC samples used in the paper is 150.000, which usually takes a several hours to compute
MC_SAMPLES = 500

# note all of these running examples have different settings!
runexp('mnist', 'vanilla', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type=args.channel_type, outage=args.outage, tx_power=1.0, noise_var=args.noise_var, l_0=2, seed=7, norm_type=args.norm_type)

runexp('mnist', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type=args.channel_type, outage=args.outage, tx_power=1.0, noise_var=args.noise_var, l_0=2, seed=7, norm_type=args.norm_type)

runexp('cifar10', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=9, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, kl_penalty=0.1, channel_type=args.channel_type, outage=args.outage, tx_power=1.0, noise_var=args.noise_var, l_0=4, seed=7, norm_type=args.norm_type)


