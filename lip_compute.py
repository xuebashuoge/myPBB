import torch
from pbb.utils import compute_lipschitz_parallel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 250
PRIOR = 'rand'

SIGMAPRIOR = 0.03
PMIN = 1e-5
CLAMPING = True
LEARNING_RATE_PRIOR = 0.005
MOMENTUM_PRIOR = 0.99

# note the number of MC samples used in the paper is 150.000, which usually takes a several hours to compute
MC_SAMPLES = 150
CHUNK_SIZE = 50

compute_lipschitz_parallel('mnist', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, layers=4, clamping=CLAMPING, mc_samples=MC_SAMPLES, chunk_size=CHUNK_SIZE, device=DEVICE, prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.5, batch_size=BATCH_SIZE, verbose=True, channel_type='bec', outage=0.1, noise_var=1.0, tx_power=1.0, l_0=2, seed=7)

print("Lipschitz computation finished.")