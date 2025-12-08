import torch
from pbb.utils import runexp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

BATCH_SIZE = 250
TRAIN_EPOCHS = 10
DELTA = 0.025
DELTA_TEST = 0.01
PRIOR = 'rand'

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
# runexp('mnist', 'vanilla', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='bec', outage=0.1, l_0=2, seed=7, norm_type='frob')

# runexp('mnist', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='bec', outage=0.1, l_0=2, seed=7, norm_type='frob')

runexp('cifar10', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=9, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, kl_penalty=0.1, channel_type='bec', outage=0.1, l_0=4, seed=7, norm_type='frob')


runexp('mnist', 'vanilla', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='rayleigh', noise_var=1.0, l_0=2, seed=7, norm_type='frob')

runexp('mnist', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='rayleigh', noise_var=1.0, l_0=2, seed=7, norm_type='frob')

runexp('cifar10', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=9, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, kl_penalty=0.1, channel_type='rayleigh', noise_var=1.0, l_0=4, seed=7, norm_type='frob')



runexp('mnist', 'vanilla', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='bec', outage=0.1, l_0=2, seed=7, norm_type='spec')

runexp('mnist', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='bec', outage=0.1, l_0=2, seed=7, norm_type='spec')

runexp('cifar10', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=9, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, kl_penalty=0.1, channel_type='bec', outage=0.1, l_0=4, seed=7, norm_type='spec')


runexp('mnist', 'vanilla', PRIOR, 'fcn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='rayleigh', noise_var=1.0, l_0=2, seed=7, norm_type='spec')

runexp('mnist', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=4, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, channel_type='rayleigh', noise_var=1.0, l_0=2, seed=7, norm_type='spec')

runexp('cifar10', 'vanilla', PRIOR, 'cnn', SIGMAPRIOR, PMIN, LEARNING_RATE, MOMENTUM, LEARNING_RATE_PRIOR, MOMENTUM_PRIOR, delta=DELTA, layers=9, clamping=CLAMPING, delta_test=DELTA_TEST, mc_samples=MC_SAMPLES, train_epochs=TRAIN_EPOCHS, device=DEVICE, prior_epochs=70, perc_train=1.0, perc_prior=0.5, verbose=True, dropout_prob=0.2, kl_penalty=0.1, channel_type='rayleigh', noise_var=1.0, l_0=4, seed=7, norm_type='spec')