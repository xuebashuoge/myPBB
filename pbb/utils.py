import math
from random import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, select_channel_network, select_network, select_prior_network, ProbConv2d, ProbLinear
from pbb.bounds import PBBobj
from pbb import data
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import gc

# TODOS: 1. make a train prior function (bbb, erm)
#        2. make train posterior function 
#        3. rename partitions of data (prior_data, posterior_data, eval_data)
#        4. implement early stopping with validation set & speed
#        5. add data augmentation (maria)
#        6. better way of logging

def runexp(name_data, objective, prior_type, model, sigma_prior, pmin, learning_rate, momentum, learning_rate_prior=0.01, momentum_prior=0.95, delta=0.025, layers=9, clamping=True, delta_test=0.01, mc_samples=1000, samples_ensemble=100, kl_penalty=1, initial_lamb=6.0, train_epochs=100, prior_dist='gaussian', verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, verbose_test=False, perc_prior=0.2, batch_size=250, channel_type='nochannel', outage=0.1, noise_var=1.0, tx_power=1.0, l_0=2, seed=7):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    
    model : string
        could be cnn or fcn
    
    sigma_prior : float
        scale hyperparameter for the prior
    
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    
    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    
    delta : float
        confidence parameter for the risk certificate
    
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    
    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)
    
    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        import random as py_random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        py_random.seed(worker_seed)

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}
    # loader_kargs['persistent_workers'] = True
    loader_kargs['generator'] = g
    loader_kargs['worker_init_fn'] = seed_worker

    train, test = data.loaddataset(name_data)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # result path for prior, posterior, and certificates
    prior_folder = f'results/prior/{model}-{layers}_{name_data}_{prior_type}_{prior_dist}_sig{sigma_prior}{f'_perc-pri{perc_prior}_epoch-pri{prior_epochs}_bs-pri{batch_size}_lr-pri{learning_rate_prior}_mom-pri{momentum_prior}_dp-pri{dropout_prob}' if prior_type == 'learnt' else ''}_seed{seed}/'

    posterior_folder = f'results/posterior/{model}-{layers}_{name_data}_{prior_type}_{prior_dist}_sig{sigma_prior}_{f'bounded-pmin{pmin}' if clamping else 'unbounded'}_epoch{train_epochs}_bs{batch_size}_lr{learning_rate}_mon{momentum}_dp{dropout_prob}_objective-{objective}{f'_perc-pri{perc_prior}_epoch-pri{prior_epochs}_bs-pri{batch_size}_lr-pri{learning_rate_prior}_mom-pri{momentum_prior}_dp-pri{dropout_prob}' if prior_type == 'learnt' else ''}_seed{seed}/'

    certificate_folder = f'{posterior_folder}/certificates/'

    os.makedirs(prior_folder, exist_ok=True)
    os.makedirs(posterior_folder, exist_ok=True)
    os.makedirs(certificate_folder, exist_ok=True)


    train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(train, test, loader_kargs, batch_size, prior=(prior_type == 'learnt'), perc_train=perc_train, perc_prior=perc_prior, seed=seed)


    net0 = select_prior_network(model, layers, name_data, dropout_prob, device=device)

    if os.path.exists(f'{prior_folder}/prior_model.pt'):
        net0.load_state_dict(torch.load(f'{prior_folder}/prior_model.pt', map_location=device))
        with open(f'{prior_folder}/prior_results.json', 'r') as f:
            result_prior = json.load(f)
        
        errornet0 = result_prior['test_error']

        print(f"Loaded prior model from file: {prior_folder}/prior_model.pt")
        if prior_type == 'learnt':
            print(f"Prior train loss: {result_prior['train_loss'][-1]}, train error: {result_prior['train_error'][-1]}, test error: {result_prior['test_error']}")
        elif prior_type == 'rand':
            print(f"Prior test error: {result_prior['test_error']}")
    else:
        print("Training prior model from scratch.")
        if prior_type == 'learnt':
            optimizer = optim.SGD(
                net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
            
            loss_pri_tr = []
            error_pri_tr = []
            for epoch in trange(prior_epochs):
                avgloss_pri, avgerr_pri = trainNNet(net0, optimizer, epoch, valid_loader, device=device, verbose=verbose)
                loss_pri_tr.append(avgloss_pri.item() if torch.is_tensor(avgloss_pri) else avgloss_pri)
                error_pri_tr.append(avgerr_pri.item() if torch.is_tensor(avgerr_pri) else avgerr_pri)

            plt.figure()
            plt.plot(range(1,prior_epochs+1), loss_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior NLL loss')
            plt.savefig(f'{prior_folder}/prior_loss.pdf', dpi=300, bbox_inches='tight')

            plt.figure()
            plt.plot(range(1,prior_epochs+1), error_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior 0-1 error')
            plt.savefig(f'{prior_folder}/prior_err.pdf', dpi=300, bbox_inches='tight')

            plt.close('all')

        # test for prior network
        # Optimization: Clean cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        with torch.no_grad():
            errornet0 = testNNet(net0, test_loader, device=device)

        # save prior model and results in human readable format
        torch.save(net0.state_dict(), f'{prior_folder}/prior_model.pt')
        result_prior = {
            'test_error': errornet0
        }
        if prior_type == 'learnt':
            result_prior['train_loss'] = loss_pri_tr
            result_prior['train_error'] = error_pri_tr

        with open(f'{prior_folder}/prior_results.json', 'w') as f:
            json.dump(result_prior, f, indent=4, default=vars)


    posterior_n_size = len(train_loader.dataset)
    bound_n_size = len(val_bound.dataset)

    toolarge = True if model == 'cnn' else False
    train_size = len(train_loader.dataset)
    classes = len(train_loader.dataset.classes)

    net = select_network(model, layers, name_data, sigma_prior, prior_dist, device=device, init_net=net0)
    net_channel = select_channel_network(model, layers, name_data, sigma_prior, prior_dist, l_0, channel_type, outage, tx_power, noise_var, device=device, init_net=net0)

    # This frees up GPU memory for the main training loop.
    net0 = net0.to('cpu')
    
    # Optimization: Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    # import ipdb
    # ipdb.set_trace()
    bound = PBBobj(objective, pmin, classes, delta, delta_test, mc_samples, kl_penalty, device, n_posterior=posterior_n_size, n_bound=bound_n_size)

    if objective == 'flamb':
        lambda_var = Lambda_var(initial_lamb, train_size).to(device)
        optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
    else:
        optimizer_lambda = None
        lambda_var = None

    if os.path.exists(f'{posterior_folder}/posterior_model.pt'):
        net.load_state_dict(torch.load(f'{posterior_folder}/posterior_model.pt', map_location=device))
        with open(f'{posterior_folder}/posterior_results.json', 'r') as f:
            result_posterior = json.load(f)
        print(f"Loaded posterior model from file: {posterior_folder}/posterior_model.pt")
        print(f"Posterior train loss: {result_posterior['train_loss'][-1]}, train error: {result_posterior['train_error'][-1]}, kl: {result_posterior['train_kl'][-1]}")
    else:
        print("Training posterior model from scratch.")


        

        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

        loss_tr = []
        err_tr = []
        kl_tr = []

        for epoch in trange(train_epochs):
            avgbound, avgkl, avgloss, avgerr = trainPNNet(net, optimizer, bound, epoch, train_loader, clamping, lambda_var, optimizer_lambda, verbose)
            # records train results
            loss_tr.append(avgloss.item() if torch.is_tensor(avgloss) else avgloss)
            err_tr.append(avgerr.item() if torch.is_tensor(avgerr) else avgerr)
            kl_tr.append(avgkl.detach().item() if torch.is_tensor(avgkl) else avgkl)

            if verbose_test and ((epoch+1) % 5 == 0):
                with torch.no_grad():
                    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge,
                    bound, device=device, lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

                    stch_loss, stch_err = testStochastic(net, test_loader, bound, device=device)
                    post_loss, post_err = testPosteriorMean(net, test_loader, bound, device=device)
                    ens_loss, ens_err = testEnsemble(net, test_loader, bound, device=device, samples=samples_ensemble)

                    print(f"***Checkpoint results***")         
                    print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
                    print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

        # save posterior model and results in human readable format
        torch.save(net.state_dict(), f'{posterior_folder}/posterior_model.pt')
        result_posterior = {
            'train_loss': loss_tr,
            'train_error': err_tr,
            'train_kl': kl_tr
        }
        with open(f'{posterior_folder}/posterior_results.json', 'w') as f:
            json.dump(result_posterior, f, indent=4, default=vars)

        plt.figure()
        plt.plot(range(1,train_epochs+1), loss_tr)
        plt.xlabel('Epochs')
        plt.ylabel('Posterior NLL loss')
        plt.savefig(f'{posterior_folder}/posterior_loss.pdf', dpi=300, bbox_inches='tight')
        plt.figure()
        plt.plot(range(1,train_epochs+1), err_tr)
        plt.xlabel('Epochs')
        plt.ylabel('Posterior 0-1 error')
        plt.savefig(f'{posterior_folder}/posterior_err.pdf', dpi=300, bbox_inches='tight')
        plt.figure()
        plt.plot(range(1,train_epochs+1), kl_tr)
        plt.xlabel('Epochs')
        plt.ylabel('Posterior KL')
        plt.savefig(f'{posterior_folder}/posterior_kl.pdf', dpi=300, bbox_inches='tight')
        plt.close('all')

    # Optimization: Final cleanup before potentially heavy certificate computation
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()


    # certificates files
    if channel_type.lower() == 'rayleigh':
        channel_specs = f'rayleigh-tx{tx_power}-noise{noise_var}'
        wireless = True
    elif channel_type.lower() == 'bec':
        channel_specs = f'bec-outage{outage}'
        wireless = True
    else:
        channel_specs = 'nochannel'
        wireless = False
    certificate_file = f"{certificate_folder}/{channel_specs}_chan-layer{l_0}_mcsamples{mc_samples}_seed{seed}_results.json"

    # load trained posterior model for certificates
    net_channel.load_state_dict(state_dict=net.state_dict())

    with torch.no_grad():
        train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(net, toolarge, bound, clamping, device=device, lambda_var=lambda_var, train_loader=val_bound, whole_train=val_bound_one_batch)

        stch_loss, stch_err = testStochastic(net_channel, test_loader, bound, wireless=wireless, clamping=clamping, device=device)
        post_loss, post_err = testPosteriorMean(net_channel, test_loader, bound, wireless=wireless, clamping=clamping, device=device)
        ens_loss, ens_err = testEnsemble(net_channel, test_loader, bound, wireless=wireless, clamping=clamping, device=device, samples=samples_ensemble)

    certificate_results = {
        'risk_certificate_ce': risk_ce,
        'risk_certificate_01': risk_01,
        'kl_divergence': kl,
        'train_nll_loss': loss_ce_train,
        'train_01_error': loss_01_train,
        'stochastic_loss': stch_loss,
        'stochastic_01_error': stch_err,
        'posterior_mean_loss': post_loss,
        'posterior_mean_01_error': post_err,
        'ensemble_loss': ens_loss,
        'ensemble_01_error': ens_err,
        'errornet0': errornet0
    }

    with open(certificate_file, 'w') as f:
        json.dump(certificate_results, f, indent=4, default=vars)


    print(f"***Final results***") 
    print(f"Objective, Dataset, Sigma, pmin, LR, momentum, LR_prior, momentum_prior, kl_penalty, dropout, Obj_train, Risk_CE, Risk_01, KL, Train NLL loss, Train 01 error, Stch loss, Stch 01 error, Post mean loss, Post mean 01 error, Ens loss, Ens 01 error, 01 error prior net, perc_train, perc_prior")
    print(f"{objective}, {name_data}, {sigma_prior :.5f}, {pmin :.5f}, {learning_rate :.5f}, {momentum :.5f}, {learning_rate_prior :.5f}, {momentum_prior :.5f}, {kl_penalty : .5f}, {dropout_prob :.5f}, {train_obj :.5f}, {risk_ce :.5f}, {risk_01 :.5f}, {kl :.5f}, {loss_ce_train :.5f}, {loss_01_train :.5f}, {stch_loss :.5f}, {stch_err :.5f}, {post_loss :.5f}, {post_err :.5f}, {ens_loss :.5f}, {ens_err :.5f}, {errornet0 :.5f}, {perc_train :.5f}, {perc_prior :.5f}")

def compute_empirical_risk(outputs, targets, pmin, clamping=True, per_sample=False):
    # compute negative log likelihood loss and bound it with pmin (if applicable)
    # rescaling by log(1/pmin) if clamping
    empirical_risk = F.nll_loss(outputs, targets, reduction='none' if per_sample else 'mean')
    if clamping == True:
        empirical_risk = (1./(np.log(1./pmin))) * empirical_risk
    return empirical_risk

def compute_lipschitz(name_data, prior_type, model, sigma_prior, pmin, learning_rate_prior=0.01, momentum_prior=0.95, layers=9, clamping=True, mc_samples=1000, chunk_size=100, prior_dist='gaussian', verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.2, batch_size=250, channel_type='nochannel', outage=0.1, noise_var=1.0, tx_power=1.0, l_0=2, seed=7):
    """
    Compute the Lipschitz constant for the given model and data. The model weights follows the prior distribution, where only the l0-th layer is determined by channel, while other layers are sampled from the prior.

    Parameters
    ----------
    name_data : str
        Name of the dataset.

    prior_type : str
        Type of the prior distribution.

    model : nn.Module
        The model to evaluate.

    sigma_prior : float
        The prior standard deviation.

    pmin : float
        The minimum probability.

    layers : int, optional
        Number of layers in the model (default is 9).

    clamping : bool, optional
        Whether to apply clamping (default is True).

    mc_samples : int, optional
        Number of Monte Carlo samples (default is 1000).

    samples_ensemble : int, optional
        Number of samples for ensemble (default is 100).

    prior_dist : str, optional
        Type of the prior distribution (default is 'gaussian').

    verbose : bool, optional
        Whether to print verbose output (default is False).

    device : str, optional
        Device to run the computation on (default is 'cuda').

    channel_type : str, optional
        Type of the channel (default is 'nochannel').

    outage : float, optional
        Outage probability (default is 0.1).

    noise_var : float, optional
        Noise variance (default is 1.0).

    tx_power : float, optional
        Transmission power (default is 1.0).

    l_0 : int, optional
        Initial layer (default is 2).

    seed : int, optional
        Random seed (default is 7).

    Returns
    -------
    float
        The computed Lipschitz constant.
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        import random as py_random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        py_random.seed(worker_seed)

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}
    # loader_kargs['persistent_workers'] = True
    loader_kargs['generator'] = g
    loader_kargs['worker_init_fn'] = seed_worker

    train, test = data.loaddataset(name_data)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # result path for prior, posterior, and certificates
    prior_folder = f'results/prior/{model}-{layers}_{name_data}_{prior_type}_{prior_dist}_sig{sigma_prior}{f'_perc-pri{perc_prior}_epoch-pri{prior_epochs}_bs-pri{batch_size}_lr-pri{learning_rate_prior}_mom-pri{momentum_prior}_dp-pri{dropout_prob}' if prior_type == 'learnt' else ''}_seed{seed}/'

    train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(train, test, loader_kargs, batch_size, prior=(prior_type == 'learnt'), perc_train=perc_train, perc_prior=perc_prior, seed=seed)

    net0 = select_prior_network(model, layers, name_data, dropout_prob, device=device)

    if os.path.exists(f'{prior_folder}/prior_model.pt'):
        net0.load_state_dict(torch.load(f'{prior_folder}/prior_model.pt', map_location=device))
        with open(f'{prior_folder}/prior_results.json', 'r') as f:
            result_prior = json.load(f)
        
        errornet0 = result_prior['test_error']

        print(f"Loaded prior model from file: {prior_folder}/prior_model.pt")
        if prior_type == 'learnt':
            print(f"Prior train loss: {result_prior['train_loss'][-1]}, train error: {result_prior['train_error'][-1]}, test error: {result_prior['test_error']}")
        elif prior_type == 'rand':
            print(f"Prior test error: {result_prior['test_error']}")
    else:
        print("Training prior model from scratch.")
        if prior_type == 'learnt':
            optimizer = optim.SGD(
                net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
            
            loss_pri_tr = []
            error_pri_tr = []
            for epoch in trange(prior_epochs):
                avgloss_pri, avgerr_pri = trainNNet(net0, optimizer, epoch, valid_loader, device=device, verbose=verbose)
                loss_pri_tr.append(avgloss_pri.item() if torch.is_tensor(avgloss_pri) else avgloss_pri)
                error_pri_tr.append(avgerr_pri.item() if torch.is_tensor(avgerr_pri) else avgerr_pri)

            plt.figure()
            plt.plot(range(1,prior_epochs+1), loss_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior NLL loss')
            plt.savefig(f'{prior_folder}/prior_loss.pdf', dpi=300, bbox_inches='tight')

            plt.figure()
            plt.plot(range(1,prior_epochs+1), error_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior 0-1 error')
            plt.savefig(f'{prior_folder}/prior_err.pdf', dpi=300, bbox_inches='tight')

            plt.close('all')

        # test for prior network
        # Optimization: Clean cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        with torch.no_grad():
            errornet0 = testNNet(net0, test_loader, device=device)

        # save prior model and results in human readable format
        torch.save(net0.state_dict(), f'{prior_folder}/prior_model.pt')
        result_prior = {
            'test_error': errornet0
        }
        if prior_type == 'learnt':
            result_prior['train_loss'] = loss_pri_tr
            result_prior['train_error'] = error_pri_tr

        with open(f'{prior_folder}/prior_results.json', 'w') as f:
            json.dump(result_prior, f, indent=4, default=vars)

    net_prime = select_channel_network(model, layers, name_data, sigma_prior, prior_dist, l_0, channel_type, outage, tx_power, noise_var, device=device, init_net=net0)
    net = select_network(model, layers, name_data, sigma_prior, prior_dist, device=device, init_net=net0)
    net_prime.eval()
    net.eval()


    # Optimization: Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    max_k = 0.0


    outputs = torch.zeros(test_loader.batch_size, test_loader.dataset.classes).to(device)
    outputs_prime = torch.zeros(test_loader.batch_size, test_loader.dataset.classes).to(device)
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            for i in range(len(data)):
                outputs[i, :] = net(data[i:i+1], sample=True, clamping=clamping, pmin=pmin)
                outputs_prime[i, :] = net_prime(data[i:i+1], sample=True, clamping=clamping, pmin=pmin, wireless=True)
            
            cross_entropy = compute_empirical_risk(outputs, target, pmin, clamping, per_sample=True)
            cross_entropy_prime = compute_empirical_risk(outputs_prime, target, pmin, clamping, per_sample=True)

            differences = torch.abs(cross_entropy - cross_entropy_prime)

            # TBC: cannot get the sampled weights
            



def compute_lipschitz_parallel(name_data, prior_type, model, sigma_prior, pmin, learning_rate_prior=0.01, momentum_prior=0.95, layers=9, clamping=True, mc_samples=1000, chunk_size=100, prior_dist='gaussian', verbose=False, device='cuda', prior_epochs=20, dropout_prob=0.2, perc_train=1.0, perc_prior=0.2, batch_size=250, channel_type='nochannel', outage=0.1, noise_var=1.0, tx_power=1.0, l_0=2, seed=7):
    """
    Compute the Lipschitz constant for the given model and data. The model weights follows the prior distribution, where only the l0-th layer is determined by channel, while other layers are sampled from the prior.

    Parameters
    ----------
    name_data : str
        Name of the dataset.

    prior_type : str
        Type of the prior distribution.

    model : nn.Module
        The model to evaluate.

    sigma_prior : float
        The prior standard deviation.

    pmin : float
        The minimum probability.

    layers : int, optional
        Number of layers in the model (default is 9).

    clamping : bool, optional
        Whether to apply clamping (default is True).

    mc_samples : int, optional
        Number of Monte Carlo samples (default is 1000).

    samples_ensemble : int, optional
        Number of samples for ensemble (default is 100).

    prior_dist : str, optional
        Type of the prior distribution (default is 'gaussian').

    verbose : bool, optional
        Whether to print verbose output (default is False).

    device : str, optional
        Device to run the computation on (default is 'cuda').

    channel_type : str, optional
        Type of the channel (default is 'nochannel').

    outage : float, optional
        Outage probability (default is 0.1).

    noise_var : float, optional
        Noise variance (default is 1.0).

    tx_power : float, optional
        Transmission power (default is 1.0).

    l_0 : int, optional
        Initial layer (default is 2).

    seed : int, optional
        Random seed (default is 7).

    Returns
    -------
    float
        The computed Lipschitz constant.
    """

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        import random as py_random
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        py_random.seed(worker_seed)

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}
    # loader_kargs['persistent_workers'] = True
    loader_kargs['generator'] = g
    loader_kargs['worker_init_fn'] = seed_worker

    train, test = data.loaddataset(name_data)

    if prior_type == 'rand':
        dropout_prob = 0.0

    # result path for prior, posterior, and certificates
    prior_folder = f'results/prior/{model}-{layers}_{name_data}_{prior_type}_{prior_dist}_sig{sigma_prior}{f'_perc-pri{perc_prior}_epoch-pri{prior_epochs}_bs-pri{batch_size}_lr-pri{learning_rate_prior}_mom-pri{momentum_prior}_dp-pri{dropout_prob}' if prior_type == 'learnt' else ''}_seed{seed}/'

    train_loader, test_loader, valid_loader, val_bound_one_batch, _, val_bound = data.loadbatches(train, test, loader_kargs, batch_size, prior=(prior_type == 'learnt'), perc_train=perc_train, perc_prior=perc_prior, seed=seed)

    net0 = select_prior_network(model, layers, name_data, dropout_prob, device=device)

    if os.path.exists(f'{prior_folder}/prior_model.pt'):
        net0.load_state_dict(torch.load(f'{prior_folder}/prior_model.pt', map_location=device))
        with open(f'{prior_folder}/prior_results.json', 'r') as f:
            result_prior = json.load(f)
        
        errornet0 = result_prior['test_error']

        print(f"Loaded prior model from file: {prior_folder}/prior_model.pt")
        if prior_type == 'learnt':
            print(f"Prior train loss: {result_prior['train_loss'][-1]}, train error: {result_prior['train_error'][-1]}, test error: {result_prior['test_error']}")
        elif prior_type == 'rand':
            print(f"Prior test error: {result_prior['test_error']}")
    else:
        print("Training prior model from scratch.")
        if prior_type == 'learnt':
            optimizer = optim.SGD(
                net0.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
            
            loss_pri_tr = []
            error_pri_tr = []
            for epoch in trange(prior_epochs):
                avgloss_pri, avgerr_pri = trainNNet(net0, optimizer, epoch, valid_loader, device=device, verbose=verbose)
                loss_pri_tr.append(avgloss_pri.item() if torch.is_tensor(avgloss_pri) else avgloss_pri)
                error_pri_tr.append(avgerr_pri.item() if torch.is_tensor(avgerr_pri) else avgerr_pri)

            plt.figure()
            plt.plot(range(1,prior_epochs+1), loss_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior NLL loss')
            plt.savefig(f'{prior_folder}/prior_loss.pdf', dpi=300, bbox_inches='tight')

            plt.figure()
            plt.plot(range(1,prior_epochs+1), error_pri_tr)
            plt.xlabel('Epochs')
            plt.ylabel('Prior 0-1 error')
            plt.savefig(f'{prior_folder}/prior_err.pdf', dpi=300, bbox_inches='tight')

            plt.close('all')

        # test for prior network
        # Optimization: Clean cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        with torch.no_grad():
            errornet0 = testNNet(net0, test_loader, device=device)

        # save prior model and results in human readable format
        torch.save(net0.state_dict(), f'{prior_folder}/prior_model.pt')
        result_prior = {
            'test_error': errornet0
        }
        if prior_type == 'learnt':
            result_prior['train_loss'] = loss_pri_tr
            result_prior['train_error'] = error_pri_tr

        with open(f'{prior_folder}/prior_results.json', 'w') as f:
            json.dump(result_prior, f, indent=4, default=vars)

    net_prime = select_channel_network(model, layers, name_data, sigma_prior, prior_dist, l_0, channel_type, outage, tx_power, noise_var, device=device, init_net=net0)
    net = select_network(model, layers, name_data, sigma_prior, prior_dist, device=device, init_net=net0)
    net_prime.eval()
    net.eval()


    # Optimization: Clear memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()
    gc.collect()

    max_k = 0.0

    # Define a function that performs the core calculation for a SINGLE data sample.
    # vmap will then vectorize this across the whole batch.
    def compute_k_for_sample(d_other_sq, sampled_weights, sampled_weights_prime, buffers, x_sample, y_sample):
        # --- NO CHANNEL (w) ---
        # Forward pass for the ideal network (wireless=False)
        outputs = torch.func.functional_call(
            net,
            (sampled_weights, buffers),
            args=(x_sample.unsqueeze(0),),
            kwargs={'sample': False}
        )
        loss = compute_empirical_risk(outputs, y_sample.unsqueeze(0), pmin, clamping, per_sample=True)

        # --- WITH CHANNEL (w') ---
        # Forward pass for the network with the channel (wireless=True)
        outputs_prime, channel_params = torch.func.functional_call(
            net_prime,
            (sampled_weights_prime, buffers),
            args=(x_sample.unsqueeze(0),),
            kwargs={'sample': False, 'wireless': True, 'return_channel_weight': True}
        )
        loss_prime = compute_empirical_risk(outputs_prime, y_sample.unsqueeze(0), pmin, clamping, per_sample=True)
        
        # --- DENOMINATOR ||w' - w||_2 ---
        channel_weight, channel_bias = channel_params
        
        # The 'ideal' weight is 1.0 and 'ideal' bias is 0.0
        # For BEC, channel_bias will be None.
        if channel_bias is not None:
            # Rayleigh channel case
            # Note: .contiguous() can sometimes help vmap performance
            flat_w_diff = (channel_weight - 1.0).contiguous().view(-1)
            flat_b_diff = channel_bias.contiguous().view(-1)
            # d_channel_sq = torch.sum(flat_w_diff.abs()**2) + torch.sum(flat_b_diff.abs()**2)

            # another valid but larger distance metric
            d_channel_sq = (torch.sqrt(torch.sum(flat_w_diff.abs()**2)) + torch.sqrt(torch.sum(flat_b_diff.abs()**2)))**2
        else:
            # BEC channel case
            d_channel_sq = torch.sum((channel_weight - 1.0)**2)

        d_w = torch.sqrt(d_other_sq + d_channel_sq)

        # simpler version without condition
        k_sample = torch.abs(loss_prime - loss) / d_w
            
        return k_sample.squeeze()

    # Vectorize our single-sample function to run on a full batch.
    vmapped_k_fn = torch.func.vmap(compute_k_for_sample, in_dims=(None, None, None, None, 0, 0), chunk_size=chunk_size, randomness="different")

    # Before the mc_samples loop, get the parameter names once
    param_names = []
    for name, module in net.named_modules():
        if isinstance(module, (ProbLinear, ProbConv2d)):
            param_names.append(f"{name}.weight.mu")
            param_names.append(f"{name}.bias.mu")
    # share the same param names for net_prime
    param_names_prime = param_names

    # record k for different mc samples
    k_mc = []

    print("Computing Lipschitz constant with the new method using torch.func...")
    with tqdm(total=len(test_loader) * mc_samples, desc="Processing") as pbar:
        for _ in range(mc_samples):

            max_k_mc = 0.0

            # 1. Sample one set of Bayesian weights for this MC iteration.
            sampled_weights = dict.fromkeys(param_names) # Pre-allocate
            sampled_weights_prime = dict.fromkeys(param_names_prime)
            # calculate other weights (which is fixed for all data samples in the loader)
            d_other_sq = torch.tensor(0.0, device=device)
            for name, module in net.named_modules():
                if isinstance(module, (ProbLinear, ProbConv2d)):
                    sampled_weights[f"{name}.weight.mu"] = module.weight.sample()
                    sampled_weights[f"{name}.bias.mu"] = module.bias.sample()
                    sampled_weights_prime[f"{name}.weight.mu"] = module.weight.sample()
                    sampled_weights_prime[f"{name}.bias.mu"] = module.bias.sample()

                    diff_weights = (sampled_weights[f"{name}.weight.mu"] - sampled_weights_prime[f"{name}.weight.mu"])
                    diff_bias = (sampled_weights[f"{name}.bias.mu"] - sampled_weights_prime[f"{name}.bias.mu"])
                    d_other_sq += torch.sum(diff_weights * diff_weights) + torch.sum(diff_bias * diff_bias)

            buffers = {name: buf for name, buf in net.named_buffers()}

            for data_batch, target_batch in test_loader:
                data_batch, target_batch = data_batch.to(device), target_batch.to(device)
                
                # 2. Compute all per-sample k values in one vectorized call
                k_values_batch = vmapped_k_fn(d_other_sq, sampled_weights, sampled_weights_prime, buffers, data_batch, target_batch)

                # 3. Find the max k in the current batch and update the global max
                batch_max_k = torch.max(k_values_batch).item()
                if batch_max_k > max_k:
                    max_k = batch_max_k
                if batch_max_k > max_k_mc:
                    max_k_mc = batch_max_k

                pbar.update(1)

            k_mc.append(max_k_mc)

            # Clean up memory after each MC sample
            del sampled_weights, sampled_weights_prime, buffers
            if device == 'cuda': torch.cuda.empty_cache()
            elif device == 'mps': torch.mps.empty_cache()
            gc.collect()

    print(f"Estimated Lipschitz constant over {mc_samples} MC samples: {max_k}")

    if channel_type.lower() == 'rayleigh':
        channel_specs = f'rayleigh-tx{tx_power}-noise{noise_var}'
    elif channel_type.lower() == 'bec':
        channel_specs = f'bec-outage{outage}'
    else:
        channel_specs = 'nochannel'

    lip_folder = f'results/lipschitz/{model}-{layers}_{name_data}_{prior_type}_{prior_dist}_sig{sigma_prior}{f'_perc-pri{perc_prior}_epoch-pri{prior_epochs}_bs-pri{batch_size}_lr-pri{learning_rate_prior}_mom-pri{momentum_prior}_dp-pri{dropout_prob}' if prior_type == 'learnt' else ''}_{channel_specs}_chan-layer{l_0}_mcsamples{mc_samples}_seed{seed}/'
    os.makedirs(lip_folder, exist_ok=True)

    with open(f'{lip_folder}/lipschitz_results.json', 'w') as f:
        json.dump({'lipschitz_constant': max_k}, f, indent=4, default=vars)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
