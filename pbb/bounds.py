import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F


def compute_bec_binomial(dimension, outage, device='cpu'):
    # 1. Setup 'r' vector (no reshaping needed for scalar)
    r = torch.arange(1, dimension + 1, device=device, dtype=torch.float64)
    
    # 2. Compute Log-Probs
    # If outage is a float, PyTorch handles it efficiently here
    dist = torch.distributions.Binomial(total_count=dimension, probs=outage)
    log_probs = dist.log_prob(r)
    
    # 3. Summation
    # .sum() returns a scalar tensor
    return (torch.exp(log_probs) * torch.sqrt(r)).sum().item()

def compute_bec_spec(dimension, outage, device='cpu'):
    outage = torch.as_tensor(outage, dtype=torch.float64, device=device)
    # 1. log1p(-p) calculates ln(1 - p) accurately even for very small p
    log_success_prob = torch.log1p(-outage)
    
    # 2. Scale by dimension in log space
    log_total_success = dimension * log_success_prob
    
    # 3. -expm1(x) calculates -(e^x - 1) = 1 - e^x
    # This prevents cancellation error when e^x is close to 1
    return -torch.expm1(log_total_success).item()

def compute_rayleigh(tx_power, noise_var, device='cpu'):
    tx_power = torch.as_tensor(tx_power, dtype=torch.float64, device=device)
    noise_var = torch.as_tensor(noise_var, dtype=torch.float64, device=device)

    x = -1.0 / tx_power
    arg = -x / 2.0

    # bessel function (torch 1.9+)
    term1 = (1 - x) * torch.special.i0(arg)
    term2 = x * torch.special.i1(arg)

    fading_term = torch.sqrt(tx_power * math.pi) / 2.0 * torch.exp(x / 2.0) * (term1 - term2)
    noise_term = torch.sqrt(math.pi * noise_var) / 2.0
    
    return (fading_term + noise_term).item()

class PBBobj():
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired 
    training objective and evaluate the risk certificate at the end of training. 

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)
    
    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem
    
    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective
    
    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    mc_samples : int
        number of Monte Carlo samples when estimating the risk

    kl_penalty : float
        penalty for the kl coefficient in the training objective
    
    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, objective='fquad', pmin=1e-4, classes=10, delta=0.025,
    delta_test=0.01, mc_samples=1000, kl_penalty=1, device='cuda', n_posterior=30000, n_bound=30000, K=1.0, channel_type='bec', outage=0.5, tx_power=1.0, noise_var=1.0, norm_type='frob', dimension=1, channel_penalty=1.0):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        self.K = K
        self.channel_penalty = channel_penalty

        if channel_type.lower() == 'bec':
            if norm_type == 'frob':
                self.channel_term = K  * compute_bec_binomial(dimension, outage, device=device)
            elif norm_type == 'spec':
                self.channel_term = K * compute_bec_spec(dimension, outage, device=device)
            else:
                raise ValueError("norm_type must be 'frob' or 'spec'")
        elif channel_type.lower() == 'rayleigh':
            self.channel_term = K * compute_rayleigh(tx_power, noise_var, device=device)
        else:
            self.channel_term = 0.0



    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded == True:
            empirical_risk = (1./(np.log(1./self.pmin))) * empirical_risk
        return empirical_risk

    def compute_losses(self, net, data, target, clamping=True):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well
        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1-(correct/total)
        return loss_ce, loss_01, outputs

    def bound(self, empirical_risk, kl, train_size, lambda_var=None):
        # compute training objectives
        if self.objective == 'fquad':
            kl = kl * self.kl_penalty
            repeated_kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            first_term = torch.sqrt(
                empirical_risk + repeated_kl_ratio)
            second_term = torch.sqrt(repeated_kl_ratio)
            train_obj = torch.pow(first_term + second_term, 2)
        elif self.objective == 'flamb':
            kl = kl * self.kl_penalty
            lamb = lambda_var.lamb_scaled
            kl_term = torch.div(
                kl + np.log((2*np.sqrt(train_size)) / self.delta), train_size*lamb*(1 - lamb/2))
            first_term = torch.div(empirical_risk, 1 - lamb/2)
            train_obj = first_term + kl_term
        elif self.objective == 'fclassic':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(
                kl + np.log((2*np.sqrt(train_size))/self.delta), 2*train_size)
            train_obj = empirical_risk + torch.sqrt(kl_ratio)
        elif self.objective == 'bbb':
            # ipdb.set_trace()
            train_obj = empirical_risk + \
                self.kl_penalty * (kl/train_size)
        elif self.objective == 'vanilla':
            train_obj = empirical_risk
        elif self.objective == 'channel':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(kl + np.log(1.0/self.delta), np.sqrt(train_size))
            train_obj = empirical_risk + self.channel_penalty * self.channel_term + kl_ratio
        else:
            raise RuntimeError(f'Wrong objective {self.objective}')
        return train_obj

    def mcsampling(self, net, input, target, batches=True, clamping=True, data_loader=None):
        # compute empirical risk with Monte Carlo sampling
        error = 0.0
        cross_entropy = 0.0
        if batches:
            for batch_id, (data_batch, target_batch) in enumerate(tqdm(data_loader)):
                data_batch, target_batch = data_batch.to(
                    self.device), target_batch.to(self.device)
                cross_entropy_mc = 0.0
                error_mc = 0.0
                for i in range(self.mc_samples):
                    loss_ce, loss_01, _ = self.compute_losses(net,  data_batch, target_batch, clamping)
                    cross_entropy_mc += loss_ce
                    error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
                cross_entropy += cross_entropy_mc/self.mc_samples
                error += error_mc/self.mc_samples
            # we average cross-entropy and 0-1 error over all batches
            cross_entropy /= batch_id
            error /= batch_id
        else:
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for i in range(self.mc_samples):
                loss_ce, loss_01, _ = self.compute_losses(net, input, target, clamping)
                cross_entropy_mc += loss_ce
                error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc/self.mc_samples
            error += error_mc/self.mc_samples
        return cross_entropy, error

    def train_obj(self, net, input, target, clamping=True, lambda_var=None):
        # compute train objective and return all metrics
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, input, target, clamping)

        train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var)
        return train_obj, kl/self.n_posterior, outputs, loss_ce, loss_01

    def compute_final_stats_risk(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        # compute all final stats and risk certificates

        kl = net.compute_kl()
        if data_loader:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=True, clamping=clamping, data_loader=data_loader)
        else:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=False, clamping=clamping)

        empirical_risk_ce = inv_kl(
            error_ce.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01 = inv_kl(
            error_01, np.log(2/self.delta_test)/self.mc_samples)

        train_obj = self.bound(empirical_risk_ce, kl, self.n_posterior, lambda_var)

        risk_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 * np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
        risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 * np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)

        if torch.is_tensor(kl):
            kl = kl.item()

        if torch.is_tensor(train_obj):
            train_obj = train_obj.item()

        return train_obj, kl/self.n_bound, empirical_risk_ce, empirical_risk_01, risk_ce, risk_01


def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1-1e-10
    while((dch-izq)/dch >= 1e-5):
        p = (izq+dch)*.5
        if qs == 0:
            ikl = ks-(0+(1-qs)*math.log((1-qs)/(1-p)))
        elif qs == 1:
            ikl = ks-(qs*math.log(qs/p)+0)
        else:
            ikl = ks-(qs*math.log(qs/p)+(1-qs) * math.log((1-qs)/(1-p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
