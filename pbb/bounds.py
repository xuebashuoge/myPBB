import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn as nn
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

from scipy.integrate import quad
from scipy.special import exp1
def phi(s):
    """Laplace transform for ratio of exponentials."""
    # Asymptotic expansion for large s to prevent overflow
    if s > 700:
        return 1.0/s - 2.0/(s**2) + 6.0/(s**3)
    if s < 1e-9:
        return 1.0
    return 1.0 - s * np.exp(s) * exp1(s)

def compute_rayleigh_zf(d, noise_var):
    """
    Robust computation for large d using Split Integration + Scaling.
    """
    sigma0 = np.sqrt(noise_var)
    
    # We integrate the standardized variable (sigma=1)
    # The integrand is (1 - phi(t)^d) / t^1.5
    
    # CUTOFF SELECTION:
    # For large d (e.g., > 100), phi(t)^d vanishes very fast.
    # We can safely cut numerical integration at T=2.0
    # For smaller d, we might need a larger T, but for d=1000+, T=2 is plenty safe.
    T_cutoff = 2.0 if d > 100 else 100.0
    
    def integrand(t):
        if t < 1e-12: return 0
        return (1.0 - phi(t)**d) / (t**1.5)

    # 1. Numerical Integral [0, T_cutoff]
    # We add points near 0 to help the integrator see the sharp transition
    transition_point = 1.0 / d
    val_num, _ = quad(integrand, 0, T_cutoff, points=[transition_point])
    
    # 2. Analytical Tail [T_cutoff, inf]
    # Approximation: phi(t)^d ~= 0, so we just integrate 1/t^1.5
    # Integral of t^-1.5 is -2*t^-0.5
    # Value is: 0 - (-2/sqrt(T)) = 2/sqrt(T)
    val_tail = 2.0 / np.sqrt(T_cutoff)
    
    # Sum them up and apply constants
    integral_sum = val_num + val_tail
    result = (integral_sum / (2 * np.sqrt(np.pi))) * sigma0
    
    return result

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
        self.norm_type = norm_type

        if channel_type.lower() == 'bec':
            if norm_type == 'frob':
                self.channel_term = compute_bec_binomial(dimension, outage, device=device)
            elif norm_type == 'spec':
                self.channel_term = compute_bec_spec(dimension, outage, device=device)
            else:
                raise ValueError("norm_type must be 'frob' or 'spec'")
        elif channel_type.lower() == 'rayleigh':
            self.channel_term = compute_rayleigh(tx_power, noise_var, device=device)
        elif channel_type.lower() == 'rayleigh_zf':
            # now only works for scalar case
            snr = tx_power / noise_var
            self.channel_term = math.pi / (2 * math.sqrt(snr))
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

    def bound(self, empirical_risk, kl, train_size, lambda_var=None, net=None):
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
            train_obj = empirical_risk + self.kl_penalty * (kl/train_size)
        elif self.objective == 'vanilla':
            train_obj = empirical_risk
        elif self.objective == 'channel':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(kl + np.log(1.0/self.delta), np.sqrt(train_size))
            train_obj = empirical_risk + self.channel_penalty * self.K * self.channel_term + kl_ratio
        elif self.objective == 'channel_gradient':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(kl + np.log(1.0/self.delta), np.sqrt(train_size))
            gradient_K = self.compute_gradient_norm(net, empirical_risk)
            train_obj = empirical_risk + self.channel_penalty * gradient_K * self.channel_term + kl_ratio
        elif self.objective == 'channel_norm':
            kl = kl * self.kl_penalty
            kl_ratio = torch.div(kl + np.log(1.0/self.delta), np.sqrt(train_size))
            norm_K = self.compute_weight_norm(net)
            train_obj = empirical_risk + self.channel_penalty * norm_K * self.channel_term + kl_ratio
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

    def power_iteration_spectral_norm(self, weight, n_iters=1, eps=1e-12):
        """
        Estimates the spectral norm (largest singular value) of a matrix using Power Iteration.
        This is faster and more stable than SVD for training loops.
        """
        # Ensure 2D
        if weight.dim() > 2:
            weight = weight.reshape(weight.size(0), -1)
        elif weight.dim() == 1:
            weight = weight.unsqueeze(0)
            
        # Initialize random vector u
        u = torch.randn(weight.size(1), 1, device=weight.device, dtype=weight.dtype)
        v = None
        
        # Power iteration
        with torch.no_grad():
            for _ in range(n_iters):
                # v = W u
                v = torch.mm(weight, u)
                v = F.normalize(v, dim=0, eps=eps)
                
                # u = W^T v
                u = torch.mm(weight.t(), v)
                u = F.normalize(u, dim=0, eps=eps)

        # After converging u and v, compute sigma = v^T W u
        # We do this calculation WITH gradients if weight requires grad
        # v and u are treated as constants (detached) for the gradient of sigma w.r.t W
        # This is the standard "Spectral Normalization" trick (Miyato et al.)
        sigma = torch.mm(torch.mm(v.t(), weight), u)
        return sigma.squeeze()


    def compute_weight_norm(self, net):
        # Compute product of spectral norms in log-space to avoid overflow.
        log_acc = 0.0
        found = False
        for name, module in net.named_modules():
            if 'prior' in name:
                continue
            w = None
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                w = module.weight
            elif hasattr(module, 'weight') and hasattr(module.weight, 'mu'):
                w = module.weight.mu

            if w is None:
                continue

            if w.dim() > 2:
                w = w.reshape(w.size(0), -1)
            elif w.dim() == 1:
                w = w.unsqueeze(0)

            if self.norm_type == 'spec':
                layer_norm = self.power_iteration_spectral_norm(w, n_iters=5)
            elif self.norm_type == 'frob':
                layer_norm = torch.norm(w, p='fro')
            

            norm_val_float = layer_norm.item() if torch.is_tensor(layer_norm) else float(layer_norm)

            # If any layer is non-finite or non-positive, short-circuit to NaN.
            if not math.isfinite(norm_val_float) or norm_val_float <= 0:
                return float('nan')

            # Use torch.log on the TENSOR, not the float, to preserve gradients
            log_acc = log_acc + torch.log(layer_norm)

            found = True

        if not found:
            return 1.0

        # Clamp exponent to stay in float range; this caps the effective product.
        # We can use torch.clamp on the tensor result
        max_log = 80.0  # exp(80) ~ 5e34, well below inf
        log_acc = torch.clamp(log_acc, min=-max_log, max=max_log)
        
        return torch.exp(log_acc)

    def power_iteration_on_gradient(self, grad, n_iters=1, eps=1e-12):
        """
        Estimates the spectral norm of the GRADIENT matrix using Power Iteration.
        """
        # Ensure 2D
        if grad.dim() > 2:
            grad = grad.reshape(grad.size(0), -1)
        elif grad.dim() == 1:
            grad = grad.unsqueeze(0)
            
        # Initialize random vector u
        u = torch.randn(grad.size(1), 1, device=grad.device, dtype=grad.dtype)
        v = None
        
        # Power iteration
        # We treat u and v calculation as non-differentiable wrt the gradient structure itself 
        # to avoid higher-order derivatives explosion, similar to standard Spectral Norm.
        with torch.no_grad():
            for _ in range(n_iters):
                # v = G u
                v = torch.mm(grad, u)
                v = F.normalize(v, dim=0, eps=eps)
                
                # u = G^T v
                u = torch.mm(grad.t(), v)
                u = F.normalize(u, dim=0, eps=eps)

        # Compute sigma = v^T G u
        # This connects G (and thus W) to the loss
        sigma = torch.mm(torch.mm(v.t(), grad), u)
        return sigma.squeeze()

    def compute_gradient_norm(self, net, loss, create_graph=True):
        # Compute the norm of the gradients of the loss w.r.t parameters
        # Crucial for BNN: w = mu + sigma * epsilon
        # Therefore, dL/dmu = dL/dw * dw/dmu = dL/dw * 1 = dL/dw
        # So, the gradient w.r.t 'mu' IS the gradient w.r.t the sampled weights.
        # We must EXCLUDE 'rho' (variance parameters) as they don't represent the Lipschitz constant w.r.t weights.
        
        relevant_params = []
        for name, param in net.named_parameters():
            if not param.requires_grad:
                continue
            # Filter out 'rho' parameters, keep 'mu', 'weight', 'bias'
            if 'rho' not in name and 'prior' not in name:
                relevant_params.append(param)

        # create_graph=True allows differentiating through the gradient norm itself (double backprop)
        grads = torch.autograd.grad(loss, relevant_params, create_graph=create_graph, retain_graph=True, allow_unused=True)
        
        total_norm_sq = torch.as_tensor(0.0, device=self.device)
        for g in grads:
            if g is not None:
                # Handle NaNs in gradients gracefully
                if not torch.isfinite(g).all():
                    continue
                if self.norm_type == 'frob':
                    total_norm_sq = total_norm_sq + g.norm(2)**2
                elif self.norm_type == 'spec':
                    spec_val = self.power_iteration_on_gradient(g, n_iters=5)
                    total_norm_sq = total_norm_sq + spec_val**2

        
        return torch.sqrt(total_norm_sq)

    def train_obj(self, net, input, target, clamping=True, lambda_var=None):
        # compute train objective and return all metrics
        kl = net.compute_kl()
        loss_ce, loss_01, outputs = self.compute_losses(net, input, target, clamping)

        train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var, net)
        return train_obj, kl/self.n_posterior, outputs, loss_ce, loss_01

    def compute_final_stats_risk(self, net, input=None, target=None, data_loader=None, clamping=True, lambda_var=None):
        # compute all final stats and risk certificates

        kl = net.compute_kl()
        if data_loader:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=True, clamping=clamping, data_loader=data_loader)
        else:
            error_ce, error_01 = self.mcsampling(net, input, target, batches=False, clamping=clamping)

        empirical_risk_ce = inv_kl(error_ce.item(), np.log(2/self.delta_test)/self.mc_samples)
        empirical_risk_01 = inv_kl(error_01, np.log(2/self.delta_test)/self.mc_samples)

        # ignore this since we cannot compute gradient norm during inference
        # train_obj = self.bound(empirical_risk_ce, kl, self.n_posterior, lambda_var, net=net)
        train_obj = 0.0

        # risk_ce = inv_kl(empirical_risk_ce, (kl + np.log((2 * np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)
        # risk_01 = inv_kl(empirical_risk_01, (kl + np.log((2 * np.sqrt(self.n_bound))/self.delta_test))/self.n_bound)

        if torch.is_tensor(kl):
            kl = kl.item()

        if torch.is_tensor(train_obj):
            train_obj = train_obj.item()

        # modify: the risk is the bound in the PBB paper, which is useless in our scenario
        return train_obj, kl/self.n_bound, empirical_risk_ce, empirical_risk_01, error_ce.item(), error_01


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

def compute_network_lipschitz(net, train_loader, bound, clamping, K, device='cuda'):
    if bound.objective == 'channel':
        K_final = K
    elif bound.objective == 'channel_norm':
        K_final = bound.compute_weight_norm(net)
    elif bound.objective == 'channel_gradient':
        total_K = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            loss_ce, _, _ = bound.compute_losses(net, data, target, clamping)
            total_K += bound.compute_gradient_norm(net, loss_ce, create_graph=False)
        
        K_final = total_K / len(train_loader)
    else:
        K_final = 0.0

    return K_final.item() if torch.is_tensor(K_final) else K_final


