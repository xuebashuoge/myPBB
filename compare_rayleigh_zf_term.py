import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
import matplotlib.pyplot as plt
import torch
# --- Configuration ---
snr_db_list = [-10, -5, 0, 5, 10]
tx_power = 1.0
d_values = [1, 5, 10, 100, 1000, 10000]  # Dimensions 1 to 16
mc_samples = 10000          # Monte Carlo samples

# Calculate Noise Variance from SNR
# SNR = P_tx / sigma^2  =>  sigma^2 = P_tx / SNR_linear
snr_linear_list = [10**(snr/10.0) for snr in snr_db_list]
noise_vars = [tx_power / snr_lin for snr_lin in snr_linear_list]

# --- Method 1: Semi-Analytical (Integral) ---
def phi(s):
    """Laplace transform for ratio of exponentials."""
    # Asymptotic expansion for large s to prevent overflow
    if s > 700:
        return 1.0/s - 2.0/(s**2) + 6.0/(s**3)
    if s < 1e-9:
        return 1.0
    return 1.0 - s * np.exp(s) * exp1(s)

def compute_analytical(d, noise_var):
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

# --- Method 2: Monte Carlo Simulation ---
def compute_mc(d, noise_var, num_samples, chunk_size=1000, device='cpu'):
    sigma2 = noise_var
    
    # We want to compute: mean( sqrt( sum( |Ni/Hi|^2 ) ) )
    # This stores the running sum for each of the num_samples
    running_sum = torch.zeros(num_samples, device=device)
    
    # Process 'd' in chunks
    # e.g., if d=10000 and chunk_size=1000, we loop 10 times
    for i in range(0, d, chunk_size):
        current_chunk = min(chunk_size, d - i)
        
        # 1. Generate small chunks of random numbers
        n_sq = torch.empty(num_samples, current_chunk, device=device).exponential_(1.0)
        h_sq = torch.empty(num_samples, current_chunk, device=device).exponential_(1.0)
        
        # 2. Compute ratios for this chunk: sigma^2 * (N/H)
        ratios = (sigma2 * n_sq) / h_sq
        
        # 3. Add to running sum (reduce along the chunk dimension)
        running_sum += torch.sum(ratios, dim=1)
        
        # Free memory explicitly (optional in Python but good for GPU)
        del n_sq, h_sq, ratios
        
    # Finally take sqrt and mean
    z = torch.sqrt(running_sum)
    return torch.mean(z).item()

# --- Execution & Plotting ---
results_analytical = {snr: [] for snr in snr_db_list}
results_mc = {snr: [] for snr in snr_db_list}

print("Running simulations...")
for i, snr in enumerate(snr_db_list):
    sigma2 = noise_vars[i]
    for d in d_values:
        # Analytical
        results_analytical[snr].append(compute_analytical(d, sigma2))
        # Monte Carlo
        results_mc[snr].append(compute_mc(d, sigma2, mc_samples, chunk_size=100000, device='cuda'))

print(results_analytical[0])
print(np.pi / (2 * np.sqrt(snr_linear_list)))


# Plot
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(snr_db_list)))

for i, snr in enumerate(snr_db_list):
    plt.plot(d_values, results_analytical[snr], color=colors[i], label=f'SNR {snr} dB (Ana)', linewidth=2)
    plt.plot(d_values, results_mc[snr], color=colors[i], label=f'SNR {snr} dB (MC)', linestyle='--', marker='o', markersize=4, alpha=0.7)

plt.title(r'Expectation of $\sqrt{\sum |N_i/H_i|^2}$ vs Dimension')
plt.xlabel('Dimension $d$')
plt.ylabel('Expectation Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig('compare_rayleigh_zf_term.png', dpi=300)