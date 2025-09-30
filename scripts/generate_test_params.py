import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import pandas as pd

def generate_test_dataset(n_samples=1000, method='mixed'):
    """
    Generate test dataset for neural network evaluation
    
    Parameters:
    n_samples: number of test points
    method: 'interpolation', 'extrapolation', 'random', 'mixed'
    
    Returns:
    Dictionary with test parameter arrays
    """
    
    # Original training bounds
    #  n_max = 1e28 * ureg.m**-3 # maximum particle density
    # n_min = 1e20 * ureg.m**-3 # minimum particle density? 
    # T_max = 30e3        # 30 keV
    # T_min = 10          # eV
    # Vp_min = 400.0      # 400 V
    # Vp_max = 10e3       # 10 kV

    # New parameter bounds from Jack

    n_min, n_max = 1e23, 1e28
    T_min, T_max = 10, 1e4  # eV
    Vp_min, Vp_max = 1, 40e3  # V
    
    # Training points (for reference)
    n_train = np.array([1e20, 5e20, 1e21, 1e22, 6e22, 1e23, 1e24, 1e26, 1e27, 1e28])
    T_train = np.linspace(T_min, T_max, 10)
    Vp_train = np.linspace(Vp_min, Vp_max, 10)
    
    print(f"Training parameter ranges:")
    print(f"  n: {n_train}")
    print(f"  T: [{T_min}, {T_max}] eV")
    print(f"  Vp: [{Vp_min}, {Vp_max}] V")
    print()
    
    if method == 'interpolation':
        # Test points BETWEEN training points (interpolation)
        print("Generating INTERPOLATION test dataset...")
        print("Testing model's ability to predict between training points")
        
        # For n: pick values between training points
        n_test = []
        for i in range(len(n_train)-1):
            # Add 2-3 points between each training point
            n_between = np.logspace(np.log10(n_train[i]), np.log10(n_train[i+1]), 4)[1:-1]
            n_test.extend(n_between)
        n_test = np.array(n_test)
        
        # For T and Vp: offset grid points
        T_test = T_train[:-1] + np.diff(T_train) / 2  # Midpoints
        Vp_test = Vp_train[:-1] + np.diff(Vp_train) / 2  # Midpoints
        
        # Create combinations using Latin Hypercube Sampling
        sampler = qmc.LatinHypercube(d=3, seed=42)
        samples = sampler.random(n_samples)
        
        # Map to parameter ranges
        n_test_vals = np.interp(samples[:, 0], [0, 1], [n_test.min(), n_test.max()])
        T_test_vals = np.interp(samples[:, 1], [0, 1], [T_test.min(), T_test.max()])
        Vp_test_vals = np.interp(samples[:, 2], [0, 1], [Vp_test.min(), Vp_test.max()])
        
    elif method == 'extrapolation':
        # Test points OUTSIDE training range (extrapolation)
        print("Generating EXTRAPOLATION test dataset...")
        print("Testing model's ability to predict outside training range")
        
        # Extend ranges beyond training data
        n_ext_min, n_ext_max = 5e19, 2e28  # Slightly beyond training range
        T_ext_min, T_ext_max = 5, 50e3     # Extended T range
        Vp_ext_min, Vp_ext_max = 200, 15e3 # Extended Vp range
        
        # Focus on extrapolation regions
        sampler = qmc.LatinHypercube(d=3, seed=42)
        samples = sampler.random(n_samples)
        
        # Mix of extrapolation regions
        n_test_vals = np.where(
            samples[:, 0] < 0.3,
            np.interp(samples[:, 0], [0, 0.3], [n_ext_min, n_min*0.9]),  # Below training
            np.interp(samples[:, 0], [0.7, 1], [n_max*1.1, n_ext_max])    # Above training
        )
        n_test_vals = np.where((samples[:, 0] >= 0.3) & (samples[:, 0] <= 0.7), 
                              np.interp(samples[:, 0], [0.3, 0.7], [n_min, n_max]), n_test_vals)
        
        T_test_vals = np.interp(samples[:, 1], [0, 1], [T_ext_min, T_ext_max])
        Vp_test_vals = np.interp(samples[:, 2], [0, 1], [Vp_ext_min, Vp_ext_max])
        
    elif method == 'random':
        # Random sampling within training bounds
        print("Generating RANDOM test dataset...")
        print("Random sampling within training parameter space")
        
        np.random.seed(42)
        n_test_vals = np.random.uniform(np.log10(n_min), np.log10(n_max), n_samples)
        n_test_vals = 10**n_test_vals  # Convert back from log space
        T_test_vals = np.random.uniform(T_min, T_max, n_samples)
        Vp_test_vals = np.random.uniform(Vp_min, Vp_max, n_samples)
        
    elif method == 'mixed':
        # Mixed approach: interpolation + extrapolation + random
        print("Generating MIXED test dataset...")
        print("Combination of interpolation, extrapolation, and random sampling")
        
        n_interp = int(0.4 * n_samples)  # 40% interpolation
        n_extrap = int(0.3 * n_samples)  # 30% extrapolation  
        n_random = n_samples - n_interp - n_extrap  # 30% random
        
        # Interpolation subset
        interp_data = generate_test_dataset(n_interp, 'interpolation')
        
        # Extrapolation subset
        extrap_data = generate_test_dataset(n_extrap, 'extrapolation')
        
        # Random subset
        random_data = generate_test_dataset(n_random, 'random')
        
        # Combine
        n_test_vals = np.concatenate([
            interp_data['n_test'], extrap_data['n_test'], random_data['n_test']
        ])
        T_test_vals = np.concatenate([
            interp_data['T_test'], extrap_data['T_test'], random_data['T_test']
        ])
        Vp_test_vals = np.concatenate([
            interp_data['Vp_test'], extrap_data['Vp_test'], random_data['Vp_test']
        ])
        
    else:
        raise ValueError("Method must be 'interpolation', 'extrapolation', 'random', or 'mixed'")
    
    # Ensure we have exactly n_samples points
    if len(n_test_vals) != n_samples:
        indices = np.random.choice(len(n_test_vals), n_samples, replace=False)
        n_test_vals = n_test_vals[indices]
        T_test_vals = T_test_vals[indices]
        Vp_test_vals = Vp_test_vals[indices]
    
    print(f"\nGenerated {len(n_test_vals)} test points:")
    print(f"  n range: {n_test_vals.min():.2e} to {n_test_vals.max():.2e}")
    print(f"  T range: {T_test_vals.min():.1f} to {T_test_vals.max():.1f} eV")
    print(f"  Vp range: {Vp_test_vals.min():.1f} to {Vp_test_vals.max():.1f} V")
    
    return {
        'n_test': n_test_vals,
        'T_test': T_test_vals,
        'Vp_test': Vp_test_vals,
        'method': method,
        'n_samples': n_samples
    }

def visualize_test_vs_train(test_data):
    """
    Visualize test points vs training points
    """
    # Training points
    n_train = np.array([1e20, 5e20, 1e21, 1e22, 6e22, 1e23, 1e24, 1e26, 1e27, 1e28])
    T_train = np.linspace(10, 30e3, 10)
    Vp_train = np.linspace(400, 10e3, 10)
    
    # Create all combinations of training points for visualization
    n_tr_grid, T_tr_grid, Vp_tr_grid = np.meshgrid(n_train, T_train, Vp_train)
    n_tr_flat = n_tr_grid.flatten()
    T_tr_flat = T_tr_grid.flatten()
    Vp_tr_flat = Vp_tr_grid.flatten()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # n vs T
    axes[0,0].scatter(np.log10(n_tr_flat), T_tr_flat, c='blue', s=20, alpha=0.6, label='Training')
    axes[0,0].scatter(np.log10(test_data['n_test']), test_data['T_test'], c='red', s=10, alpha=0.6, label='Test')
    axes[0,0].set_xlabel('log₁₀(n) [m⁻³]')
    axes[0,0].set_ylabel('T [eV]')
    axes[0,0].set_title('n vs T')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # n vs Vp
    axes[0,1].scatter(np.log10(n_tr_flat), Vp_tr_flat, c='blue', s=20, alpha=0.6, label='Training')
    axes[0,1].scatter(np.log10(test_data['n_test']), test_data['Vp_test'], c='red', s=10, alpha=0.6, label='Test')
    axes[0,1].set_xlabel('log₁₀(n) [m⁻³]')
    axes[0,1].set_ylabel('Vp [V]')
    axes[0,1].set_title('n vs Vp')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # T vs Vp
    axes[0,2].scatter(T_tr_flat, Vp_tr_flat, c='blue', s=20, alpha=0.6, label='Training')
    axes[0,2].scatter(test_data['T_test'], test_data['Vp_test'], c='red', s=10, alpha=0.6, label='Test')
    axes[0,2].set_xlabel('T [eV]')
    axes[0,2].set_ylabel('Vp [V]')
    axes[0,2].set_title('T vs Vp')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Distribution histograms
    axes[1,0].hist(np.log10(test_data['n_test']), bins=30, alpha=0.7, color='red', label='Test')
    axes[1,0].axvline(np.log10(n_train).min(), color='blue', linestyle='--', label='Train bounds')
    axes[1,0].axvline(np.log10(n_train).max(), color='blue', linestyle='--')
    axes[1,0].set_xlabel('log₁₀(n) [m⁻³]')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('n Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].hist(test_data['T_test'], bins=30, alpha=0.7, color='red', label='Test')
    axes[1,1].axvline(T_train.min(), color='blue', linestyle='--', label='Train bounds')
    axes[1,1].axvline(T_train.max(), color='blue', linestyle='--')
    axes[1,1].set_xlabel('T [eV]')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('T Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    axes[1,2].hist(test_data['Vp_test'], bins=30, alpha=0.7, color='red', label='Test')
    axes[1,2].axvline(Vp_train.min(), color='blue', linestyle='--', label='Train bounds')
    axes[1,2].axvline(Vp_train.max(), color='blue', linestyle='--')
    axes[1,2].set_xlabel('Vp [V]')
    axes[1,2].set_ylabel('Frequency')
    axes[1,2].set_title('Vp Distribution')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Test Dataset: {test_data["method"].capitalize()} Method ({test_data["n_samples"]} points)')
    plt.tight_layout()
    plt.show()

def save_test_parameters(test_data, filename='test_parameters.csv'):
    """
    Save test parameters to CSV for running simulations
    """
    df = pd.DataFrame({
        'n': test_data['n_test'],
        'T': test_data['T_test'],
        'Vp': test_data['Vp_test']
    })
    
    df.to_csv(filename, index=False)
    print(f"Test parameters saved to {filename}")
    print(f"You can now run your Vlasov simulations with these {len(df)} parameter sets")
    
    return df

# Recommended test strategies
def recommend_test_strategy():
    """
    Provide recommendations for test dataset design
    """
    print("=== RECOMMENDED TEST STRATEGY ===")
    print()
    print("1. INTERPOLATION TEST (400 points):")
    print("   - Tests model's ability to predict between training points")
    print("   - Should have high accuracy if model learned smoothly")
    print("   - Most important for practical use")
    print()
    print("2. EXTRAPOLATION TEST (300 points):")
    print("   - Tests model's ability to predict outside training range")
    print("   - May have lower accuracy - indicates model limitations")
    print("   - Important for understanding model boundaries")
    print()
    print("3. RANDOM TEST (300 points):")
    print("   - General validation across parameter space")
    print("   - Mix of easy and challenging predictions")
    print("   - Good baseline comparison")
    print()
    print("TOTAL: 1000 test points")
    print()
    print("Expected Results:")
    print("- Interpolation: Highest R² (should be close to training R²)")
    print("- Random: Medium R² (general model performance)")  
    print("- Extrapolation: Lowest R² (model limitations)")

# Example usage
if __name__ == "__main__":
    print("=== Neural Network Test Dataset Generator ===")
    
    # Show recommendations
    recommend_test_strategy()
    print("\n" + "="*60 + "\n")
    
    # Generate different types of test datasets
    methods = ['interpolation', 'extrapolation', 'random', 'mixed']
    
    for method in methods:
        print(f"--- {method.upper()} METHOD ---")
        test_data = generate_test_dataset(n_samples=1000, method=method)
        
        # Visualize
        visualize_test_vs_train(test_data)
        
        # Save to CSV
        filename = f'test_parameters_{method}.csv'
        save_test_parameters(test_data, filename)
        
        print(f"Generated {method} test dataset with {len(test_data['n_test'])} points")
        print(f"Run your Vlasov simulations with parameters from {filename}")
        print("\n" + "-"*50 + "\n")
    
    print("=== NEXT STEPS ===")
    print("1. Choose which test method(s) to use based on your goals")
    print("2. Run Vlasov simulations with the generated parameter sets") 
    print("3. Compare neural network predictions with simulation results")
    print("4. Calculate R², RMSE, and identify where the model fails")