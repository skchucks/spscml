import sys
sys.path.append("src")
sys.path.append("tesseracts")

import os
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import wdm.tesseract_api as tesseract_api
import sheaths.tanh_sheath.tesseract_api as tanh_sheath_tesseract_api
import sheaths.vlasov.tesseract_api as vlasov_sheath_tesseract_api
import jpu
import optimistix as optx
import scipy.optimize as opt

from spscml.whole_device_model.local_wrapper import apply
from spscml.fusion import fusion_power, bremsstrahlung_power

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()



# parse args & hopefully bound params within realistic ranges:
# Parse command line arguments
def parse_args():
    args = sys.argv[1:]  # Skip the script name
    
    
    # Use the initial plasma from Fig 7 of Shumlak et al. (2012) as an example
    n0 = 6e22 * ureg.m**-3 #intial particle density? 

    # default RLC params
    R = 1.5e-3
    L = 2.0e-7
    C = 222*1e-6

    # Default values/initial values
    Vc0 = 40*1e3  # 50kV
    T = 20.0      # 20 eV
    Vp = 500.0    # 500 V
    tesseract = "tanh_sheath"  # Default tesseract image
    
    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == '--Vc0' and i + 1 < len(args):
            Vc0 = float(args[i + 1])
            i += 2
        elif args[i] == '--T' and i + 1 < len(args):
            T = float(args[i + 1])
            i += 2
        elif args[i] == '--Vp' and i + 1 < len(args):
            Vp = float(args[i + 1])
            i += 2
        elif args[i] == '--tesseract' and i + 1 < len(args):
            tesseract = args[i + 1]
            i += 2
        elif args[i] == '--R' and i + 1 < len(args):
            R = float(args[i + 1])
            i += 2
        elif args[i] == '--L' and i + 1 < len(args):
            L = float(args[i + 1])
            i += 2
        elif args[i] == '--C' and i + 1 < len(args):
            C = float(args[i + 1])
            i += 2
        elif args[i] == '--n0' and i + 1 < len(args):
            n0 = float(args[i + 1]) * ureg.m**-3
            i += 2
        elif args[i] == '--help' or args[i] == '-h':
            print("Usage: python run_wdm.py [--Vc0 VALUE] [--T VALUE] [--Vp VALUE] [--image NAME]")
            print("  --Vc0 VALUE   Total capacitor voltage in volts (default: 40000)")
            print("  --T VALUE     Initial temperature in eV (default: 20.0)")
            print("  --Vp VALUE    Initial plasma voltage in volts (default: 500.0)")
            print("  --tesseract NAME  Tesseract name (default: tanh_sheath)")
            print("  --help, -h    Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)
        
    return Vc0, T, Vp, n0, tesseract, R, L, C


Vc0, T_input, Vp_input, n0, tesseract_name, R, L, C = parse_args()

# params we give a shit about: RLC or Vp, Tinput and n0

print(f"Running WDM simulation with:")
print(f"  Vc0 = {Vc0} V")
print(f"  T = {T_input} eV") 
print(f"  Vp = {Vp_input} V")
print(f"  n0 = {n0} m^-3")
print(f"  Tesseract = {tesseract_name}")
print(f"  R = {R} ohms")
print(f"  L = {L} henries")
print(f"  C = {C} farads")


Lz = 0.5
Lp = -.4e-7
Lp_prime = Lp / Lz
L_tot = L - Lp

Z = 1.0



if tesseract_name == "tanh_sheath":
    tesseract_api = tanh_sheath_tesseract_api
elif tesseract_name == "vlasov_sheath":
    tesseract_api = vlasov_sheath_tesseract_api

sheath_tx = Tesseract.from_tesseract_api(tesseract_api)

def tessCallback(Vp_input, T_input, n0, tesseract_api) -> dict:

    Vp0 = Vp_input * ureg.volts

    T0 = T_input * ureg.eV

    # if isinstance(Vp_i,jax.Array):
    #     Vp0 = Vp_i.magnitude
    # else:
    #     Vp0 = jnp.array(Vp_i.magnitude)

    j = apply_tesseract(sheath_tx, dict(
        n=jnp.array(n0.magnitude), T=jnp.array(T0.magnitude), 
        Vp=jnp.array(Vp0.magnitude), Lz=jnp.array(0.5)
        ))["j"] * (ureg.A / ureg.m**2)
    N = ((8*jnp.pi * (1 + Z) * T0 * n0**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
    # print("N = ", N)

    Ip = (j * N / n0).to(ureg.A)
    # print("Ip = ", Ip)

    a0 = ((N / n0 / jnp.pi)**0.5).to(ureg.m)

    result = apply(dict(
        Vc0=Vc0,
        Ip0=Ip.magnitude,
        a0=a0.magnitude,
        N=N.magnitude,
        Lp_prime=Lp_prime,
        Lz=Lz,
        R=R, L=L, C=C,
        dt=5e-8,
        t_end=1e-5,
        mlflow_parent_run_id=None
    ), sheath_tx)

    n = result['n']
    T = result['T']

    a = jnp.sqrt(N.magnitude / n / jnp.pi)  # plasma radius
    result['a'] = a
    result['fusion_power'] = fusion_power(n, Lz, a, T)
    result['bremsstrahlung_power'] = bremsstrahlung_power(n, Lz, a, T)

    return result
# bound params within realistic ranges

def objective_fn(Vp):
    result = tessCallback(Vp, T_input, n0, tesseract_api)

    total_power = result['fusion_power'] - result['bremsstrahlung_power']
    dt = result['ts'][1] - result['ts'][0]  # Assuming ts is a 1D array of time steps
    return jnp.sum(-total_power * dt)



def scipy_vg(_Vp):
    """Value and gradient function for scipy optimization."""
    # __Vp = jnp.array(_Vp)
    __Vp = jnp.array(extract_value(_Vp))  

    fval, gradVal = fgrad_fn(__Vp)
    fval = np.array(fval)
    gradVal = np.array(gradVal)
    print(f"Vp={__Vp}, Objective={fval}, Gradient={gradVal}")
    return fval, gradVal


def extract_value(x):
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return x.item()
    else:
        return x

def print_results(res):
# current iteration value
    print(f"Iteration: {res.nit}, Objective: {res.fun:.2f}, Vp: {res.x[0]:.2f}, Gradient: {res.jac[0]:.2f}, Success: {res.success}, Message: {res.message}")


obj_fn_value = objective_fn(Vp_input)
print(f"Objective function value for Vp={Vp_input} V: {obj_fn_value:.2f} W")

# try out grad of objective function with respect to Vp

f = lambda Vp: objective_fn(Vp)
fgrad_fn = jax.value_and_grad(f)

    
fval, gradVal = fgrad_fn(Vp_input)
print(f"Value and Gradient of objective function at Vp={Vp_input} f: {fval:.5f} W df/dx: {gradVal:.5f} W/V")


res = opt.minimize(scipy_vg, Vp_input, method='L-BFGS-B', jac=True, options={'disp': True, 'maxiter': 100}, bounds=[(400, 10e3)])


print(f"Optimized Vp: {res.x[0]:.2f} V")
print(f"Objective function value at optimized Vp: {res.fun:.2f} W")
print(f"Gradient at optimized Vp: {res.jac[0]:.2f} W/V")
print(f"Success: {res.success}, Message: {res.message}")
print(f"Number of iterations: {res.nit}")

