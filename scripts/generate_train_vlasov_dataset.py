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
from datetime import datetime
import csv


from spscml.whole_device_model.local_wrapper import apply
from spscml.fusion import fusion_power, bremsstrahlung_power

jax.config.update("jax_enable_x64", True)

ureg = jpu.UnitRegistry()

# parse args & hopefully bound params within realistic ranges:
# Parse command line arguments
def parse_args():
    args = sys.argv[1:]  # Skip the script name
    
    # initial capacitor bank voltage - i dont think i need this
    Vc0 = 40*1e3        # 40 kV


    # Default 
    # n_max = 1e28 * ureg.m**-3 # maximum particle density
    # n_min = 1e20 * ureg.m**-3 # minimum particle density? 
    # T_max = 30e3        # 30 keV
    # T_min = 10          # eV
    # Vp_min = 400.0      # 400 V
    # Vp_max = 10e3       # 10 kV

    # new training bounds
    n_min = 1e23 * ureg.m **-3 
    n_max = 1e28 * ureg.m **-3
    T_min, T_max = 10, 1e4  # eV
    Vp_min, Vp_max = 1, 40e3  # V

    tesseract_name = "vlasov_sheath"  # Default tesseract image
    
    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == '--Vc0' and i + 1 < len(args):
            Vc0 = float(args[i + 1])
            i += 2
        elif args[i] == '--T_max' and i + 1 < len(args):
            T_max = float(args[i + 1])
            i += 2
        elif args[i] == '--T_min' and i + 1 < len(args):
            T_min = float(args[i + 1])
            i += 2
        elif args[i] == '--Vp_max' and i + 1 < len(args):
            Vp_max = float(args[i + 1])
            i += 2
        elif args[i] == '--Vp_min' and i + 1 < len(args):
            Vp_min = float(args[i + 1])
            i += 2
        elif args[i] == '--n_max' and i + 1 < len(args):
            n_max = float(args[i + 1]) * ureg.m**-3
            i += 2
        elif args[i] == '--n_min' and i + 1 < len(args):
            n_min = float(args[i + 1]) * ureg.m**-3
            i += 2
        elif args[i] == '--tesseract' and i + 1 < len(args):
            tesseract = args[i + 1]
            i += 2
        elif args[i] == '--help' or args[i] == '-h':
            print("Usage: python generate_vlasov_dataset.py [--Vc0 VALUE] [--T_max VALUE] [--T_min VALUE] [--Vp_max VALUE] [--Vp_min VALUE] [--n_max VALUE] [--n_min VALUE] [--tesseract NAME]")
            print("  --Vc0 VALUE   Total capacitor voltage in volts (default: 40000)")
            print("  --T_max VALUE     Maximum temperature in eV (default: 30000)")
            print("  --T_min VALUE    Minimum temperature in eV (default: 10)")
            print("  --Vp_max VALUE    Maximum plasma voltage in volts (default: 500)")
            print("  --Vp_min VALUE    Minimum plasma voltage in volts (default: 0)")
            print("  --n_max VALUE    Maximum particle density in m^-3 (default: 1e28)")
            print("  --n_min VALUE    Minimum particle density in m^-3 (default: 1e20)")
            print("  --tesseract NAME  Tesseract name (default: vlasov_sheath)")
            print("  --help, -h    Show this help message")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}")
            print("Use --help for usage information")
            sys.exit(1)

    return Vc0, T_max, T_min, Vp_max, Vp_min, n_max, n_min, tesseract_name


Vc0, T_max, T_min, Vp_max, Vp_min, n_max, n_min, tesseract_name = parse_args()
Lz = 0.5


Z = 1.0
# params we give a shit about n, Vp, 

print(f"Running Vlasov simulations with:")
print(f"  Vc0: {Vc0}")
print(f"  T_max: {T_max}")
print(f"  T_min: {T_min}")
print(f"  Vp_max: {Vp_max}")
print(f"  Vp_min: {Vp_min}")
print(f"  n_max: {n_max}")
print(f"  n_min: {n_min}")
print(f"  tesseract: {tesseract_name}")





if tesseract_name == "vlasov_sheath":
    tesseract_api = vlasov_sheath_tesseract_api
elif tesseract_name == "tanh_sheath":
    tesseract_api = tanh_sheath_tesseract_api

sheath_tx = Tesseract.from_tesseract_api(tesseract_api)




# run vlasov sim with given NUT params
def vlasovSimCallback(Vp_input, T_input, n0_input, tesseract_api) -> dict:
    try:
        n = n0_input * ureg.m**-3
        Vp = Vp_input * ureg.V
        T = T_input * ureg.eV

        j = apply_tesseract(sheath_tx, dict(
            n=jnp.array(n.magnitude), T=jnp.array(T.magnitude), 
            Vp=jnp.array(Vp.magnitude), Lz=jnp.array(0.5)
            ))["j"] * (ureg.A / ureg.m**2)

        # N = ((8*jnp.pi * (1 + Z) * T * n**2) / (ureg.mu0 * j**2)).to(ureg.m**-1)
        # # jax.debug.print("N = {}", N)
        # Ip = (j * N / n).to(ureg.A)
        # A = ((N / n / jnp.pi)**0.5).to(ureg.m)

        return dict(Vp=Vp, T=T, n=n, j=j)
    except KeyboardInterrupt:
        raise

n_vals = jnp.logspace(jnp.log10(n_min.magnitude), jnp.log10(n_max.magnitude), 10)
print("n_vals:", n_vals)
T_vals = jnp.linspace(T_min, T_max, 10)
print("T_vals:", T_vals)
Vp_vals = jnp.linspace(Vp_min, Vp_max, 10)
print("Vp_vals:", Vp_vals)



date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"vlasov_traindata_results_newbounds_{date_str}.csv"
with open(filename, "w") as f:
    f.write("Vp,\tT,\tn,\tj\n")



for n in n_vals:
    for Vp in Vp_vals:
        for T in T_vals:
            try:
                result = vlasovSimCallback(Vp, T, n, tesseract_api)
                print(f"Result for Vp={result['Vp']}, T={result['T']}, n={result['n']}: {result['j']}")
                with open(filename, "a") as f:
                    f.write(f"{result['Vp'].magnitude},\t{result['T'].magnitude},\t{result['n'].magnitude},\t{result['j'].magnitude}\n")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected in simulation. Exiting...")
                raise  # Re-raise the KeyboardInterrupt so outer handler catches it
            except RuntimeError:
                print(f"Simulation failed for Vp={Vp}, T={T}, n={n}")
                with open(filename, "a") as f:
                    f.write(f"{Vp},\t{T},\t{n},\t0\n")
                continue


# # # # trial sim 
# # # Vp = 500.0
# # # T = 20.0
# # # n = 6e22 * ureg.m**-3

# # result = vlasovSimCallback(Vp, T, n, tesseract_api)
# # print(f"Result for Vp={result['Vp']}, T={result['T']}, n={result['n']}: {result['j']}")
# # with open(filename, "a") as f:
# #     f.write(f"{result['Vp'].magnitude},\t{result['T'].magnitude},\t{result['n'].magnitude},\t{result['j'].magnitude}\n")
