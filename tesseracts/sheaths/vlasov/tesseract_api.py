# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Tesseract API module for vlasov_sheath
# Generated by tesseract 0.9.0 on 2025-06-07T21:47:13.287915

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field
from tesseract_core.runtime import Differentiable, Float64
import mlflow

from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

from spscml.sheath_interface import SheathInputSchema, SheathOutputSchema
from spscml.fulltensor_vlasov.sheath_model import make_plasma, reduced_mfp_for_sim, calculate_plasma_current
from spscml.utils import first_moment
from spscml.normalization import plasma_norm
from spscml.plasma import TwoSpeciesPlasma

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

#
# Schemata
#


# Note: This template uses equinox filter_jit to automatically treat non-array
# inputs/outputs as static. As Tesseract scalar objects (e.g. Float32) are
# essentially just wrappers around numpy 0D arrays, they will be considered to
# be dynamic and will be traced by JAX.
# If you want to treat scalar numerical values as static you will need to use
# built-in Python types (e.g. float, int) instead of Float32.


class InputSchema(SheathInputSchema):
    pass


class OutputSchema(SheathOutputSchema):
    pass



#
# Required endpoints
#


# TODO: Add or import your function here, must be JAX-jittable and
# take/return a single pytree as an input/output conforming respectively
# to Input/OutputSchema
@eqx.filter_jit
def apply_helper(inputs: dict) -> dict:
    return calculate_plasma_current(**inputs)


def apply_jit(inputs: dict) -> dict:
    out = apply_helper(inputs)
    return dict(j=out["j_avg"])


def apply_jit_for_jvp(inputs: dict) -> dict:
    out = apply_helper({**inputs, 'adjoint_method': 'jvp'})
    return dict(j=out["j_avg"])


def apply(inputs: InputSchema) -> OutputSchema:
    # Optional: Insert any pre-processing/setup that doesn't require tracing
    # and is only required when specifically running your apply function
    # and not your differentiable endpoints.
    # For example, you might want to set up a logger or mlflow server.
    # Pre-processing should not modify any input that could impact the
    # differentiable outputs in a nonlinear way (a constant shift
    # should be safe)
    norm = plasma_norm(inputs.T, inputs.n)
    plasma = make_plasma(norm)
    sim_mfp = reduced_mfp_for_sim(norm, plasma.Ae, inputs.Lz)
    inputs = inputs.model_dump()

    with mlflow.start_run(run_name="Sheath solve",
                          parent_run_id=inputs["mlflow_parent_run_id"]) as mlflow_run:
        for param in ["Vp", "T", "Lz"]:
            mlflow.log_param(param, inputs[param])
        mlflow.log_param("n_vol", inputs["n"])
        mlflow.log_param("sim_mfp", sim_mfp)
        mlflow.log_param("Ae", plasma.Ae)
        mlflow.log_param("Ai", plasma.Ai)
        mlflow.log_param("Ze", plasma.Ze)
        mlflow.log_param("Zi", plasma.Zi)

        out = apply_helper(inputs)

        #mlflow.log_figure(f_plots(norm, inputs, plasma, out), "plots/f.png")
        #mlflow.log_figure(jE_plots(norm, inputs, plasma, out), "plots/jE.png")

    # Optional: Insert any post-processing that doesn't require tracing
    # For example, you might want to save to disk or modify a non-differentiable
    # output. Again, do not modify any differentiable output in a non-linear way.
    return dict(j=out["j_avg"])



#
# Jax-handled AD endpoints (no need to modify)
#


def jacobian(
    inputs: InputSchema,
    jac_inputs: set[str],
    jac_outputs: set[str],
):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jvp_jit(
        inputs.model_dump(),
        tuple(jvp_inputs),
        tuple(jvp_outputs),
        tangent_vector,
    )


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return vjp_jit(
        inputs.model_dump(),
        tuple(vjp_inputs),
        tuple(vjp_outputs),
        cotangent_vector,
    )


def abstract_eval(abstract_inputs):
    """Calculate output shape of apply from the shape of its inputs."""
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)

    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        inputs = eqx.combine(static_inputs, dynamic_inputs)
        return apply_jit(inputs)

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


#
# Helper functions
#


@eqx.filter_jit
def jac_jit(
    inputs: dict,
    jac_inputs: tuple[str],
    jac_outputs: tuple[str],
):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacrev(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )


@eqx.filter_jit
def jvp_jit(
    inputs: dict, jvp_inputs: tuple[str], jvp_outputs: tuple[str], tangent_vector: dict
):
    filtered_apply = filter_func(apply_jit_for_jvp, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(
    inputs: dict,
    vjp_inputs: tuple[str],
    vjp_outputs: tuple[str],
    cotangent_vector: dict,
):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


def f_plots(norm, inputs, plasma, out):
    v0 = norm["v0"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    Lz = inputs["Lz"]
    vte = jnp.sqrt(1.0 / plasma.Ae) * v0
    vti = jnp.sqrt(1.0 / plasma.Ai) * v0
    axes[0].imshow(out["fe"].T, origin='lower', extent=(-Lz/2, Lz/2, -6*vte.magnitude, 6*vte.magnitude))
    axes[0].set_aspect('auto')
    axes[0].set_title("$f_e$")
    axes[0].set_ylabel("$v [m/s]$")

    axes[1].imshow(out["fi"].T, origin='lower', extent=(-Lz/2, Lz/2, -6*vti.magnitude, 6*vti.magnitude))
    axes[1].set_aspect('auto')
    axes[1].set_title("$f_i$")
    axes[1].set_xlabel("$x / \\lambda_D$")
    axes[1].set_ylabel("$v [m/s]$")

    plt.tight_layout()

    return fig


def jE_plots(norm, inputs, plasma, out):
    v0 = norm["v0"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    Lz = inputs["Lz"]
    axes[0].plot(out["E"])

    axes[1].plot(out["je"], label='je')
    axes[1].plot(out["ji"], label='ji')
    axes[1].plot(out["je"] + out["ji"], label='j')
    axes[1].legend()

    plt.tight_layout()

    return fig
