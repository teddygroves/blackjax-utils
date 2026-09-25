"""Sampling from a posterior whose likelihood involves a diffrax ODE solve.

These tests exist because chain parallelism interacts badly with diffrax's
default adjoint. The single-device `jax.vmap` path is fine, but `jax.pmap`
and `shard_map` both fail while tracing the gradient, so a model that samples
happily on one device breaks as soon as chains are spread over several.

See `test_diffrax_pmap_default_adjoint_fails` for the precise cause.
"""

import jax
import jax.numpy as jnp
import pytest
from blackjax_utils import run_nuts

diffrax = pytest.importorskip("diffrax")

requires_4_devices = pytest.mark.skipif(
    jax.local_device_count() < 4,
    reason="Requires >= 4 CPU devices (set JAX_NUM_CPU_DEVICES=4)",
)

# Exponential decay, dy/dt = -k * y, observed at TS with known noise. The
# smallest thing that is still a real ODE solve: the solver's step count
# depends on the sampled parameters, so the adaptive `while_loop` that trips
# pmap up is genuinely exercised.
T1 = 5.0
TS = jnp.linspace(0.5, T1, 10)
SIGMA = 0.05
TRUE = {"log_k": jnp.log(0.7), "log_y0": jnp.log(2.0)}


def solve(params, adjoint):
    """Solve the decay ODE at TS. Returns ``(ys, solver_succeeded)``.

    ``throw=False`` is what makes this usable inside MCMC. The sampler will
    propose parameters that make the solve hopeless, and the default
    ``throw=True`` turns the first such proposal into a hard crash of the
    whole run. Reporting failure in ``sol.result`` instead lets the log
    density return ``-inf`` so the proposal is simply rejected. It also drops
    the host callback that raises the error, which is itself a hazard under
    ``pmap``.
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, k: -k * y),
        diffrax.Tsit5(),
        t0=0.0,
        t1=T1,
        dt0=0.01,
        y0=jnp.exp(params["log_y0"]),
        args=jnp.exp(params["log_k"]),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
        saveat=diffrax.SaveAt(ts=TS),
        max_steps=2**14,
        adjoint=adjoint,
        throw=False,
    )
    return sol.ys, sol.result == diffrax.RESULTS.successful


YOBS = solve(TRUE, diffrax.DirectAdjoint())[0] + SIGMA * jax.random.normal(
    jax.random.PRNGKey(0), (TS.size,)
)


def make_log_posterior(adjoint):
    """Standard-normal priors on both log parameters, Gaussian likelihood."""

    def log_posterior(params):
        ys, ok = solve(params, adjoint)
        prior = sum(-0.5 * jnp.sum(v**2) for v in params.values())
        # nan_to_num before the arithmetic, not after: a failed solve leaves
        # nans in ys, and a nan reaching the reverse pass would poison the
        # gradient even on the branch jnp.where discards.
        resid = (YOBS - jnp.nan_to_num(ys)) / SIGMA
        return jnp.where(ok, prior - 0.5 * jnp.sum(resid**2), -jnp.inf)

    return log_posterior


def sample(adjoint, chain_map, n_chain=4):
    return run_nuts(
        key=jax.random.PRNGKey(1),
        log_posterior=make_log_posterior(adjoint),
        init_params={"log_k": jnp.array(0.0), "log_y0": jnp.array(0.5)},
        init_sd=0.05,
        n_chain=n_chain,
        n_warmup=10,
        n_sample=10,
        chain_map=chain_map,
        max_num_doublings=2,
    )


def check_draws(states):
    for name in TRUE:
        draws = states.position[name]
        assert draws.shape == (4, 10)
        assert jnp.all(jnp.isfinite(draws)), name


def test_diffrax_vmap_default_adjoint():
    """On one device the default adjoint samples with vmap."""
    states, _ = sample(diffrax.RecursiveCheckpointAdjoint(), jax.vmap)
    check_draws(states)


@requires_4_devices
def test_diffrax_pmap_default_adjoint_fails():
    """pmap cannot differentiate through diffrax's default adjoint.

    `RecursiveCheckpointAdjoint` runs the solve inside
    `equinox.internal.while_loop(kind="checkpointed")`, whose custom
    derivative rule closure-converts the loop body. Under `pmap` the body is
    re-traced with different abstract values than the ones the conversion was
    built from, and the rule rejects them.

    This is an equinox/diffrax limitation, not a blackjax-utils one: plain
    `jax.pmap(jax.grad(...))` over such a loop fails identically, with no
    blackjax involved. The test pins the failure so that a fix upstream shows
    up here as an unexpected pass rather than going unnoticed.
    """
    with pytest.raises(ValueError, match="Closure-converted function"):
        sample(diffrax.RecursiveCheckpointAdjoint(), jax.pmap)


@requires_4_devices
@pytest.mark.parametrize(
    "adjoint",
    [diffrax.DirectAdjoint(), diffrax.BacksolveAdjoint()],
    ids=["DirectAdjoint", "BacksolveAdjoint"],
)
def test_diffrax_pmap_works_with_other_adjoints(adjoint):
    """Adjoints that avoid the checkpointed loop parallelise over devices."""
    states, _ = sample(adjoint, jax.pmap)
    check_draws(states)
