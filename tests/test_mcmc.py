import jax
import jax.numpy as jnp
import pytest
from blackjax_utils.mcmc import get_init_params, run_chain, run_nuts


def log_density_fn(params):
    return -0.5 * jnp.sum(params["x"] ** 2)


def _make_shard_chain_map(n_chain: int):
    """Build a ``chain_map`` adapter for ``shard_map``.

    Uses a sub-mesh of exactly ``n_chain`` devices so that chains map
    1:1 onto devices and the remaining devices stay free.  The adapter
    handles the API mismatch between ``shard_map`` (which preserves the
    logical mesh axis inside the function) and ``vmap``/``pmap`` (which
    strip it).
    """
    from jax import shard_map
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(jax.devices()[:n_chain], axis_names=("chains",))

    def chain_map(func, in_axes):
        def wrapped(key, params):
            # shard_map preserves the logical mesh axis; each device
            # sees a leading dim of 1.  Index at 0 to strip it (safe
            # during JIT tracing, unlike squeeze which requires the
            # axis size to be statically known as 1).
            key = key[0]
            params = jax.tree.map(lambda x: x[0], params)
            states, info = func(key, params)
            # Restore the chain axis that shard_map expects.
            states = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), states)
            info = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), info)
            return states, info

        return shard_map(
            wrapped,
            mesh=mesh,
            in_specs=(PartitionSpec("chains"), PartitionSpec("chains")),
            out_specs=PartitionSpec("chains"),
            check_vma=False,
        )

    return chain_map


# ---------------------------------------------------------------------------
# Single-device tests
# ---------------------------------------------------------------------------


def test_get_init_params_no_jitter():
    key = jax.random.PRNGKey(0)
    base_params = {"x": jnp.array([1.0, 2.0])}

    params_none = get_init_params(key, base_params, sd=None)
    assert jnp.array_equal(params_none["x"], base_params["x"])

    params_zero = get_init_params(key, base_params, sd=0.0)
    assert jnp.array_equal(params_zero["x"], base_params["x"])


def test_get_init_params_jitter():
    key = jax.random.PRNGKey(0)
    base_params = {"x": jnp.array([1.0, 2.0])}
    sd = 0.1

    params_jitter = get_init_params(key, base_params, sd=sd)
    assert not jnp.array_equal(params_jitter["x"], base_params["x"])
    assert params_jitter["x"].shape == base_params["x"].shape


def test_run_nuts_vmap():
    """Explicit vmap chain mapping (the default)."""
    key = jax.random.PRNGKey(2)
    init_params = {"x": jnp.array([10.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=2,
        n_warmup=200,
        n_sample=500,
        chain_map=jax.vmap,
        max_num_doublings=5,
    )

    samples = states.position["x"]
    assert samples.shape == (2, 500, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2


def test_run_nuts_shard_map_single_device():
    """shard_map with a single device."""
    n_chain = 1
    key = jax.random.PRNGKey(3)
    init_params = {"x": jnp.array([10.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=n_chain,
        n_warmup=200,
        n_sample=500,
        chain_map=_make_shard_chain_map(n_chain),
        max_num_doublings=5,
    )

    samples = states.position["x"]
    assert samples.shape == (n_chain, 500, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2


def test_run_chain_forwards_sample_kwargs():
    """Test that run_chain passes sample_kwargs through to inference_loop."""
    key = jax.random.PRNGKey(42)
    init_params = {"x": jnp.array([1.0])}

    states, info = run_chain(
        key=key,
        init_params=init_params,
        target_density=log_density_fn,
        warmup_kwargs={},
        n_warmup=100,
        n_sample=100,
        max_num_doublings=1,
    )

    samples = states.position["x"]
    assert samples.shape == (100, 1)


def test_run_nuts_sampling_options_override():
    """Test that sampling_options overrides kwargs for the sampling stage only."""
    key = jax.random.PRNGKey(99)
    init_params = {"x": jnp.array([1.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=0.1,
        n_chain=1,
        n_warmup=100,
        n_sample=200,
        max_num_doublings=10,
        sampling_options=dict(max_num_doublings=1),
    )

    samples = states.position["x"]
    assert samples.shape == (1, 200, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.3
    assert jnp.abs(std - 1.0) < 0.3


def test_run_nuts_warmup_options():
    """warmup_options accepts arguments that the NUTS kernel rejects.

    ``initial_step_size`` and ``target_acceptance_rate`` are
    ``window_adaptation`` parameters that ``blackjax.nuts`` does not take, so
    passing them as plain ``**kwargs`` (which reach both stages) raises
    TypeError.  ``warmup_options`` is the only channel that works.
    """
    key = jax.random.PRNGKey(11)
    init_params = {"x": jnp.array([1.0])}

    with pytest.raises(TypeError):
        run_nuts(
            key=key,
            log_posterior=log_density_fn,
            init_params=init_params,
            n_chain=1,
            n_warmup=100,
            n_sample=200,
            initial_step_size=0.1,
        )

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=0.1,
        n_chain=1,
        n_warmup=100,
        n_sample=200,
        warmup_options=dict(initial_step_size=0.1, target_acceptance_rate=0.9),
    )

    samples = states.position["x"]
    assert samples.shape == (1, 200, 1)
    assert jnp.abs(jnp.mean(samples)) < 0.3
    assert jnp.abs(jnp.std(samples) - 1.0) < 0.3


def test_run_nuts_warmup_options_take_effect():
    """A warmup-only argument changes the adapted sampler, not just plumbing."""
    key = jax.random.PRNGKey(12)
    init_params = {"x": jnp.array([10.0])}
    common = dict(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=1,
        n_warmup=400,
        n_sample=500,
    )

    _, info_low = run_nuts(**common, warmup_options=dict(target_acceptance_rate=0.6))
    _, info_high = run_nuts(**common, warmup_options=dict(target_acceptance_rate=0.95))

    # Dual averaging adapts the step size towards the requested acceptance
    # rate, so the higher target must yield the higher realised rate.
    assert jnp.mean(info_high.acceptance_rate) > jnp.mean(info_low.acceptance_rate)


def test_run_nuts_warmup_options_do_not_leak_into_sampling():
    """Kernel kwargs set for warmup only must not reach the sampling kernel.

    blackjax returns any kernel-level kwarg given to ``window_adaptation``
    inside its tuned parameters, so without the warmup-only strip in
    ``run_chain`` sampling would inherit ``max_num_doublings=1`` and be capped
    at a single leapfrog step per draw.
    """
    states, info = run_nuts(
        key=jax.random.PRNGKey(13),
        log_posterior=log_density_fn,
        init_params={"x": jnp.array([10.0])},
        init_sd=1.0,
        n_chain=1,
        n_warmup=400,
        n_sample=500,
        warmup_options=dict(max_num_doublings=1),
    )

    samples = states.position["x"]
    assert samples.shape == (1, 500, 1)
    # max_num_doublings=1 allows at most 2 integration steps.
    assert jnp.max(info.num_integration_steps) > 2
    assert jnp.abs(jnp.mean(samples)) < 0.3
    assert jnp.abs(jnp.std(samples) - 1.0) < 0.3


def test_run_nuts_both_option_dicts():
    """warmup_options and sampling_options coexist and each wins its stage."""
    states, info = run_nuts(
        key=jax.random.PRNGKey(14),
        log_posterior=log_density_fn,
        init_params={"x": jnp.array([1.0])},
        init_sd=0.1,
        n_chain=1,
        n_warmup=200,
        n_sample=200,
        max_num_doublings=10,
        warmup_options=dict(max_num_doublings=2),
        sampling_options=dict(max_num_doublings=5),
    )

    samples = states.position["x"]
    assert samples.shape == (1, 200, 1)
    # sampling_options wins for sampling: at most 2**5 integration steps.
    assert jnp.max(info.num_integration_steps) <= 2**5


def test_run_chain_strips_warmup_only_kwargs():
    """run_chain handles warmup-only kwargs on the un-vmapped, un-jitted path."""
    states, info = run_chain(
        key=jax.random.PRNGKey(15),
        init_params={"x": jnp.array([1.0])},
        target_density=log_density_fn,
        warmup_kwargs=dict(initial_step_size=0.5, max_num_doublings=1),
        n_warmup=100,
        n_sample=100,
    )

    assert states.position["x"].shape == (100, 1)
    assert jnp.max(info.num_integration_steps) > 2


# ---------------------------------------------------------------------------
# Multi-device tests  (run with JAX_NUM_CPU_DEVICES=4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    jax.local_device_count() < 4,
    reason="Requires ≥ 4 CPU devices (set JAX_NUM_CPU_DEVICES=4)",
)
def test_run_nuts_pmap():
    """pmap across all 4 devices."""
    n_chain = 4
    key = jax.random.PRNGKey(1)
    init_params = {"x": jnp.array([10.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=n_chain,
        n_warmup=200,
        n_sample=500,
        chain_map=jax.pmap,
        max_num_doublings=5,
    )

    samples = states.position["x"]
    assert samples.shape == (n_chain, 500, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2


@pytest.mark.skipif(
    jax.local_device_count() < 4,
    reason="Requires ≥ 4 CPU devices (set JAX_NUM_CPU_DEVICES=4)",
)
def test_run_nuts_shard_map_subset_devices():
    """shard_map with a sub-mesh: 2 chains on 2 of 4 devices."""
    n_chain = 2
    key = jax.random.PRNGKey(4)
    init_params = {"x": jnp.array([10.0])}

    states, info = run_nuts(
        key=key,
        log_posterior=log_density_fn,
        init_params=init_params,
        init_sd=1.0,
        n_chain=n_chain,
        n_warmup=200,
        n_sample=500,
        chain_map=_make_shard_chain_map(n_chain),
        max_num_doublings=5,
    )

    samples = states.position["x"]
    assert samples.shape == (n_chain, 500, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2
