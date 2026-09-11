import jax
import jax.numpy as jnp
import pytest
from blackjax_utils.mcmc import (
    _flatten_position,
    get_init_params,
    make_nuts_runner,
    run_chain,
    run_nuts,
)


def log_density_fn(params):
    return -0.5 * jnp.sum(params["x"] ** 2)


# A multi-leaf target whose leaves have different locations and scales, so a
# wrong ravel/unravel permutation shows up as leaves recovering each other's
# moments rather than their own.
MULTILEAF_TARGET = {"a": (0.0, 1.0), "b": (5.0, 0.5), "c": (-3.0, 2.0)}
MULTILEAF_INIT = {
    "a": jnp.zeros(2),
    "b": jnp.full((3,), 5.0),
    "c": jnp.zeros(()),
}


def log_density_multileaf(params):
    return sum(
        -0.5 * jnp.sum(((params[name] - mu) / sd) ** 2)
        for name, (mu, sd) in MULTILEAF_TARGET.items()
    )


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


@pytest.mark.parametrize("flatten", [True, False])
def test_run_nuts_vmap(flatten):
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
        flatten=flatten,
    )

    samples = states.position["x"]
    assert samples.shape == (2, 500, 1)
    mean = jnp.mean(samples)
    std = jnp.std(samples)
    assert jnp.abs(mean) < 0.2
    assert jnp.abs(std - 1.0) < 0.2


@pytest.mark.parametrize("flatten", [True, False])
def test_run_nuts_shard_map_single_device(flatten):
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
        flatten=flatten,
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

    _, info_low = run_nuts(
        **common, warmup_options=dict(target_acceptance_rate=0.6)
    )
    _, info_high = run_nuts(
        **common, warmup_options=dict(target_acceptance_rate=0.95)
    )

    # Dual averaging adapts the step size towards the requested acceptance
    # rate, so the higher target must yield the higher realised rate.
    assert jnp.mean(info_high.acceptance_rate) > jnp.mean(
        info_low.acceptance_rate
    )


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
# Position flattening
# ---------------------------------------------------------------------------


def test_flatten_position_round_trip():
    """Ravelling and unravelling a heterogeneous PyTree is lossless."""
    tree = {"a": jnp.zeros(()), "b": jnp.arange(2.0), "c": jnp.ones((3, 4))}

    flat, unflatten = _flatten_position(tree)

    assert flat.shape == (1 + 2 + 12,)
    restored = unflatten(flat)
    assert jax.tree.structure(restored) == jax.tree.structure(tree)
    for name, leaf in tree.items():
        assert restored[name].shape == leaf.shape
        assert restored[name].dtype == leaf.dtype
        assert jnp.array_equal(restored[name], leaf)


def test_flatten_position_rejects_integer_leaves():
    """Integer leaves are refused rather than silently truncated.

    ravel_pytree would promote the leaf to float for the flat array and then
    cast it back on the way out, rounding samples to integers.
    """
    with pytest.raises(TypeError, match="floating-point"):
        _flatten_position({"n": jnp.arange(3)})

    with pytest.raises(TypeError, match="floating-point"):
        run_chain(
            key=jax.random.PRNGKey(0),
            init_params={"n": jnp.arange(3)},
            target_density=lambda p: -0.5 * jnp.sum(p["n"] ** 2),
            warmup_kwargs={},
            n_warmup=10,
            n_sample=10,
        )


def test_run_nuts_multileaf_dict():
    """A multi-leaf position round-trips through flattening correctly."""
    states, info = run_nuts(
        key=jax.random.PRNGKey(5),
        log_posterior=log_density_multileaf,
        init_params=MULTILEAF_INIT,
        init_sd=0.5,
        n_chain=2,
        n_warmup=400,
        n_sample=1000,
    )

    # Structure is the caller's, not the flat array blackjax actually sampled.
    assert jax.tree.structure(states.position) == jax.tree.structure(
        MULTILEAF_INIT
    )
    assert jax.tree.structure(info.momentum) == jax.tree.structure(
        MULTILEAF_INIT
    )
    assert jax.tree.structure(
        info.trajectory_leftmost_state.position
    ) == jax.tree.structure(MULTILEAF_INIT)

    for name, (mu, sd) in MULTILEAF_TARGET.items():
        samples = states.position[name]
        assert samples.shape == (2, 1000, *MULTILEAF_INIT[name].shape)
        assert jnp.abs(jnp.mean(samples) - mu) < 0.25
        assert jnp.abs(jnp.std(samples) - sd) < 0.25


def test_run_nuts_flatten_opt_out_matches_moments():
    """flatten=True and flatten=False sample the same distribution."""
    common = dict(
        key=jax.random.PRNGKey(6),
        log_posterior=log_density_multileaf,
        init_params=MULTILEAF_INIT,
        init_sd=0.5,
        n_chain=2,
        n_warmup=400,
        n_sample=1000,
    )

    flat_states, _ = run_nuts(**common, flatten=True)
    tree_states, _ = run_nuts(**common, flatten=False)

    assert jax.tree.structure(flat_states.position) == jax.tree.structure(
        tree_states.position
    )
    # Compare moments, not raw draws: the two paths are algebraically but not
    # bitwise identical (XLA reassociates float ops differently on a flat
    # array), so do not tighten this to allclose on the samples themselves.
    for name in MULTILEAF_TARGET:
        flat_samples = flat_states.position[name]
        tree_samples = tree_states.position[name]
        assert flat_samples.shape == tree_samples.shape
        assert jnp.abs(jnp.mean(flat_samples) - jnp.mean(tree_samples)) < 0.3
        assert jnp.abs(jnp.std(flat_samples) - jnp.std(tree_samples)) < 0.3


def test_run_chain_flatten_and_unflatten():
    """run_chain flattens on the un-vmapped, un-jitted path too."""
    states, info = run_chain(
        key=jax.random.PRNGKey(8),
        init_params=MULTILEAF_INIT,
        target_density=log_density_multileaf,
        warmup_kwargs={},
        n_warmup=200,
        n_sample=200,
    )

    assert jax.tree.structure(states.position) == jax.tree.structure(
        MULTILEAF_INIT
    )
    for name, leaf in MULTILEAF_INIT.items():
        assert states.position[name].shape == (200, *leaf.shape)
        assert info.momentum[name].shape == (200, *leaf.shape)


# ---------------------------------------------------------------------------
# Reusable runner
# ---------------------------------------------------------------------------


def test_make_nuts_runner_matches_run_nuts():
    """The factory is the same computation run_nuts performs."""
    key = jax.random.PRNGKey(6)
    common = dict(n_chain=2, n_warmup=400, n_sample=1000)

    expected, _ = run_nuts(
        key=key,
        log_posterior=log_density_multileaf,
        init_params=MULTILEAF_INIT,
        init_sd=0.5,
        **common,
    )
    sample = make_nuts_runner(log_density_multileaf, **common)
    actual, _ = sample(key, MULTILEAF_INIT, 0.5)

    for name in MULTILEAF_TARGET:
        assert jnp.array_equal(actual.position[name], expected.position[name])


def test_make_nuts_runner_is_reusable():
    """Repeated calls work and are deterministic in the key."""
    sample = make_nuts_runner(
        log_density_fn, n_chain=1, n_warmup=100, n_sample=100
    )
    init_params = {"x": jnp.array([1.0])}

    first, _ = sample(jax.random.PRNGKey(20), init_params, 0.1)
    again, _ = sample(jax.random.PRNGKey(20), init_params, 0.1)
    other, _ = sample(jax.random.PRNGKey(21), init_params, 0.1)

    assert first.position["x"].shape == (1, 100, 1)
    assert jnp.array_equal(first.position["x"], again.position["x"])
    assert not jnp.array_equal(first.position["x"], other.position["x"])


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
