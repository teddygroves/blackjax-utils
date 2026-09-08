from functools import partial
from typing import Any, Callable

import blackjax
import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import PRNGKeyArray, PyTree


def get_init_params(key: PRNGKeyArray, base: PyTree, sd: PyTree | None) -> PyTree:
    """Initialize parameters by adding jitter to a base value.

    Args:
        key: A JAX PRNG key.
        base: The base parameters (PyTree) to jitter around.
        sd: The standard deviation of the jitter. If None, no jitter is applied
            (sd is treated as zero).

    Returns:
        The jittered parameters with the same structure as `base`.
    """

    def jitter_leaf(key: PRNGKeyArray, base_leaf: Any, sd_leaf: Any) -> Any:
        return base_leaf + jax.random.normal(key, shape=base_leaf.shape) * sd_leaf

    flat_means, treedef = jax.tree.flatten(base)
    keys = jax.random.split(key, num=len(flat_means))
    keytree = jax.tree.unflatten(treedef, keys)
    if sd is None:
        sd = jax.tree.map(jnp.zeros_like, base)
    elif isinstance(sd, (int, float)) or (hasattr(sd, "shape") and sd.shape == ()):
        sd_scalar = sd
        sd = jax.tree.map(lambda _: sd_scalar, base)
    return jax.tree.map(jitter_leaf, keytree, base, sd)


def _flatten_position(position: PyTree) -> tuple[Any, Callable[[Any], PyTree]]:
    """Ravel a position PyTree into a single 1-D array.

    Blackjax is faster when the position it manipulates is one flat array
    rather than a multi-leaf PyTree: JAX allocates a separate buffer for every
    leaf carried through the sampling `jax.lax.scan`, which multiplies HLO
    nodes and inflates both compile and run time.  See
    https://blackjax-devs.github.io/blackjax/examples/speed_up_guide.html

    Args:
        position: The position PyTree to ravel.

    Returns:
        A tuple ``(flat, unflatten)`` where ``flat`` is a 1-D array and
        ``unflatten`` maps such an array back to the original structure.

    Raises:
        TypeError: If any leaf has a non-floating-point dtype. ``ravel_pytree``
            promotes such a leaf to float for the flat array and then truncates
            it on the way back, which would silently corrupt samples.
    """
    for path, leaf in jax.tree_util.tree_flatten_with_path(position)[0]:
        dtype = jnp.result_type(leaf)
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise TypeError(
                f"Cannot flatten position leaf "
                f"{jax.tree_util.keystr(path) or '<root>'} with dtype {dtype}: "
                "flattening requires floating-point leaves. Cast the leaf to a "
                "float dtype, or pass flatten=False."
            )
    return ravel_pytree(position)


def _unflatten_state(state: Any, unflatten_draws: Callable) -> Any:
    """Restore PyTree positions in a stacked trace of chain states.

    Args:
        state: A blackjax ``HMCState`` (NUTS reuses it) whose leaves are
            stacked over draws, with a flat position.
        unflatten_draws: ``jax.vmap`` of the unflatten function, mapping over
            the draw axis.

    Returns:
        The same state with ``position`` and ``logdensity_grad`` restored to
        the original PyTree structure. ``logdensity`` is a scalar per draw and
        is left alone.
    """
    return state._replace(
        position=unflatten_draws(state.position),
        logdensity_grad=unflatten_draws(state.logdensity_grad),
    )


def _unflatten_info(info: Any, unflatten_draws: Callable) -> Any:
    """Restore PyTree positions in a stacked trace of ``NUTSInfo``.

    Args:
        info: A blackjax ``NUTSInfo`` whose leaves are stacked over draws.
        unflatten_draws: ``jax.vmap`` of the unflatten function, mapping over
            the draw axis.

    Returns:
        The same info with every position-space field restored to the original
        PyTree structure. The remaining fields (``is_divergent``,
        ``is_turning``, ``energy``, ``num_trajectory_expansions``,
        ``num_integration_steps``, ``acceptance_rate``) are scalars per draw.
    """

    def unflatten_integrator_state(s: Any) -> Any:
        return s._replace(
            position=unflatten_draws(s.position),
            momentum=unflatten_draws(s.momentum),
            logdensity_grad=unflatten_draws(s.logdensity_grad),
        )

    return info._replace(
        momentum=unflatten_draws(info.momentum),
        trajectory_leftmost_state=unflatten_integrator_state(
            info.trajectory_leftmost_state
        ),
        trajectory_rightmost_state=unflatten_integrator_state(
            info.trajectory_rightmost_state
        ),
    )


def inference_loop(
    key: PRNGKeyArray,
    tuned_params: dict[str, Any],
    initial_state: PyTree,
    num_samples: int,
    log_posterior: Callable,
    **static_params: Any,
) -> tuple[PyTree, PyTree]:
    """Run a sampling loop.

    Args:
        key: A JAX PRNG key.
        tuned_params: Dict of tuned NUTS parameters (step_size, inverse_mass_matrix, etc.)
        initial_state: Initial state for sampling from warmup.
        num_samples: Number of samples to draw.
        log_posterior: The log-probability density function of the target distribution.
        **static_params: Static algorithm parameters (e.g., max_num_doublings) that
            are not tuned during warmup.

    Returns:
        A tuple containing (states, info) where states is the tree of samples
        and info contains sampling diagnostics.

    Note:
        The scan body is deliberately not jitted: ``jax.lax.scan`` is a
        primitive, so the loop is compiled as one XLA computation either way,
        and a nested jit only adds trace-time work.  For best performance jit
        the whole chain (warmup and sampling together) rather than this
        function alone -- ``run_nuts`` does that for you.
    """
    # Merge static params with tuned params for the kernel
    all_params = {**tuned_params, **static_params}
    kernel = blackjax.nuts(log_posterior, **all_params).step

    def one_step(state: Any, rng_key: PRNGKeyArray) -> tuple[Any, tuple[Any, Any]]:
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(key, num_samples)
    _, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info


def run_chain(
    key: PRNGKeyArray,
    init_params: PyTree,
    target_density: Callable,
    warmup_kwargs: dict[str, Any],
    n_warmup: int,
    n_sample: int,
    *,
    flatten: bool = True,
    **sample_kwargs: Any,
) -> tuple[PyTree, PyTree]:
    """Run warmup and sampling for a single chain.

    This function runs the full MCMC workflow for a single chain: warmup
    (window adaptation) followed by NUTS sampling.

    Kernel parameters that appear in ``warmup_kwargs`` but not in
    ``sample_kwargs`` apply to warmup only: they are stripped from the tuned
    parameters before sampling, so sampling falls back to the blackjax default.

    This function is traceable but not itself jitted: wrap it in `jax.jit` (or
    call it through `run_nuts`, which does so) to compile warmup and sampling
    as a single cached computation. The returned samples always have the
    structure of ``init_params``, whether or not ``flatten`` is set.

    Args:
        key: A JAX PRNG key.
        init_params: Initial parameter values.
        target_density: The log-density function to sample from.
        warmup_kwargs: Additional arguments passed to `blackjax.window_adaptation`.
            As well as kernel parameters such as ``max_num_doublings``, this may
            contain warmup-only arguments that `blackjax.nuts` does not accept:
            ``initial_step_size``, ``target_acceptance_rate``,
            ``is_mass_matrix_diagonal``, ``initial_inverse_mass_matrix``,
            ``imm_shrinkage_to_previous`` and ``adaptation_info_fn``.
        n_warmup: Number of warmup (adaptation) steps.
        n_sample: Number of sampling steps.
        flatten: Whether to ravel the position into a single 1-D array before
            passing it to blackjax, restoring the original structure on the way
            out. See `run_nuts` for the trade-offs.
        **sample_kwargs: Static parameters passed to the NUTS kernel during
            sampling (e.g., max_num_doublings).

    Returns:
        A tuple containing (states, info) where states is the tree of samples
        and info contains sampling diagnostics.
    """
    # The flattening lives here, inside the per-chain function, so that
    # ravel_pytree sees one chain's leaves and the unflatten closure is built
    # and consumed within the same trace.  Under vmap the leaves are tracers
    # whose .shape is the unbatched per-chain shape, which is what we want.
    # Moving this out of run_chain would require rebuilding unflatten from an
    # unbatched, post-jitter template or samples come back mis-shaped.
    if flatten:
        initial_position, unflatten = _flatten_position(init_params)

        def density(position: Any) -> Any:
            return target_density(unflatten(position))
    else:
        initial_position, unflatten = init_params, None
        density = target_density

    warmup_key, sample_key = jax.random.split(key)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        density,
        **warmup_kwargs,
    )
    (warmed_up_state, tuned_params), _ = warmup.run(
        warmup_key,
        initial_position,
        n_warmup,  # type: ignore
    )
    warmup_only = set(warmup_kwargs) - set(sample_kwargs)
    tuned_params = {k: v for k, v in tuned_params.items() if k not in warmup_only}
    states, info = inference_loop(
        sample_key,
        tuned_params,
        warmed_up_state,
        num_samples=n_sample,
        log_posterior=density,
        **sample_kwargs,
    )
    if unflatten is not None:
        unflatten_draws = jax.vmap(unflatten)
        states = _unflatten_state(states, unflatten_draws)
        info = _unflatten_info(info, unflatten_draws)
    return states, info


def make_nuts_runner(
    log_posterior: Callable,
    n_chain: int = 4,
    n_warmup: int = 500,
    n_sample: int = 500,
    chain_map: Callable = jax.vmap,
    sampling_options: dict[str, Any] | None = None,
    warmup_options: dict[str, Any] | None = None,
    flatten: bool = True,
    **kwargs: Any,
) -> Callable[..., tuple[PyTree, PyTree]]:
    """Build a reusable sampler whose compiled code is cached across calls.

    `run_nuts` constructs a fresh `jax.jit` on every call, so its compilation
    cache is always empty and calling it in a loop recompiles each time. This
    factory builds the jitted, chain-mapped function once; the callable it
    returns then recompiles only when the shape or dtype signature of
    ``init_params`` changes. Use it to sample repeatedly from one model, e.g.
    across datasets or seeds.

    Args:
        log_posterior: The log-probability density function of the target
            distribution.
        n_chain: Number of MCMC chains to run.
        n_warmup: Number of warmup (adaptation) steps per chain.
        n_sample: Number of sampling steps per chain.
        chain_map: Chain parallelism strategy, as for `run_nuts`.
        sampling_options: Sampling-stage overrides, as for `run_nuts`.
        warmup_options: Warmup-stage overrides, as for `run_nuts`.
        flatten: Whether to ravel the position, as for `run_nuts`.
        **kwargs: Arguments forwarded to both stages, as for `run_nuts`.

    Returns:
        A callable ``sample(key, init_params, init_sd=None)`` returning
        ``(states, info)``, exactly as `run_nuts` does.

    Example:
        >>> sample = make_nuts_runner(log_density, n_chain=4)  # doctest: +SKIP
        >>> for i, key in enumerate(jax.random.split(key, 10)):  # doctest: +SKIP
        ...     states, info = sample(key, init_params)  # compiles once
    """
    warmup_kwargs: dict[str, Any] = {**kwargs, **(warmup_options or {})}
    sample_kwargs: dict[str, Any] = {**kwargs, **(sampling_options or {})}

    # jit goes inside chain_map so that warmup and sampling compile as a single
    # cached XLA computation per chain, and so the same placement is correct for
    # vmap, pmap and shard_map alike (jit-of-pmap is discouraged).  Every
    # non-array argument is bound by the partial, so the jitted callable takes
    # only the two traced arguments and needs no static_argnums.
    run_this_chain = jax.jit(
        partial(
            run_chain,
            target_density=log_posterior,
            warmup_kwargs=warmup_kwargs,
            n_warmup=n_warmup,
            n_sample=n_sample,
            flatten=flatten,
            **sample_kwargs,
        )
    )
    run_these_chains = chain_map(run_this_chain, in_axes=(0, 0))
    jitter_chains = jax.vmap(get_init_params, in_axes=(0, None, None))

    def sample(
        key: PRNGKeyArray,
        init_params: PyTree,
        init_sd: PyTree | None = None,
    ) -> tuple[PyTree, PyTree]:
        key1, key2 = jax.random.split(key)
        init_keys = jax.random.split(key1, n_chain)
        sample_keys = jax.random.split(key2, n_chain)
        chain_init_params = jitter_chains(init_keys, init_params, init_sd)
        return run_these_chains(sample_keys, chain_init_params)

    return sample


def run_nuts(
    key: PRNGKeyArray,
    log_posterior: Callable,
    init_params: PyTree,
    init_sd: PyTree | None = None,
    n_chain: int = 4,
    n_warmup: int = 500,
    n_sample: int = 500,
    chain_map: Callable = jax.vmap,
    sampling_options: dict[str, Any] | None = None,
    warmup_options: dict[str, Any] | None = None,
    flatten: bool = True,
    **kwargs: Any,
) -> tuple[PyTree, PyTree]:
    """Run NUTS sampling with parallelization across multiple chains.

    This function coordinates the full MCMC workflow: initialization, warmup
    (adaptation), and sampling. Chain parallelism is controlled by the
    ``chain_map`` argument.

    Args:
        key: A JAX PRNG key.
        log_posterior: The log-probability density function of the target distribution.
        init_params: Initial values for the parameters. Will be jittered by ``init_sd``.
        init_sd: Standard deviation for jittering the initial parameters. If None,
            start exactly at ``init_params``.
        n_chain: Number of MCMC chains to run.
        n_warmup: Number of warmup (adaptation) steps per chain.
        n_sample: Number of sampling steps per chain.
        chain_map: A callable with the same interface as ``jax.vmap`` / ``jax.pmap``
            (i.e. ``chain_map(func, in_axes=...)`` returns a vectorized function).
            Defaults to ``jax.vmap`` for single-device vectorization. Pass
            ``jax.pmap`` for multi-device SPMD parallelism, or a
            ``jax.experimental.shard_map.shard_map`` partial for explicit
            sharding control.
        sampling_options: Optional dictionary of keyword arguments forwarded
            to the NUTS kernel during sampling. When provided, these values
            are added to, and override, the corresponding ``**kwargs`` for the
            sampling stage only. Warmup still uses the original ``**kwargs``
            values.
        warmup_options: Optional dictionary of keyword arguments forwarded to
            ``blackjax.window_adaptation`` during warmup. When provided, these
            values are added to, and override, the corresponding ``**kwargs``
            for the warmup stage only. Sampling still uses the original
            ``**kwargs`` values.

            This is the only way to set arguments that ``window_adaptation``
            accepts but ``blackjax.nuts`` rejects, since ``**kwargs`` values
            reach both stages. Those arguments are ``initial_step_size``,
            ``target_acceptance_rate``, ``is_mass_matrix_diagonal``,
            ``initial_inverse_mass_matrix``, ``imm_shrinkage_to_previous`` and
            ``adaptation_info_fn``.
        flatten: Whether to ravel the position into a single 1-D array before
            it reaches blackjax, restoring the original PyTree structure in the
            returned samples. This is the blackjax speed-up guide's advice for
            multi-leaf positions and is measurably faster; it is a no-op for a
            position that is already a single array. Two caveats: every leaf
            must have a floating-point dtype (a ``TypeError`` is raised
            otherwise), and because flattening is only algebraically -- not
            bitwise -- equivalent, a given key produces different (equally
            valid) draws than ``flatten=False``.
        **kwargs: Additional keyword arguments forwarded to both
            ``blackjax.window_adaptation`` (warmup) and the NUTS kernel
            (sampling). Use ``warmup_options`` or ``sampling_options`` to set
            or override stage-specific values.

    Returns:
        A tuple containing (states, info) where:
        - states: Tree of posterior samples with shape (n_chain, n_sample, ...)
        - info: Dictionary with sampling diagnostics (e.g., divergence info)

    Note:
        Each call builds a fresh `jax.jit`, whose cache is therefore empty, so
        calling this function in a loop recompiles every time. Use
        `make_nuts_runner` to sample repeatedly from the same model.
    """
    runner = make_nuts_runner(
        log_posterior,
        n_chain=n_chain,
        n_warmup=n_warmup,
        n_sample=n_sample,
        chain_map=chain_map,
        sampling_options=sampling_options,
        warmup_options=warmup_options,
        flatten=flatten,
        **kwargs,
    )
    return runner(key, init_params, init_sd)
