"""Benchmarks for the blackjax speed-up guide recommendations.

These measure the two things the guide motivates: ravelling a multi-leaf
position (guide section 3) and reusing a jitted chain function (sections 1
and 5).  See
https://blackjax-devs.github.io/blackjax/examples/speed_up_guide.html

Timings are disabled by default (``--benchmark-disable`` in
``pyproject.toml``), so a normal test run executes each benchmark body once as
a smoke test and reports nothing.  To measure:

    uv run python -m pytest tests/test_benchmarks.py --benchmark-enable

The problem size can be varied with the ``BENCH_*`` environment variables
below, e.g. ``BENCH_N_CHAIN=8 BENCH_N_LEAVES=16``.
"""

import os

import jax
import jax.numpy as jnp
import pytest
from blackjax_utils import make_nuts_runner, run_nuts


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


N_LEAVES = _env_int("BENCH_N_LEAVES", 8)
LEAF_SIZE = _env_int("BENCH_LEAF_SIZE", 5)
N_CHAIN = _env_int("BENCH_N_CHAIN", 4)
N_WARMUP = _env_int("BENCH_N_WARMUP", 500)
N_SAMPLE = _env_int("BENCH_N_SAMPLE", 500)

SAMPLING_KWARGS = dict(n_chain=N_CHAIN, n_warmup=N_WARMUP, n_sample=N_SAMPLE)
KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def target():
    """A multi-leaf standard normal: one named parameter per leaf."""
    names = [f"p{i}" for i in range(N_LEAVES)]
    init_params = {name: jnp.zeros(LEAF_SIZE) for name in names}

    def log_density(params):
        return -0.5 * sum(jnp.sum(params[name] ** 2) for name in names)

    return log_density, init_params


@pytest.mark.benchmark(group="run_nuts (fresh jit per call)")
@pytest.mark.parametrize("flatten", [False, True], ids=["pytree", "flat"])
def test_benchmark_run_nuts(benchmark, target, flatten):
    """Time `run_nuts` with and without position flattening.

    No warmup rounds: `run_nuts` builds a fresh `jax.jit` on every call, so
    each round pays compilation. That is the cost being measured.
    """
    log_density, init_params = target

    def run():
        return jax.block_until_ready(
            run_nuts(
                key=KEY,
                log_posterior=log_density,
                init_params=init_params,
                init_sd=1.0,
                flatten=flatten,
                **SAMPLING_KWARGS,
            )
        )

    states, _ = benchmark.pedantic(run, rounds=3, iterations=1, warmup_rounds=0)
    assert states.position["p0"].shape == (N_CHAIN, N_SAMPLE, LEAF_SIZE)


@pytest.mark.benchmark(group="make_nuts_runner (jit reused)")
@pytest.mark.parametrize("flatten", [False, True], ids=["pytree", "flat"])
def test_benchmark_make_nuts_runner(benchmark, target, flatten):
    """Time a reused sampler, i.e. steady-state sampling cost.

    One warmup round takes compilation out of the measurement, which is the
    point of the factory: only the first call compiles.
    """
    log_density, init_params = target
    sample = make_nuts_runner(log_density, flatten=flatten, **SAMPLING_KWARGS)

    def run():
        return jax.block_until_ready(sample(KEY, init_params, 1.0))

    states, _ = benchmark.pedantic(run, rounds=5, iterations=1, warmup_rounds=1)
    assert states.position["p0"].shape == (N_CHAIN, N_SAMPLE, LEAF_SIZE)


@pytest.mark.skipif(
    "JAX_TRACE_DIR" not in os.environ,
    reason="Set JAX_TRACE_DIR=<dir> to write a JAX profiler trace",
)
def test_write_jax_profile(target):
    """Write a profiler trace for the steady-state sampler (guide section 7).

    Open the result with ``tensorboard --logdir $JAX_TRACE_DIR``, or upload the
    ``.trace.json.gz`` to https://ui.perfetto.dev
    """
    log_density, init_params = target
    trace_dir = os.environ["JAX_TRACE_DIR"]
    sample = make_nuts_runner(log_density, **SAMPLING_KWARGS)

    jax.block_until_ready(sample(KEY, init_params, 1.0))  # compile first
    with jax.profiler.trace(trace_dir):
        states, _ = jax.block_until_ready(sample(KEY, init_params, 1.0))

    assert states.position["p0"].shape == (N_CHAIN, N_SAMPLE, LEAF_SIZE)
    assert os.listdir(trace_dir)
