# blackjax-utils

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Functions for running MCMC with [Blackjax](https://blackjax-devs.github.io/blackjax/).

The main aim is to provide a simple interface for running the NUTS sampler with blackJAX, much like Stan, PyMC or numpyro.

This approach is nice, in my opinion, as you don't have to learn a specialised probabilistic programming language. You 'just' have to write a JAX-compatible log density function.

## Installation

blackjax-utils isn't on PyPI, so install it from GitHub:

```bash
pip install git+https://github.com/teddygroves/blackjax-utils.git
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add git+https://github.com/teddygroves/blackjax-utils.git
```

## Quick start

```python
import jax
import jax.numpy as jnp
from blackjax_utils import run_nuts

# Define a log-density to sample from (e.g. a standard normal)
def log_density(params):
    return -0.5 * jnp.sum(params["x"] ** 2)

# Run 4 chains in parallel
key = jax.random.PRNGKey(42)
states, info = run_nuts(
    key=key,
    log_posterior=log_density,
    init_params={"x": jnp.array([0.0])},
    init_sd=1.0,
    n_chain=4,
    n_warmup=500,
    n_sample=1000,
)

# states.position is a PyTree of samples with shape (n_chain, n_sample, ...)
samples = states.position["x"]  # shape: (4, 1000, 1)
```

### Multi-device parallelism

```bash
# Run with 4 CPU devices
JAX_NUM_CPU_DEVICES=4 python my_script.py
```

```python
# pmap across all devices
states, info = run_nuts(
    ...,
    n_chain=4,
    chain_map=jax.pmap,
)

# Or use shard_map with a sub-mesh
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec

mesh = Mesh(jax.devices()[:2], axis_names=("chains",))

def chain_map(func, in_axes):
    def wrapped(key, params):
        key = key[0]
        params = jax.tree.map(lambda x: x[0], params)
        states, info = func(key, params)
        states = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), states)
        info = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), info)
        return states, info
    return shard_map(
        wrapped,
        mesh=mesh,
        in_specs=(PartitionSpec("chains"), PartitionSpec("chains")),
        out_specs=PartitionSpec("chains"),
    )

states, info = run_nuts(..., n_chain=2, chain_map=chain_map)
```

### Passing warmup and sampling kwargs

By default, `**kwargs` are forwarded to both warmup and sampling. Use
`warmup_options` and `sampling_options` to set or override values for a single
stage:

```python
run_nuts(
    ...,
    max_num_doublings=10,                          # goes to both warmup and sampling
    warmup_options=dict(initial_step_size=0.1),    # warmup only
    sampling_options=dict(max_num_doublings=5),    # sampling only
)
```

`warmup_options` is the only way to reach arguments that
[`blackjax.window_adaptation`](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/window_adaptation/index.html)
accepts but `blackjax.nuts` rejects, since a plain `**kwargs` value is sent to
both stages and the NUTS kernel would raise `TypeError`. These are:

- `initial_step_size`
- `target_acceptance_rate`
- `is_mass_matrix_diagonal`
- `initial_inverse_mass_matrix`
- `imm_shrinkage_to_previous`
- `adaptation_info_fn`

Kernel parameters given only to warmup stay there: `warmup_options=dict(max_num_doublings=1)`
leaves sampling on the blackjax default rather than carrying the value over.

### Using another sampler

`run_nuts` is an alias for `run_sampler`, which builds its warmup and its
sampling kernel through a `sampler` argument: a `Sampler` pair of factories,
defaulting to `NUTS`. Replace it and the same multi-chain, jittering,
flattening machinery runs a different sampler:

```python
Sampler(make_warmup, make_kernel)

make_warmup(density, **warmup_kwargs)  # -> .run(key, position, num_steps)
make_kernel(density, **params)         # -> step(key, state) -> (state, info)
```

`make_warmup`'s return must yield `((state, tuned_params), info)`, and
`make_kernel` receives the tuned parameters merged with the static ones. The
target density is called with whatever keyword arguments the sampler passes it,
so a sampler that threads extra state through the density — as
[grapevine](https://github.com/dtu-qmcm/grapevine) does with its guess — works
without blackjax-utils knowing anything about it:

```python
from blackjax_utils import run_sampler
from grapevine import grapenuts

states, info = run_sampler(
    key=key,
    log_posterior=log_density,     # (position, guess) -> (log_density, solution)
    init_params=init_params,
    n_chain=4,
    sampler=grapenuts(default_guess),
)
```

Bind anything algorithm-specific into those two factories yourself, with
`functools.partial`, rather than passing it through `warmup_kwargs`: warmup
echoes its extra arguments back in `tuned_params`, which then reach the kernel
as per-step parameters.

## Performance

blackjax-utils follows the
[blackjax speed-up guide](https://blackjax-devs.github.io/blackjax/examples/speed_up_guide.html).
What it does for you:

- **One `jax.jit` per chain**, wrapping warmup and sampling together, placed
  inside `chain_map` so it composes correctly with `vmap`, `pmap` and
  `shard_map`.
- **`jax.lax.scan`** for the sampling loop, with the NUTS kernel built once
  outside it.
- **A flat position.** `run_nuts` ravels your position PyTree into a single 1-D
  array before it reaches blackjax and restores your structure in the returned
  samples, so a dict of parameters does not become one buffer per leaf in the
  scan carry.
- **Chain parallelism** via `chain_map` (see above).

Measured on one CPU device with an 8-leaf dict position (40 dimensions),
4 chains, 500 warmup and 500 sampling steps (median of the runs reported by
`tests/test_benchmarks.py`, see [Benchmarks](#benchmarks)):

| | position | median per call |
| --- | --- | --- |
| `run_nuts(flatten=False)` | PyTree | 1431ms |
| `run_nuts()` | flat | 696ms |
| `make_nuts_runner()` (`flatten=False`) | PyTree | 102ms |
| `make_nuts_runner()` | flat | **64ms** |

`run_nuts` recompiles on every call, so its figures include compilation;
`make_nuts_runner` compiles once, so its figures are steady-state sampling
cost. Flattening is worth ~1.6x on its own and reusing the compiled sampler a
further ~10x.

### Sampling repeatedly

`run_nuts` builds a fresh `jax.jit` on each call, so its compilation cache is
always empty and calling it in a loop recompiles every time. To sample the same
model more than once, build the sampler once with `make_nuts_runner`:

```python
from blackjax_utils import make_nuts_runner

sample = make_nuts_runner(log_density, n_chain=4, n_warmup=500, n_sample=1000)

for key in jax.random.split(jax.random.PRNGKey(0), 10):
    states, info = sample(key, init_params, 1.0)   # compiles once, not ten times
```

It takes the same arguments as `run_nuts` apart from `key`, `init_params` and
`init_sd`, which move to the returned callable. Recompilation still happens if
the shape or dtype of `init_params` changes.

### Flattening caveats

Pass `flatten=False` to keep your PyTree as the sampler position. Two reasons
you might need to:

- Every leaf must have a floating-point dtype. Integer leaves raise `TypeError`
  rather than being silently rounded on the way back out.
- Flattening is algebraically but not always bitwise identical to sampling the
  PyTree directly, because XLA reassociates float operations differently on a
  single array. Differences start at float rounding level (order 1e-7 in
  float32) and could compound on an ill-conditioned target, so a fixed seed is
  not guaranteed to reproduce `flatten=False` numbers exactly.

Flattening is a no-op when the position is already a single array, and gains
little for two or three scalar leaves; it matters most for many leaves or large
arrays.

### Other tips from the guide

- On GPU, prefer `float32` (the JAX default -- leave `jax_enable_x64` off) and
  run more chains rather than fewer, since a single chain rarely saturates a
  GPU.
- Under `vmap`, NUTS trajectory expansion is a `lax.while_loop`, so all chains
  run in lockstep to the longest trajectory in the batch. With several devices
  available, `chain_map=jax.pmap` avoids that coupling.
- To profile, set `JAX_TRACE_DIR` and run the trace test, then open the result
  in [Perfetto](https://ui.perfetto.dev) or TensorBoard:

  ```bash
  JAX_TRACE_DIR=/tmp/jax-trace uv run pytest tests/test_benchmarks.py::test_write_jax_profile
  tensorboard --logdir /tmp/jax-trace
  ```

### Benchmarks

`tests/test_benchmarks.py` uses
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Timings are
disabled by default, so a normal test run executes each benchmark body once as
a smoke test and reports nothing. To measure:

```bash
uv run pytest tests/test_benchmarks.py --benchmark-enable
```

Vary the problem size with environment variables, e.g.:

```bash
BENCH_N_LEAVES=16 BENCH_N_CHAIN=8 uv run pytest tests/test_benchmarks.py --benchmark-enable
```

## Development

Clone and install with dev dependencies:

```bash
git clone https://github.com/teddygroves/blackjax-utils.git
cd blackjax-utils
uv sync
```

### Running tests

```bash
uv run pytest
```

Multi-device tests require ≥ 4 CPU devices:

```bash
JAX_NUM_CPU_DEVICES=4 uv run pytest
```

### Linting

```bash
uv run ruff check .
```

## License

MIT
