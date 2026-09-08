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
