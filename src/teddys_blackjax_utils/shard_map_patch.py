"""Patch for equinox and lineax so that JAX shard mapping works.

At time of writing, there is a bug in equinox and lineax that prevents parallel
MCMC sampling with JAX's shard_map (now the only JAX-native option) from
working.

This module works around this bug by overwriting equinox's and lineax's
strip_weak_dtype functions to a new one (_drop_sharding) that doesn't consider
sharding.
"""

from importlib import import_module
from types import ModuleType

import jax
import jax.tree_util as jtu
from jaxtyping import PyTree

PATCHED_VERSIONS = {"equinox": "0.13.8", "lineax": "0.1.1"}
PATCHED_FUNCTIONS = (
    ("equinox._ad", "_strip_weak_dtype"),
    ("lineax._misc", "strip_weak_dtype"),
    ("lineax._operator", "strip_weak_dtype"),
    ("lineax._solve", "strip_weak_dtype"),
    ("lineax._solver.misc", "strip_weak_dtype"),
)


def _drop_sharding(tree: PyTree) -> PyTree:
    """Overwrite ShapeDtypeStruct leaves with non-sharded ones."""
    return jtu.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
        if type(x) is jax.ShapeDtypeStruct
        else x,
        tree,
    )


def get_patched_functions() -> list[tuple[ModuleType, str]]:
    """Get each module that holds a function to patch, with the function's name."""
    return [
        (import_module(module_name), function_name)
        for module_name, function_name in PATCHED_FUNCTIONS
    ]


def patch_for_shard_map() -> None:
    """Rebind strip_weak_dtype to functions that ignore sharding.

    Call once before tracing, and remember to pass check_vma=False to shard_map
    if you call it yourself.

    Raises a RuntimeError if the installed equinox or lineax version doesn't
    match the ones in PATCHED_VERSIONS (perils of patching private functions)
    """
    for module_name, version in PATCHED_VERSIONS.items():
        installed = import_module(module_name).__version__
        if installed != version:
            raise RuntimeError(
                f"teddys-blackjax-utils' shard_map patch targets {module_name} "
                f"{version}, but {installed} is installed.",
            )
    for module, function_name in get_patched_functions():
        setattr(module, function_name, _drop_sharding)
