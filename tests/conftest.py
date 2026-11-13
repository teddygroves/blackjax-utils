import pytest
from teddys_blackjax_utils.shard_map_patch import (
    get_patched_functions,
    patch_for_shard_map,
)


@pytest.fixture
def shard_map_patch(monkeypatch):
    pytest.importorskip("equinox")
    pytest.importorskip("lineax")
    for module, function_name in get_patched_functions():
        monkeypatch.setattr(
            module, function_name, getattr(module, function_name)
        )
    patch_for_shard_map()
