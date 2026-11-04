"""CPU-only NUMA selection tests; CUDA and libnuma calls are mocked."""

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.workers import worker as worker_module


@pytest.mark.parametrize(
    "current,physical,mask,cpu_less,unavailable,expected",
    [
        pytest.param(4, list(range(8)), None, False, False, 1, id="unmasked-current-gpu4"),
        pytest.param(0, [4], "4", False, False, 1, id="single-visible-gpu4"),
        pytest.param(1, [6, 1], "6,1", False, False, 0, id="reordered-mask"),
        pytest.param(0, [4], "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", False, False, 1, id="uuid-mask"),
        pytest.param(4, list(range(8)), None, True, False, 1, id="cpu-less-pci-node"),
        pytest.param(4, list(range(8)), None, False, True, 1, id="missing-pci-properties"),
    ],
)
def test_numa_affinity_uses_current_cuda_device(monkeypatch, current, physical, mask, cpu_less, unavailable, expected):
    """CUDA owns visibility remapping; PCI locality selects a CPU-bearing node."""
    monkeypatch.setattr(worker_module, "_SET_AFFINITY", False)
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    selected = []

    def properties(index):
        selected.append(index)
        if unavailable:
            raise RuntimeError("PCI attributes unavailable")
        return SimpleNamespace(pci_domain_id=0, pci_bus_id=physical[index] + 1, pci_device_id=0)

    def opened(path, *args, **kwargs):
        node_lists = {
            "/sys/devices/system/node/node0/cpulist": "0-47,96-143",
            "/sys/devices/system/node/node1/cpulist": "48-95,144-191",
            "/sys/devices/system/node/node2/cpulist": "",
        }
        if path in node_lists:
            return io.StringIO(node_lists[path])
        if path.endswith("/numa_node"):
            physical_gpu = int(path.split("/")[5].split(":")[1], 16) - 1
            return io.StringIO("2" if cpu_less else str(int(physical_gpu >= 4)))
        if path.endswith("/local_cpulist"):
            return io.StringIO("48-95,144-191")
        raise FileNotFoundError(path)

    def listed(path):
        if path == "/sys/devices/system/node":
            return ["node0", "node1", "node2"]
        if path == "/sys/devices/system/cpu/cpu48":
            return ["node1"]
        raise FileNotFoundError(path)

    libnuma = MagicMock()
    bitmask = object()
    libnuma.numa_parse_nodestring.return_value = bitmask
    monkeypatch.setattr(worker_module, "CDLL", lambda _: libnuma)
    monkeypatch.setattr(worker_module.torch.cuda, "current_device", lambda: current)
    monkeypatch.setattr(worker_module.torch.cuda, "get_device_properties", properties)
    # Avoid actor construction: the real method only needs this fallback ordinal.
    worker = SimpleNamespace(_local_rank=4)
    with (
        patch("ctypes.util.find_library", return_value="libnuma.so"),
        patch("builtins.open", side_effect=opened),
        patch.object(worker_module.os, "listdir", side_effect=listed),
    ):
        worker_module.DistributedTorchRayActor._set_numa_affinity(worker, 4)

    assert selected == [current]
    libnuma.numa_parse_nodestring.assert_called_once_with(str(expected).encode("ascii"))
    libnuma.numa_run_on_node_mask.assert_called_once_with(bitmask)
    libnuma.numa_set_membind.assert_called_once_with(bitmask)
    assert worker_module._SET_AFFINITY is True
