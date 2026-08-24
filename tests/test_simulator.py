import pytest

from asic_sim.hardware import get_hardware
from asic_sim.models import get_model
from asic_sim.simulator import simulate_decode


def test_nonresident_model_returns_no_tps() -> None:
    result = simulate_decode(get_model("kimi-k3"), get_hardware("fabric-32x32"), bits_per_weight=4.0)
    assert not result.resident
    assert result.single_stream_memory_noc_roofline_tps is None
    assert result.steady_state_memory_roofline_tps is None


def test_kimi_fits_on_2tb_fabric_at_four_bit() -> None:
    result = simulate_decode(get_model("kimi-k3"), get_hardware("fabric-64x32"), bits_per_weight=4.0)
    assert result.resident
    assert result.remote_activation_bytes_per_token > 0
    assert 0.99 < result.local_data_fraction < 1.0


def test_glm_fits_on_1tb_fabric_at_four_bit() -> None:
    result = simulate_decode(get_model("glm-5.2"), get_hardware("fabric-32x32"), bits_per_weight=4.0)
    assert result.resident
    assert result.storage_bytes == pytest.approx(390.6e9)
    assert result.steady_state_memory_roofline_tps > result.single_stream_memory_noc_roofline_tps


def test_single_pool_has_no_intertile_activation_traffic() -> None:
    result = simulate_decode(get_model("glm-5.2"), get_hardware("hbm4-illustrative"), bits_per_weight=2.0)
    assert result.remote_activation_bytes_per_token == 0
