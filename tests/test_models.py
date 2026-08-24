import pytest

from asic_sim.models import get_model


def test_qwen_38_27b_q4_fits_raptor_like_tile() -> None:
    model = get_model("qwen3.8-27b")
    assert model.total_parameters == model.active_parameters == 27e9
    assert model.num_layers == 64
    assert model.hidden_size == 5120
    assert model.storage_bytes(4.0, 0.05) == pytest.approx(14.175e9)
    assert model.storage_bytes(4.0, 0.05) < 32e9


def test_kimi_k3_cannot_fit_in_1tb_at_four_bit_with_overhead() -> None:
    model = get_model("kimi-k3")
    assert model.storage_bytes(4.0, 0.05) > 1024e9
    assert model.minimum_bits_for_capacity(1024e9, 0.05) < 3.0


def test_glm_53_aliases_glm_52_base_shape() -> None:
    a = get_model("glm-5.2")
    b = get_model("glm-5.3")
    assert a.total_parameters == b.total_parameters == 744e9
    assert a.active_parameters == b.active_parameters == 40e9
    assert a.num_layers == b.num_layers == 78
    assert a.num_experts == b.num_experts == 256
    assert a.experts_per_token == b.experts_per_token == 8


def test_invalid_quantization_rejected() -> None:
    with pytest.raises(ValueError):
        get_model("glm-5.2").storage_bytes(0)
