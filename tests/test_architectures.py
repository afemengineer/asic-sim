import math

from asic_sim.architectures import compare_architectures, estimate_layers
from asic_sim.hardware import get_hardware
from asic_sim.models import get_model
from asic_sim.placement import decompose_model


def test_shared_experts_are_explicit_for_kimi_and_glm() -> None:
    kimi = get_model("kimi-k3")
    glm = get_model("glm-5.2")
    kd = decompose_model(kimi)
    gd = decompose_model(glm)

    assert kimi.shared_experts == 2
    assert glm.shared_experts == 1
    assert math.isclose(kd.shared_expert_parameters / 1e9, 12.155092992, rel_tol=1e-9)
    assert math.isclose(gd.shared_expert_parameters / 1e9, 2.8311552, rel_tol=1e-9)
    assert kd.other_always_on_parameters > 0
    assert gd.other_always_on_parameters > 0


def test_layer_estimates_preserve_published_counts() -> None:
    for key in ("kimi-k3", "glm-5.2"):
        model = get_model(key)
        layers = estimate_layers(model)
        assert len(layers) == model.num_layers
        assert math.isclose(sum(x.total_parameters for x in layers), model.total_parameters, rel_tol=1e-10)
        assert math.isclose(sum(x.active_parameters for x in layers), model.active_parameters, rel_tol=1e-10)
        assert sum(x.shared_parameters for x in layers) > 0


def test_kimi_uniform_q4_compares_three_resident_mappings() -> None:
    reports = compare_architectures(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        bits_per_weight=4.0,
        shared_expert_bits=4.0,
        tokens=8,
    )
    assert [r.mode for r in reports] == ["layer-pipeline", "cluster-4", "expert-mesh"]
    assert all(r.resident for r in reports)
    pipeline, cluster, mesh = reports
    assert pipeline.network_payload_bytes_per_token < cluster.network_payload_bytes_per_token
    assert cluster.hop_bytes_per_token < mesh.hop_bytes_per_token
    assert mesh.ideal_memory_floor_s < pipeline.ideal_memory_floor_s


def test_kimi_bf16_shared_experts_break_64_tile_layer_pipeline_fit() -> None:
    reports = compare_architectures(
        get_model("kimi-k3"),
        get_hardware("fabric-64x32"),
        bits_per_weight=4.0,
        shared_expert_bits=16.0,
        tokens=8,
    )
    by_mode = {r.mode: r for r in reports}
    assert not by_mode["layer-pipeline"].resident
    assert by_mode["cluster-4"].resident
    assert by_mode["expert-mesh"].resident
    assert by_mode["layer-pipeline"].max_tile_storage_bytes > 32e9
    assert by_mode["cluster-4"].shared_active_bytes_per_token > 20e9
