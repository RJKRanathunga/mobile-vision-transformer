from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str) -> dict:
    with (ROOT / "configs" / name).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_base_config_matches_model_geometry() -> None:
    config = load_config("mvt_base.yaml")
    assert config["reference"]["commit"] == (
        "880e74b1d1f720cb51d78b68655a4ffef94ed954"
    )
    assert config["data"]["search"]["size"] == 256
    assert config["data"]["template"]["size"] == 128
    assert config["model"]["head"]["feature_size"] == 16


def test_training_and_inference_configs_match_upstream_experiment() -> None:
    train = load_config("train.yaml")
    inference = load_config("inference.yaml")
    assert train["training"]["optimizer"] == "adamw"
    assert train["training"]["scheduler"]["type"] == "cosine_anneal"
    assert train["training"]["losses"] == {
        "giou_weight": 2.0,
        "l1_weight": 5.0,
        "upstream_heatmap_bug_compatible": False,
    }
    assert inference["search"]["size"] // inference["backbone_stride"] == 16
    assert inference["apply_hann_window"] is True
