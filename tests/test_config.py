"""
Smoke test — verifies that the config loads correctly and has expected keys.
Run with: pytest tests/
"""
from src.utils.config import load_config


def test_config_loads():
    cfg = load_config()
    assert isinstance(cfg, dict), "Config should be a dictionary"


def test_config_has_required_sections():
    cfg = load_config()
    required = ["data", "detector", "mediapipe", "classifier", "training", "inference"]
    for section in required:
        assert section in cfg, f"Missing config section: '{section}'"


def test_num_classes():
    cfg = load_config()
    assert cfg["data"]["num_classes"] == 26, "Expected the 26 available ASL letter classes"
    assert len(cfg["data"]["class_names"]) == cfg["data"]["num_classes"]
