"""Unit tests for model.yaml validation."""
import copy

from vgreens_ai import modelspec

VALID = {
    "schema_version": 1,
    "model": {"name": "strawberry_ripeness_disease", "task": "object_detection", "owner": "ai-team"},
    "template": {"preset": "ultralytics_detection", "template_version": "v1.0.0"},
    "dataset": {"name": "strawberry_disease_auto", "version": "v1"},
    "training": {"base_model": "yolo11n.pt", "epochs": 50, "gpu_count": 1},
    "validation": {"thresholds": {"mAP50": 0.5}},
    "repository": {"namespace": "achrefmessai", "name": "vgreens-model-x"},
}


def test_valid_spec():
    assert modelspec.validate(VALID) == []


def test_bad_schema_version():
    s = copy.deepcopy(VALID); s["schema_version"] = 2
    assert any("schema_version" in e for e in modelspec.validate(s))


def test_missing_model_name():
    s = copy.deepcopy(VALID); s["model"].pop("name")
    assert any("model.name" in e for e in modelspec.validate(s))


def test_unknown_preset():
    s = copy.deepcopy(VALID); s["template"]["preset"] = "magic"
    assert any("preset" in e for e in modelspec.validate(s))


def test_task_preset_mismatch():
    s = copy.deepcopy(VALID); s["model"]["task"] = "classification"
    assert any("implements task" in e for e in modelspec.validate(s))


def test_bad_gpu_count():
    s = copy.deepcopy(VALID); s["training"]["gpu_count"] = 3
    assert any("gpu_count" in e for e in modelspec.validate(s))


def test_missing_dataset_version():
    s = copy.deepcopy(VALID); s["dataset"].pop("version")
    assert any("dataset.version" in e for e in modelspec.validate(s))


def test_threshold_state():
    assert modelspec.threshold_state(VALID) == "configured"
    s = copy.deepcopy(VALID); s["validation"]["thresholds"] = {"mAP50": None}
    assert modelspec.threshold_state(s) == "thresholds_not_configured"
