"""Tests for the model repo generator."""
import copy
import py_compile
from pathlib import Path

from vgreens_ai import modelgen

SPEC = {
    "schema_version": 1,
    "model": {"name": "sd_test", "display_name": "SD test", "task": "object_detection"},
    "template": {"preset": "ultralytics_detection", "template_version": "v1.0.0"},
    "dataset": {"name": "strawberry_disease_auto", "version": "v1"},
    "training": {"base_model": "yolo11n.pt", "epochs": 30, "image_size": 640, "batch_size": 16},
    "validation": {"thresholds": {"mAP50": 0.5}},
    "processing": {"confidence_threshold": 0.3},
    "repository": {"namespace": "achrefmessai", "name": "vgreens-model-sd-test"},
}


def test_generate_detection_repo(tmp_path):
    out = tmp_path / "repo"
    files = modelgen.generate(SPEC, str(out))
    for f in ("entrypoint.py", "train.py", "process.py", "Dockerfile",
              "configs/sd_test.yaml", ".gitlab-ci.yml", "tests/test_contract.py", "README.md"):
        assert f in files and (out / f).is_file()
    # config carries dataset-driven hyperparameters + thresholds
    cfg = (out / "configs/sd_test.yaml").read_text()
    assert "epochs: 30" in cfg and "mAP50: 0.5" in cfg
    # CI wires the model + dataset
    ci = (out / ".gitlab-ci.yml").read_text()
    assert "MODEL_NAME: sd_test" in ci and "DATASET_VERSION: v1" in ci
    # generated python compiles
    for name in ("entrypoint.py", "train.py", "process.py"):
        py_compile.compile(str(out / name), doraise=True)


def test_refuses_nonempty_dir(tmp_path):
    out = tmp_path / "repo"
    modelgen.generate(SPEC, str(out))
    try:
        modelgen.generate(SPEC, str(out))
        assert False, "should have refused a non-empty dir"
    except FileExistsError:
        pass
    # force overwrites
    assert modelgen.generate(SPEC, str(out), force=True)


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "repo"
    files = modelgen.generate(SPEC, str(out), dry_run=True)
    assert files and not out.exists()


def test_custom_python_skeleton_not_placeholder(tmp_path):
    s = copy.deepcopy(SPEC)
    s["model"]["task"] = "custom"
    s["template"]["preset"] = "custom_python"
    out = tmp_path / "custom"
    modelgen.generate(s, str(out))
    train = (out / "train.py").read_text()
    # The custom skeleton is a RUNNABLE reference (honours the /data->/out
    # contract), not a bare NotImplementedError placeholder.
    assert "RUNNABLE REFERENCE" in train
    assert "def main(" in train and "/out" in train
    # dependency declaration is first-class: a requirements.txt is generated and
    # the Dockerfile installs it.
    assert (out / "requirements.txt").exists()
    assert "pip install --no-cache-dir -r /app/requirements.txt" in (out / "Dockerfile").read_text()


def test_unimplemented_preset_raises(tmp_path):
    s = copy.deepcopy(SPEC)
    s["model"]["task"] = "classification"
    s["template"]["preset"] = "ultralytics_classification"
    try:
        modelgen.generate(s, str(tmp_path / "c"))
        assert False, "should not implement classification yet"
    except ValueError as e:
        assert "not implemented" in str(e)
