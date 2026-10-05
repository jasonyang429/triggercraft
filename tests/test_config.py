import argparse
import sys

import pytest

from triggercraft.utils import parse_experiment_args


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--data", required=True)
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--pretrained", action="store_true")
    return result


def test_yaml_defaults_and_cli_override(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("common:\n  data: /tmp/images\ntrain:\n  epochs: 2\n  pretrained: true\n")
    monkeypatch.setattr(sys, "argv", ["test", "--config", str(config), "--epochs", "3"])
    args = parse_experiment_args(parser(), "train")
    assert (args.data, args.epochs, args.pretrained) == ("/tmp/images", 3, True)


def test_unknown_yaml_key_fails(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("data: /tmp/images\nepochss: 2\n")
    monkeypatch.setattr(sys, "argv", ["test", "--config", str(config)])
    with pytest.raises(SystemExit):
        parse_experiment_args(parser(), "train")


def test_other_section_is_ignored(tmp_path, monkeypatch):
    config = tmp_path / "run.yaml"
    config.write_text("common:\n  data: /tmp/images\ntrain:\n  epochs: 2\nevaluate:\n  checkpoint: model.pth\n")
    monkeypatch.setattr(sys, "argv", ["test", "--config", str(config)])
    assert parse_experiment_args(parser(), "train").epochs == 2
