"""Minimal helpers used by experiment scripts."""

import random
import argparse
import sys
from pathlib import Path
import numpy as np
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_experiment_args(parser, section: str):
    """Read shared and script-specific YAML settings, then apply CLI overrides."""
    parser.add_argument("--config", type=Path, help="YAML experiment settings")
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    initial, _ = config_parser.parse_known_args()
    if initial.config:
        if initial.config.suffix.lower() not in {".yaml", ".yml"}:
            parser.error("--config must be a .yaml or .yml file")
        with initial.config.open(encoding="utf-8") as config_file:
            settings = yaml.safe_load(config_file) or {}
        if not isinstance(settings, dict):
            parser.error("config must contain a YAML mapping")
        section_keys = {"common", "train", "evaluate", "generate", "defend", "neural_cleanse", "fine_prune", "cognitive_distillation", "frequency_screen", "nad", "suggest", "questions", "answer", "select"}
        if section_keys & settings.keys():
            unknown_sections = set(settings) - section_keys
            if unknown_sections:
                parser.error(f"unknown config sections: {', '.join(sorted(map(str, unknown_sections)))}")
            common = settings.get("common", {})
            specific = settings.get(section, {})
            if not isinstance(common, dict) or not isinstance(specific, dict):
                parser.error("common and script sections must be YAML mappings")
            settings = {**common, **specific}
        actions = {action.dest: action for action in parser._actions}
        unknown = set(settings) - set(actions)
        if unknown:
            parser.error(f"unknown config keys: {', '.join(sorted(map(str, unknown)))}")
        for key, value in settings.items():
            action = actions[key]
            if key in {"help", "config"}:
                parser.error(f"{key} cannot be set in config")
            if action.required and value is None:
                parser.error(f"{key} cannot be null in config")
            if action.type is not None and value is not None:
                try:
                    value = action.type(value)
                except (TypeError, ValueError) as error:
                    parser.error(f"invalid config value for {key}: {error}")
            if isinstance(action.default, bool) and not isinstance(value, bool):
                parser.error(f"{key} must be a YAML boolean")
            if action.choices is not None and value not in action.choices:
                parser.error(f"invalid config choice for {key}: {value!r}")
            if isinstance(action, argparse._AppendAction):
                if not isinstance(value, list):
                    parser.error(f"{key} must be a YAML list")
                if any(token == option or token.startswith(option + "=")
                       for token in sys.argv[1:] for option in action.option_strings):
                    continue
            action.required = False
            parser.set_defaults(**{key: value})
    args = parser.parse_args()
    for name in ("data", "checkpoint"):
        if name in {action.dest for action in parser._actions} and getattr(args, name) is None:
            parser.error(f"--{name} is required, either on the command line or in config")
    return args
