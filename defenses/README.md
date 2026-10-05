# Defense experiments

These are follow-up studies on generated images or saved classifiers, separate from the three-module dataset workflow. Each entry script is here; reusable implementation stays in the importable `triggercraft` package. Run these commands from the repository root after installing the `vision` extra. Settings are under `config/defenses/`.

| Script | Method | Output |
| --- | --- | --- |
| `strip.py` | STRIP-style blended-image entropy | Entropy and calibrated threshold |
| `neural_cleanse.py` | Mask search across target classes | Masks and anomaly scores |
| `fine_prune.py` | Clean-activation channel ranking | Pruned checkpoint and accuracy |
| `cognitive_distillation.py` | Per-image logit-preserving mask | Screening scores |
| `frequency_screen.py` | High-frequency Fourier energy | Screening scores |
| `nad.py` | Attention distillation | Repaired checkpoint and before/after results |

```bash
python defenses/strip.py --config config/defenses/strip.yaml \
  --clean /data/clean/val --query /data/query \
  --checkpoint /models/classifier.pth --output results/defenses/strip.json
python defenses/neural_cleanse.py --config config/defenses/neural_cleanse.yaml \
  --clean /data/clean/val --checkpoint /models/classifier.pth \
  --output results/defenses/neural_cleanse.json
python defenses/fine_prune.py --config config/defenses/fine_prune.yaml \
  --clean /data/clean/val --checkpoint /models/classifier.pth \
  --output results/defenses/fine_prune.json
python defenses/cognitive_distillation.py \
  --config config/defenses/cognitive_distillation.yaml \
  --clean /data/clean/val --query /data/query \
  --checkpoint /models/classifier.pth --output results/defenses/cd.json
python defenses/frequency_screen.py --config config/defenses/frequency_screen.yaml \
  --clean /data/clean/val --query /data/query \
  --output results/defenses/frequency.json
python defenses/nad.py --config config/defenses/nad.yaml \
  --data /data/experiment --checkpoint /models/classifier.pth \
  --output results/defenses/nad
```

STRIP and the per-image screens provide scores, not proof that an image contains a trigger. Neural Cleanse and FinePruner depend on the sample, feature layer, and optimization settings. NAD requires feature layers returning `B,C,H,W` tensors. Keep the clean reference set and model checkpoint fixed when comparing methods.
