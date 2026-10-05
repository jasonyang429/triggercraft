# TriggerCraft: A Reproducible Framework for Physical Backdoor Dataset Synthesis

Small research scripts for constructing and studying physical backdoor image datasets. The workflow is deliberately visible: ask a vision-language model which objects fit an image (Module 1), generate candidate images (Module 2), then score and select candidates (Module 3). Training, evaluation, and defenses are separate follow-up experiments.

## Repository map

| Where | What it does |
| --- | --- |
| `questions.py`, `answer.py`, `suggest.py` | Module 1: query images and summarize object suggestions |
| `generate.py` | Module 2: generate text- or image-conditioned candidates |
| `select.py` | Module 3: rank candidates with ImageReward |
| `train.py`, `evaluate.py`, `audit.py` | Train and inspect classifiers and datasets |
| `defenses/` | Defense experiment scripts and their own [README](defenses/README.md) |
| `triggercraft/` | Small importable implementations shared by the scripts |
| `config/` | Copyable YAML settings for individual experiments |
| `experiments/` | A place to keep named experiment commands and settings |

Run the scripts from the repository root. Give each experiment its own YAML, input-image list, model revision, seed, and output directory. Data, model checkpoints, and generated results are not bundled with the package.

## Install with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first. On the GPU server, the system Python already has a CUDA-enabled PyTorch build. `--system-site-packages` makes it visible to these environments, but uv may resolve a newer PyTorch when installing extras. Check the installed `torch` and `torchvision` versions and use a PyTorch build matching the machine's accelerator.

```bash
uv venv --python python3 --system-site-packages .venv
source .venv/bin/activate
uv pip install -e '.[vision,generation,vlm]'
python -c 'import triggercraft; print(triggercraft.__file__)'
```

The base package needs only NumPy, Pillow, and PyYAML; `vision`, `generation`, and `vlm` are optional extras. A CPU-only reader can install just `uv pip install -e .`. Keep this environment active for Modules 1 and 2.

ImageReward uses older pinned libraries than the generation environment. Keep it in a second environment. The following uv commands mirror the pinned install order of the working small-scale run; the uv reward install itself has not yet been run end to end:

```bash
uv venv --python python3 --system-site-packages .venv-reward
source .venv-reward/bin/activate
uv pip install --no-deps 'image-reward==1.5'
uv pip install 'timm==0.6.13' 'fairscale==0.4.13' \
  'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
uv pip install 'transformers==4.38.2' \
  'huggingface-hub==0.25.2' 'diffusers==0.27.2' 'accelerate==0.27.2'
uv pip install --no-deps -e .
python -c 'import triggercraft, ImageReward; print("ready")'
source .venv/bin/activate
```

For a local ImageReward checkpoint, download both `ImageReward.pt` and `med_config.json` from [THUDM/ImageReward](https://huggingface.co/THUDM/ImageReward/tree/main), then pass their paths to `select.py`. Record their hashes with every run. The tested files had SHA-256 values `d71dc8d4912bc633c06ab8ef1ef6a81e8199d4369f14b47b29baf1d19de561f6` and `d8b5029cc3c0124313d17009d043f0701bbe497f3721fa4ceb818720fe665326`, respectively. Run `python -m pytest -q` from the main environment after adding the `test` extra if changing code.

## Module 1: Trigger Suggestion Module

Use an ImageFolder-style tree such as `train/dog/*.jpg` and `train/cat/*.jpg`. `questions.py` writes one question per image; `answer.py` applies the selected vision-language checkpoint; `suggest.py` groups normalized answers by class. Its JSON reports each class's queried-image total and each object's `count` and `frequency`. An object counts at most once per image, so `frequency = count / images`.

```bash
python questions.py --images /data/imagenet5/train \
  --question "What are five suitable objects to be added to the image?" \
  --output results/module1/questions.jsonl
python answer.py --questions results/module1/questions.jsonl \
  --model-id /models/vlm --output results/module1/answers.jsonl
python suggest.py --config config/suggestions.yaml \
  --answers results/module1/answers.jsonl --classes-file config/classes.yaml \
  --output results/module1/suggestions.json
```

`config/classes.yaml` maps the five example classes (bag, bottle, chair, dog, cat); change it for another dataset. Save the exact VLM revision and decoding settings. A four-image plumbing test used SmolVLM2-256M; paper-scale results require the specified LLaVA checkpoint and full image split. Suggestion frequency describes the model's answers, not whether the object appears in an image.

## Module 2: Trigger Generation Module

`generate.py` supports interchangeable Diffusers pipelines. `text` starts from a prompt, `image` uses a source image, and `edit` expects an InstructPix2Pix-compatible pipeline. It writes numbered PNGs and a `manifest.jsonl` recording prompts, seeds, source paths, and settings. Keep one pool per subject, object, and generation route.

```bash
python generate.py --config config/flux2-klein.yaml \
  --model-id /models/FLUX.2-klein-4B \
  --prompt "A realistic photo of a dog with a book in a park" \
  --output results/module2/dog-book-text
python generate.py --config config/flux2-klein.yaml --mode image \
  --model-id /models/FLUX.2-klein-4B --images /data/imagenet5/train/dog \
  --prompt "Add a book into the image." \
  --output results/module2/dog-book-image
```

Replace `/models/FLUX.2-klein-4B` with a downloaded model directory or Hugging Face model ID. The checked-in FLUX YAML is a 256-pixel, four-step smoke setting. Copy and edit it for a named experiment. The paper's main routes use InstructDiffusion and Realistic Vision V5.1; those exact checkpoint routes have not been verified here. A generic Diffusers edit pipeline should not be treated as equivalent to InstructDiffusion.

## Module 3: Poison Selection Module

`select.py` scores each candidate against one subject-and-object sentence with ImageReward and saves the full ranking plus the top `--keep` paths. Give each pool its own prompt and selection report; keep the candidate-count denominator and rounding rule used to choose `--keep`. Review selected images visually, since a high reward score does not certify a complete, singular object.

Switch to the reward environment for this module:

```bash
source .venv-reward/bin/activate
python select.py --images results/module2/dog-book-text \
  --prompt "A photo of a dog with a book." --keep 100 \
  --model-id /models/imagereward/ImageReward.pt \
  --med-config /models/imagereward/med_config.json --device cuda \
  --output results/module3/dog-book-ranked.json
```

The ranking JSON includes the ImageReward checkpoint/config hashes, runtime versions, every candidate score, and selected paths. In a one-candidate smoke run, the dog-with-book image scored `1.7217`; this checks the software path, not selection quality or paper-scale results.

## Train, evaluate, and defenses

`train.py` expects `train/<class>/*.jpg` and `val/<class>/*.jpg`. `config/imagenet5-paper.yaml` records example ResNet-18 SGD/cosine settings. `evaluate.py` reports clean accuracy, confusion matrix, and target response on named comparison sets. Record each selected image's source, assigned label, and poisoning-rate denominator when building a training variant.

Switch back to the main environment for training and evaluation:

```bash
source .venv/bin/activate
python audit.py /data/experiment/train --verify-images
python train.py --config config/imagenet5-paper.yaml \
  --data /data/experiment --device cuda --output results/classifier
python evaluate.py --config config/imagenet5-paper.yaml \
  --data /data/experiment/val --checkpoint results/classifier/best_model.pth \
  --comparison triggered=/data/triggered --target-class dog --device cuda \
  --output results/evaluation.json
```

The [defense scripts](defenses/README.md) run from the repository root; their YAML files are under `config/defenses/`. From Python, the shallow package exposes `ImageFiles`, `ImageGenerator`, `summarize_suggestions`, `rank_candidates`, and `load_classifier`, for example `from triggercraft import summarize_suggestions`.

## Citation

```bibtex
@inproceedings{
  yang2026triggercraft,
  title={TriggerCraft: A Reproducible Framework for Physical Backdoor Dataset Synthesis},
  author={Sze Jue Yang and Chinh Duc La and Quang H Nguyen and Eugene Bagdasarian and Kok-Seng Wong and Chee Seng Chan and Khoa D Doan},
  booktitle={Eighteenth Asian Conference on Computer Vision},
  year={2026},
  url={https://openreview.net/forum?id=OHF4tXf7XZ}
}
```
