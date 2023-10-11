# FONDEF Hole point detection

## Install instructions

- Install poetry, then

```bash
poetry install
```

- Donwload datasets
  - The dataset must be inside a *datasets* folder in this directory

---

## Train

- Launch docker instance

```bash
./docker/launch.sh
```

- Train experiments (inside docker) using scripts

```bash
# For example 
./scripts/train_best_holes_resnext.sh
```

### Find best parameters

- Use *optuna* scripts inside /cfg

```bash
cd ./cfg/optuna && python [opt_name]
```

## Convert to ONNX

Use convert.py

```bash
poetry run python convert.py --help
```

## Evaluation

Use eval.py

```bash
poetry run python eval.py --help
```
