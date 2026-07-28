# boa_ft: BOA on LeMat-Rho

Trains BOA (Basis Overlap Architecture, github.com/sciai-lab/boa) from scratch
on LeMat-Rho r2SCAN charge densities (15x15x15 grids, periodic solids).

BOA learns per atom-pair Gaussian-orbital coefficients and is supervised
directly on grid density values (no grid-to-basis projection step). We add:

- `preprocess.py`: LeMat-Rho parquet -> sharded LMDB of scdp `AtomicData`.
- `transforms.py`: `PBCRadiusEdgeIndex`, a periodic (minimum-image) neighbor
  graph that replaces BOA's open-boundary `radius_graph` and removes the
  `torch_cluster` dependency from the training path.
- `configs/`: Hydra overlays wiring the datamodule to our LMDB and transform,
  and turning the periodic decoder on (`model.pbc=true`, `model.orb_cutoff=3.0`).

## Setup

BOA and its vendored `scdp` and `sciai-dft` (`mldft`) packages are not on PyPI.
Clone BOA as a sibling of this repo and editable-install the three packages into
the same environment that has this repo's dependencies:

```bash
git clone https://github.com/sciai-lab/boa ../boa
uv pip install -e ../boa/scdp -e ../boa/sciai-dft -e ../boa
```

`preprocess.py` also imports the shared parquet decoder from `charge3net_ft`,
which requires the charge3net repo cloned as a sibling (`../charge3net`), exactly
as the deepdft and charge3net arms already need.

## Preprocess

```bash
python -m boa_ft.preprocess \
    --parquet-dir /path/to/lemat_rho_chunks \
    --out-dir $BOA_DATA/lematrho \
    --num-shards 16
```

This writes `$BOA_DATA/lematrho/data/*.lmdb`, `$BOA_DATA/lematrho/datasplits.json`
(charge3net split convention: seed 42, 5% val, 5% test), and
`$BOA_DATA/lematrho/atomic_numbers.json`. The element list is the union over the
*train* split only, because BOA builds one orbital per training element and
aligns the basis positionally against the train split. This means the train
split must contain every element you want a basis for (a 90% split of a large
dataset does; a tiny subset may not). Use `--limit N` for a quick smoke subset.

## Train

BOA reads three environment variables for its paths (`PROJECT_ROOT` = the boa
clone, `BOA_DATA` = the dataset root that holds `lematrho/`, `BOA_MODELS` = the
run/checkpoint output root). BOA's Hydra root lives in the boa clone, so add our
config directory to the search path and select the experiment:

```bash
export PROJECT_ROOT=/path/to/boa
export BOA_DATA=/path/to/data_root
export BOA_MODELS=/path/to/model_root

ATOMIC_NUMBERS=$(python -c "import json; print(json.load(open('$BOA_DATA/lematrho/atomic_numbers.json')))")

python "$PROJECT_ROOT/boa/train.py" \
    hydra.searchpath="[file://$(pwd)/boa_ft/configs]" \
    experiment=lematrho \
    data.basis_info.atomic_numbers="$ATOMIC_NUMBERS"
```

The `experiment=lematrho` overlay sets `model.pbc=true`, `model.orb_cutoff=3.0`,
`model.linear_basis=true` (the periodic-safe density decoder), and
`initial_guess_pre_training_steps=0`. BOA's separate initial-guess pre-training
opens a second datamodule on the same LMDB, which py-lmdb forbids in one
process, so we disable it; the initial-guess module still trains jointly in the
main loop.

### Smoke run (CPU)

`+trainer.devices=1` uses a leading `+` because the trainer config has no
`devices` key by default. A short `hydra.run.dir` avoids a run-directory name
built from the (long) override string.

```bash
python "$PROJECT_ROOT/boa/train.py" \
    hydra.searchpath="[file://$(pwd)/boa_ft/configs]" \
    hydra.run.dir=/tmp/boa_smoke \
    experiment=lematrho \
    data.basis_info.atomic_numbers="$ATOMIC_NUMBERS" \
    trainer.accelerator=cpu +trainer.devices=1 \
    trainer.max_steps=3 trainer.max_epochs=1 trainer.num_sanity_val_steps=0 \
    data.datamodule.num_workers.train=0 data.datamodule.num_workers.val=0 \
    data.datamodule.batch_size.train=2 data.datamodule.n_probe.train=512
```

## Adastra

Use `submit_boa_adastra.sh` at the repo root (MI250, account `c1816212`, single
GCD first). It sets the proxy, activates `venv311`, exports the three BOA path
variables, reads `atomic_numbers.json`, and launches the Hydra command above.
