# Frozen CLIP fake-news fusion analysis

Code and derived results for the manuscript:

> **Where do fusion gains come from in low-resource multimodal fake news detection? Operating points, archive artifacts and frozen CLIP baselines**

The repository supports the paper's post-hoc analysis of saved test predictions. It is designed to make the reported metrics and data-quality summaries checkable without redistributing third-party news articles or images.

## Repository layout

```text
analysis/                 CPU-only metric re-analysis scripts and saved outputs
code/datasets/             Dataset and collator implementations
code/models/               CLIP, co-attention, prompt and knowledge modules
code/scripts/              Data preparation, feature caching, training and baselines
code/configs/              Small smoke-test configurations
results/                   Saved predictions, run logs and data-quality summaries
```

The `results/` directory contains hard predictions (`0 = real`, `1 = fake`), test labels and duplicate flags, CSV/JSON run summaries, and the outputs used in the manuscript. The repository does not contain the original articles, images, social-media metadata, CLIP feature caches, original split ID lists, or few-shot sample lists.

## Recompute the reported analysis

Create a Python 3 environment and install the lightweight analysis dependency:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
```

Run the three analyses from the repository root:

```bash
python analysis/analyze.py results out

# macOS/Linux
P2R=results python analysis/analyze2.py out
P2R=results python analysis/analyze3.py out

# Windows PowerShell
# $env:P2R = "results"
# python analysis/analyze2.py out
# python analysis/analyze3.py out
```

The scripts write `reanalysis.json`, `reanalysis_bacc.json`, and `per_run_operating_points.json` to `out/`. The saved files with the same names under `analysis/` are the reference outputs used for the manuscript.

## Training and data preparation

The model and data-preparation scripts require a separate environment with PyTorch, Transformers, scikit-learn, pandas, Pillow, requests, BeautifulSoup, tqdm and PyYAML. Install the documented package set with:

```bash
python -m pip install -r requirements-training.txt
```

The training scripts expect local inputs that are intentionally excluded from this repository:

- FakeNewsNet CSV files under `data/raw/`;
- re-collected article text and image files under `data/processed/` and `data/assets/`;
- split files under `data/splits_real/`;
- frozen CLIP feature caches under `cache/clip/`;
- optional Wikidata knowledge files under `knowledge/`.

Run the smoke-test configurations only after those local inputs and caches have been prepared. The full matrix scripts can require a GPU and substantial storage.

## Reproducibility scope

The supplied predictions and labels allow the metrics, operating-point analysis, duplicate-stratified analysis, paired bootstrap intervals and Holm correction reported in the paper to be recomputed. They do not reproduce the original training runs or split construction because the original split IDs, few-shot sample lists and feature caches were not exported. The third-party articles and images are not redistributed; users must retrieve them under the terms of their original sources and the FakeNewsNet release.

## Data-sharing description

This repository contains **derived data and code**, including saved test predictions, labels/duplicate flags, run logs, quality summaries and analysis scripts. It does not contain original news text or images. The public repository URL should be added to the manuscript's data/code availability statements only after the repository is published and the authors have checked the selected license.

## Citation

If you use this repository, cite the manuscript and the repository version. A `CITATION.cff` file is included for GitHub's citation tools.

## License

The contents of this repository are licensed under the MIT License; see [`LICENSE`](LICENSE). This license does not grant rights to third-party articles, images, social-media content, or other material governed by separate terms. The supplied derived results support research reproducibility and do not include the original articles or images.
