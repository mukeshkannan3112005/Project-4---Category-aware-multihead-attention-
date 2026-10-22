# Category-Aware Multi-Task Learning Framework for ABSA — Phase 1

**Phase 1 scope:** an accurate, modular reproduction of the base paper

> *"Aspect-Level Sentiment Classification of Consumer Reviews Utilizing BERT and Category-Aware Multi-Head Attention"* (Zhao et al., IEEE Transactions on Consumer Electronics, 2025)

The Multi-Task Learning novelty (aspect extraction + opinion extraction heads) is **not implemented yet**. The architecture below is deliberately modular so that phase adds new heads on top of `SharedFeatureExtractor` without touching the code you validate in this phase.

---

## 1. Project structure

```
project/
├── data/                        <- put your CSV files here
│   ├── Laptop_Train_v2.csv
│   ├── Laptops_Test_Data_PhaseA.csv
│   ├── Laptops_Test_Data_PhaseB.csv
│   ├── laptop-trial.csv
│   ├── Restaurants_Train_v2.csv
│   ├── Restaurants_Test_Data_PhaseA.csv
│   ├── Restaurants_Test_Data_PhaseB.csv
│   └── restaurants-trial.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     <- CSV/XLSX loader, column auto-detection, label mapping
│   ├── dataset.py           <- PyTorch Dataset: BERT pair-encoding + aspect mask
│   ├── attention.py         <- Category-Aware Multi-Head Attention (core contribution)
│   ├── model.py             <- BERTEncoder, AspectProjection, SharedFeatureExtractor,
│   │                          SentimentClassificationHead, ABSCModel
│   ├── train.py             <- training loop, CLI, ablation flags, checkpointing
│   ├── evaluate.py          <- metrics + standalone checkpoint evaluation
│   ├── inspect_data.py      <- Phase 1 "look at my CSV" script
│   └── utils.py             <- seeding, device selection, checkpoint I/O
│
├── checkpoints/              <- best_model.pt gets saved here
├── results/                  <- training_history.json / test_metrics.json per run
├── configs/                  <- (reserved for saved run configs)
├── requirements.txt
└── README.md
```

### Architecture (matches Fig. 2 of the paper)

```
Consumer Review + Aspect Term
        v
BERT Tokenizer (pair encoding: [CLS] review [SEP] aspect [SEP])
        v
BERTEncoder (pretrained bert-base-uncased)
        v
Contextual Token Embeddings  ──────────────┐
        v                                  │
AspectProjection (mean-pool aspect tokens, │
   linear-project into context space)      │
        v                                  │
CategoryAwareMultiHeadAttention  <──────────┘
        v
Per-sentiment-category Fused Representation
        v
SentimentClassificationHead
        v
Sentiment Prediction (+ confidence, + attention weights)
```

`SharedFeatureExtractor` exposes BOTH the sentence-level fused representation (`fusion`, used above) and the raw token-level BERT hidden states (`token_hidden`), unchanged, precisely so the future `AspectExtractionHead` / `OpinionExtractionHead` can attach to `token_hidden` without any changes to this phase's code.

---

## 2. Design decision you should know about (category-aware mechanism)

The paper's category weights `w^{Sj}_category` are indexed by **sentiment category** (positive / negative / neutral / ...), not by aspect category (e.g. "food" / "service"). Your Laptop CSVs don't have an aspect-category column at all, and even the Restaurant CSVs' `category` column is a different concept from the paper's `w_category`. So:

- `num_categories` in `CategoryAwareMultiHeadAttention` = number of **sentiment classes** detected in your training data (this matches the paper's Eq. 3 literally).
- If a dataset *does* have an aspect-category column, it's captured (`category2id`) and stored in the dataset/checkpoint for future use, but Phase 1 does not condition attention on it. This is intentionally left available for your Phase-2 novelty if you want to extend the category-aware mechanism to use it.

This is documented again inline in `src/attention.py`.

---

## 3. Windows setup (PowerShell)

Open PowerShell in your project folder (adjust the path to wherever you place this project):

```powershell
cd "D:\sem6\mini project 4\project_folder"
```

Create and activate a virtual environment:

```powershell
python -m venv venv // py -3.11 -m venv venv
//Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
//.\venv\Scripts\Activate.ps1  
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you have a CUDA GPU and want GPU-accelerated PyTorch specifically (the `requirements.txt` installs a CPU-or-CUDA build depending on your pip/platform defaults), follow the official selector at https://pytorch.org/get-started/locally/ and run the printed command instead of the generic `torch` line — then re-run `pip install -r requirements.txt` to grab the rest.

Place your dataset CSVs inside the `data\` folder (already shown in the structure above).

---

## 4. Step-by-step run order

### Phase 1 — Inspect your dataset (always do this first on a new file)

```powershell
python src/inspect_data.py --data data/Restaurants_Train_v2.csv
```

Expected output: the raw column names in your file, the auto-detected mapping (`text`, `aspect`, `label`, `category`), and the raw label value counts. **Read this output carefully.** If a required column can't be found, the script tells you exactly which candidate names it searched for — fix the CSV header or extend the candidate list in `src/preprocessing.py` rather than guessing.

### Phase 2 — Full training run (BERT-CA-WMA, the complete base-paper model)

**Important — dataset splitting:** only `Restaurants_Train_v2.csv` / `Laptop_Train_v2.csv` are labeled (they have `Aspect Term` + `polarity` columns). The `*_Test_Data_PhaseA.csv` / `*_Test_Data_PhaseB.csv` files only contain `id` + `Sentence` — **no labels** — so they are never used during training or evaluation. By default (no `--test_data` flag), `train.py` performs an internal **stratified 80% train / 10% validation / 10% test split** of `--train_data` itself, stratified on `polarity`. The 10% test portion is held out completely and is only touched once, after training, using the best checkpoint.

```powershell
python src/train.py --train_data data/Restaurants_Train_v2.csv --epochs 5 --batch_size 8 --learning_rate 2e-5 --max_length 128 --num_heads 6 --output_dir results/restaurants_full --checkpoint_dir checkpoints/restaurants_full
```

For the laptop domain (evaluate separately per the spec — do not mix domains):

```powershell
python src/train.py --train_data data/Laptop_Train_v2.csv --epochs 5 --batch_size 8 --learning_rate 2e-5 --max_length 128 --num_heads 6 --output_dir results/laptop_full --checkpoint_dir checkpoints/laptop_full
```

Expected console output:

```
===================================
DEVICE CONFIGURATION
===================================
Device: CPU
GPU Usage: Disabled
CUDA Available: Not Used
===================================
Total labeled dataset size (from data/Restaurants_Train_v2.csv): 3608
No --val_data/--test_data given: performing an internal stratified 80%/10%/10% train/val/test split of --train_data.

Train dataset size: 2886
Train sentiment label distribution:
  positive  :  1332  (46.2%)
  negative  :   805  (27.9%)
  neutral   :   636  (22.0%)
  conflict  :   113  (3.9%)

Validation dataset size: 361
...
Test dataset size: 361
...

Epoch 3/5  (118.4s)
  train_loss=0.5123  val_loss=0.4821  val_acc=0.8734  val_weighted_f1=0.8690
Checkpoint saved to: checkpoints/restaurants_full/best_model.pt

Loading best checkpoint and evaluating on the held-out test split (never seen during training or validation)...

===== Held-Out Test Set Results =====
Accuracy         : 0.8801
Precision (wtd)  : 0.8756
Recall (wtd)     : 0.8801
Weighted F1      : 0.8770
```

If you ever want to supply your own separate, labeled validation/test files instead of the internal split (e.g. once you have a properly labeled held-out set), pass `--val_data` and/or `--test_data` explicitly — passing an unlabeled Phase A/B file will raise a clear `ValueError` listing the missing columns rather than silently failing.

### Phase 3 — Standalone evaluation of a saved checkpoint (any time later)

```powershell
python src/evaluate.py --checkpoint checkpoints/restaurants_full/best_model.pt --test_data data/Restaurants_Test_Data_PhaseA.csv
```

### Phase 4 — Ablation study (Section 11 of the spec)

Run all three to compare the contribution of each component (spec Section 11):

```powershell
# Experiment 1: BERT baseline
python src/train.py --train_data data/Restaurants_Train_v2.csv --test_data data/Restaurants_Test_Data_PhaseB.csv --no-use_aspect_projection --no-use_category_attention --output_dir results/exp1_baseline --checkpoint_dir checkpoints/exp1_baseline

# Experiment 2: BERT + Aspect Projection
python src/train.py --train_data data/Restaurants_Train_v2.csv --test_data data/Restaurants_Test_Data_PhaseB.csv --use_aspect_projection --no-use_category_attention --output_dir results/exp2_aspect_proj --checkpoint_dir checkpoints/exp2_aspect_proj

# Experiment 3: BERT + Aspect Projection + Category-Aware Attention (full model)
python src/train.py --train_data data/Restaurants_Train_v2.csv --test_data data/Restaurants_Test_Data_PhaseB.csv --use_aspect_projection --use_category_attention --output_dir results/exp3_full --checkpoint_dir checkpoints/exp3_full
```

Compare `results/exp*/test_metrics.json` (weighted F1 especially, since the datasets are class-imbalanced).

---

## 5. Full CLI reference (`train.py`)

| Flag | Default | Meaning |
|---|---|---|
| `--train_data` | *(required)* | Path to the LABELED training CSV (e.g. `Restaurants_Train_v2.csv`) |
| `--val_data` | `None` | Optional separate LABELED validation CSV |
| `--test_data` | `None` | Optional separate LABELED test CSV. Do **not** pass Phase A/B files — they have no labels. Leave unset for the default internal 80/10/10 split |
| `--val_split` | `0.1` | Fraction of the TOTAL dataset held out for validation (10%) when `--val_data`/`--test_data` not given |
| `--test_split` | `0.1` | Fraction of the TOTAL dataset held out for testing (10%) when `--val_data`/`--test_data` not given |
| `--pretrained_name` | `bert-base-uncased` | HuggingFace BERT checkpoint |
| `--epochs` | `5` | Number of training epochs |
| `--batch_size` | `8` | Batch size |
| `--learning_rate` | `2e-5` | AdamW learning rate |
| `--max_length` | `128` | Max BERT sequence length |
| `--num_heads` | `8` | Attention heads (must divide 768: try 6, 8, 12) |
| `--dropout` | `0.1` | Dropout rate |
| `--weight_decay` | `0.01` | AdamW weight decay |
| `--seed` | `42` | Random seed |
| `--early_stopping_patience` | `3` | Epochs with no val-F1 improvement before stopping |
| `--use_aspect_projection` / `--no-use_aspect_projection` | `True` | Toggle aspect projection layer |
| `--use_category_attention` / `--no-use_category_attention` | `True` | Toggle category-aware attention |
| `--output_dir` | `results/default_run` | Where `training_history.json` / `test_metrics.json` are written |
| `--checkpoint_dir` | `checkpoints` | Where `best_model.pt` is written |

---

## 6. What gets saved in a checkpoint

`checkpoints/<run>/best_model.pt` contains:
- `model_state_dict`, `optimizer_state_dict`
- `epoch`, `val_metrics`
- `label2id` (reproducible sentiment-label mapping — required to decode predictions)
- `category2id` (aspect-category mapping if the dataset had one, else `None`)
- `model_config` (hyperparameters needed to reconstruct the exact same model at eval time)

---

## 7. Avoiding the previous environment's known problems

- **CSV vs. Excel:** `src/preprocessing.py` dispatches strictly on file extension (`.csv` → `pd.read_csv`, `.xlsx`/`.xls` → `pd.read_excel`) and raises a clear error for anything else — it never assumes `dataset.xlsx`.
- **Wrong working directory:** all commands above assume you are in the project root (the folder containing `src/`, `data/`, etc.) — that's why the first PowerShell command is always `cd "...\project_folder"`.
- **Dataset path errors:** the dataset path is always a required `--train_data` / `--test_data` CLI argument, never hardcoded.
- **Device:** this project is **CPU-only by design**. `src/utils.py::get_device()` always returns `torch.device("cpu")` and never calls `torch.cuda.is_available()` — even if a GPU is present on your machine, it is ignored. `DataLoader` calls use `num_workers=0` (no `pin_memory`) throughout, since that's only useful for GPU transfers.

---

## 8. Next phase (not implemented here)

`SharedFeatureExtractor.forward()` already returns `token_hidden` (per-token BERT states) alongside the sentiment `fusion` representation. Phase 2 will add:

```python
class AspectExtractionHead(nn.Module):   # token_hidden -> BIO tags
class OpinionExtractionHead(nn.Module):  # token_hidden -> BIO tags
```

and combine losses as `L_total = λ1*L_sentiment + λ2*L_aspect + λ3*L_opinion`, without modifying `model.py`'s existing classes.
