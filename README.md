# Freight Rate Prediction

Predict how much a freight load will pay (`posted_rate`) from its route, distance, equipment, weight and date.

- **Train on:** 48,000 labeled loads from Jan–Oct 2025 (`data/train_test.csv`)
- **Predict:** 12,000 unlabeled loads from Nov–Dec 2025 (`data/validation.csv`)
- **Also predict:** one fixed load (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) for each day of December, shown as a chart

Because the loads to predict come *after* the training period, this is a **forecasting** problem. That one fact drives
most of the design below.

---

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
python -m pip install -r requirements.txt
python run_pipeline.py
```

- Prefer a notebook? Open `model.ipynb` and *Run All*. It is the same code with an explanation next to every step.
- Google Colab: upload the project, `%cd` into its folder, then run the notebook.
- Setup helpers: `setup_env.sh` (macOS/Linux), `setup_env.bat` (Windows), `environment.yml` (conda).

The run is seeded, so it reproduces the same predictions every time.

Check the output files with the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

---

## What the data looks like

| Column | Meaning |
|---|---|
| `pickup`, `delivery`, + lat/lon | origin and destination city |
| `distance` | miles (noisy: the same city pair shows slightly different values) |
| `equipment` | Dry Van, Flatbed or Reefer |
| `weight` | load weight in lb |
| `date` | pickup date |
| `market_index`, `quote_signal` | market indicators |
| `posted_rate` | **target**, only in train |

What the exploration showed:

- Rate is almost a straight line in distance (correlation 0.91), with an equipment premium on top
  (Reefer pays most per mile, Dry Van least).
- Rate per mile follows a seasonal curve: about $2.03 in January, a peak of $2.25 in June, then roughly $2.16 in autumn.

### Data-quality problems and how each was handled

| Problem | Size | What was done and why |
|---|---|---|
| Negative weights | 292 train / 145 validation | Looks like a sign error (similar size to valid weights), so take `abs()` and keep the row |
| Missing weight | 300 / 165 | Fill with the train median; add a "was missing" flag |
| Missing `market_index` | 374 / 249 | Fill with the same-date median, else the train median |
| Label outliers (rate per mile under 0.5× or over 2× the norm for that equipment and month) | 672 train rows (1.4%) | Two clear noise clumps, so they are treated as wrong labels and **dropped from train only** |
| Cities and lanes never seen in training | 8 cities; 12.2% of validation rows | Nothing dropped; city/lane effects fall back to neutral and the model relies on distance, weight, equipment |
| Duplicates, bad dates | none | none needed |

Golden rule: **never drop rows from validation.** All 12,000 predictions are required, so validation rows are fixed or
filled, not removed.

---

## How the solution works, step by step

**1. Split the data by time, not randomly.**
The real task is predicting the *next* two months, so validation mimics that with an expanding window:

| Fold | Train on | Test on |
|---|---|---|
| 1 | Jan–Jun | Jul–Aug |
| 2 | Jan–Jul | Aug–Sep |
| 3 | Jan–Aug | Sep–Oct |

A random split mixes loads from the same days into train and test, which lets a model use information it won't have
in real use. Section "Why the split matters" below shows a real example of this.

**2. Learn only from the training window (no leakage).**
Fill values, the outlier medians, the baseline line and the city/lane effects are all computed from training rows only,
then applied to the test rows. City and lane effects on training rows are computed out-of-fold, so no row sees its own label.

**3. Build features** (`features.py`, 27 in total):
- distance: raw, log, straight-line, and the lane's typical distance
- load: weight, weight-missing flag, equipment
- geography: city coordinates and direction of travel
- calendar: day of week, weekend, day of month, near a US holiday (month is *left out* on purpose: train has no Nov/Dec)
- `quote_signal`
- the baseline line's prediction, plus a learned premium or discount per origin city, destination city and lane

**4. Model: a line plus trees.**
1. A simple line, `log(rate) ~ log(distance)` fitted per equipment type, captures the big, predictable effect.
2. LightGBM then predicts only what the line missed (weight, quote signal, cities, calendar).

Trees cannot extrapolate beyond what they have seen, but a line can, so this split of work is safer for a forecast.
The model works on `log(rate)` so errors are percentage errors, uses an L1 (absolute-error) loss so a few noisy
labels have little pull, and averages 5 random seeds for stability.

**5. Final model and predictions.**
Refit on all Jan–Oct data, predict the 12,000 validation loads, and fill `validation_predictions.csv` in the template's order.

**6. December chart.**
The chart file has no `quote_signal`, so each date uses that date's average `quote_signal` from `validation.csv`
(a feature, not a label). `score.py` then draws the chart.

---

## Results

Average over the three time-based folds. Lower is better.
"Raw" scores all test rows. "Clean" leaves out test rows whose label looks wrong, to show model quality without noise.

| Model | MAE $ | MAPE % | Clean MAE $ | Clean MAPE % |
|---|---|---|---|---|
| Naive: median rate per mile × distance | 255.7 | 11.42 | 205.6 | 9.23 |
| Linear: distance × equipment | 131.9 | 5.87 | 79.5 | 3.56 |
| Linear + weight | 117.0 | 5.28 | 64.5 | 2.96 |
| **Line + LightGBM on its residual (final)** | **102.6** | **4.40** | **50.0** | **2.08** |
| Final, without `quote_signal` | 102.8 | 4.44 | 50.2 | 2.12 |
| Final, with `market_index` features | 140.6 | 5.96 | 88.4 | 3.74 |

Spotter computes the real metric after submission and does not say which one it is, so both MAE and MAPE are reported.

### Why the split matters

The same two models, scored on a random 80/20 split and on the time-based folds:

| Model | Random split MAPE | Time-based MAPE |
|---|---|---|
| Without `market_index` | 4.90% | 4.40% |
| With `market_index` | 4.23% | 5.96% |

With a random split, `market_index` looks like the best model. On the time-based split, which matches the real task,
it is the worst. So `market_index` was left out. A random split would have chosen the model that fails at forecasting.

### December chart

Predictions run from about $800 to $818. The same lane and equipment paid roughly $778–$877 per month (scaled to 360 miles)
in Jan–Oct, so the level is in line with history.

---

## Output files

| File | Description |
|---|---|
| `validation_predictions.csv` | **Submission file**: `load_id,predicted_rate`, 12,000 rows |
| `data/december_chart_inputs.csv` | the 31 December rows with `predicted_rate` filled in |
| `scorer_results/candidate_december.png` | December chart produced by `score.py` |
| `report/Freight_Rate_Report.docx` and `.pdf` | report: validation approach and the chart |
| `figures/`, `results/` | charts and metric tables used in the report |

`data/validation_predictions_template.csv` is only a template; it is never edited. The pipeline reads its `load_id` order
and writes the new `validation_predictions.csv`.

## Repository layout

```
model.ipynb          explained walkthrough: data audit -> split -> models -> predictions
run_pipeline.py      the same code as a plain script
features.py          cleaning and feature engineering (FeatureBuilder: fit on train, transform the rest)
score.py             provided scorer (unchanged)
data/                input CSVs
figures/ results/    charts and metric tables
report/              Word and PDF report
requirements.txt     pinned package versions
```

## Limitations

- November and December have no training history. If the market level shifts seasonally, the model cannot know.
  The December chart therefore varies only about ±1%, and only through inputs that change by date (day of week,
  holiday proximity, daily `quote_signal`).
- About 1.4% of labels look wrong and cannot be predicted. They are why raw MAPE (4.40%) is about double clean MAPE (2.08%).
- About 12% of validation loads are on lanes never seen in training, so they rely on distance, weight, equipment and
  `quote_signal` rather than lane history.
- All scores come from three folds with light tuning, so treat them as a guide, not a guarantee of the final score.
