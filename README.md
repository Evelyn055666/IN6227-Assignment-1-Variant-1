# IN6227 Assignment 1 Variant 1

This repository contains the source code for XU YIFAN's Variant 1 report. It compares Logistic Regression and Random Forest on the teacher-provided binary classification dataset. The notebook and Python script run the same analysis; the PDF report is submitted separately to NTULearn.

## Setup

Use Python 3.9 or newer. Put the teacher-provided `train.csv` and `test.csv` in a `dataset/` folder beside the code. On macOS, the code also checks `~/Downloads/dataset/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `IN6227_Variant1_starter.ipynb` and run all cells in order, or run:

```bash
python IN6227_Variant1.py
```

The code writes figures and result tables to `assignment1_results/`. It uses five-fold stratified cross-validation on the training file, then evaluates the selected models once on the test file. With the supplied CSVs and the tested environment (Python 3.9.6, scikit-learn 1.6.1), the held-out `yes`-class F1 scores are 0.648 for Logistic Regression and 0.663 for Random Forest. The exact values are saved in `04_test_metrics.csv`; false-positive and false-negative counts are in `07_error_counts.csv`.

Do not commit the teacher-provided dataset or generated result files to a public repository without permission.
