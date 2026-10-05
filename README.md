# IN6227 Assignment 1 Variant 1

This project compares Logistic Regression and Random Forest on the teacher-provided binary classification dataset. It includes a Jupyter notebook and an equivalent Python script.

## Setup

Use Python 3.9 or newer. Put the teacher-provided `train.csv` and `test.csv` in a `dataset/` folder beside the code. On macOS, the code also checks `~/Downloads/dataset/`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

Open `IN6227_Variant1_starter.ipynb` and run all cells, or run:

```bash
python IN6227_Variant1.py
```

The code writes figures and result tables to `assignment1_results/`. It uses five-fold stratified cross-validation on the training file, then evaluates the selected models once on the test file.

Do not commit the teacher-provided dataset or generated result files to a public repository without permission.
