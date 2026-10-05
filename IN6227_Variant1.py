"""IN6227 Assignment 1 Variant 1: reproducible classifier comparison."""

# Plain-text display lets the same cells run outside Jupyter.
display = print

# IN6227 Assignment 1 Variant 1 Starter

# 1. Setup and load the provided data

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_DIR = Path('dataset')
if not (DATA_DIR / 'train.csv').exists():
    DATA_DIR = Path.home() / 'Downloads' / 'dataset'
assert (DATA_DIR / 'train.csv').exists() and (DATA_DIR / 'test.csv').exists(), (
    f'Cannot find train.csv and test.csv in {DATA_DIR.resolve()}'
)
OUTPUT_DIR = Path('assignment1_results')
OUTPUT_DIR.mkdir(exist_ok=True)

train_raw = pd.read_csv(DATA_DIR / 'train.csv')
test_raw = pd.read_csv(DATA_DIR / 'test.csv')
assert list(train_raw.columns) == list(test_raw.columns)
assert 'label' in train_raw.columns
print('Python:', sys.version.split()[0], 'scikit-learn:', sklearn.__version__)
print('Train shape:', train_raw.shape, 'Test shape:', test_raw.shape)
print('Files:', DATA_DIR.resolve())

# 2. Explore data quality

feature_cols = [c for c in train_raw.columns if c != 'label']
numeric_cols = train_raw[feature_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = [c for c in feature_cols if c not in numeric_cols]

print('Numeric features:', numeric_cols)
print('Categorical features:', categorical_cols)
print('Duplicate rows:', train_raw.duplicated().sum(), test_raw.duplicated().sum())
print('Missing label rows:', train_raw['label'].isna().sum(), test_raw['label'].isna().sum())
print('Missing cells by column (train):')
display(train_raw.isna().sum().to_frame('missing_count').T)
print('Target counts (train):')
display(train_raw['label'].value_counts(dropna=False).to_frame('count'))
print('Numeric summary (train):')
display(train_raw[numeric_cols].describe().T.round(2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
train_raw['label'].fillna('Missing').value_counts().plot.bar(
    ax=axes[0], color='#326aa8', rot=0
)
axes[0].set(title='Training labels', xlabel='Label', ylabel='Rows')
train_raw.isna().sum().sort_values(ascending=False).plot.bar(
    ax=axes[1], color='#b8862b'
)
axes[1].set(title='Missing cells by column', xlabel='Column', ylabel='Missing cells')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / '01_labels_and_missingness.png', dpi=180, bbox_inches='tight')
plt.show()

q1 = train_raw[numeric_cols].quantile(0.25)
q3 = train_raw[numeric_cols].quantile(0.75)
iqr = q3 - q1
outlier_counts = (
    (train_raw[numeric_cols] < q1 - 1.5 * iqr) |
    (train_raw[numeric_cols] > q3 + 1.5 * iqr)
).sum().sort_values(ascending=False)
print('IQR outlier flags (not necessarily data errors):')
display(outlier_counts.to_frame('flagged_rows'))

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
for ax, col in zip(axes.flat, numeric_cols):
    ax.hist(train_raw[col].dropna(), bins=35, color='#326aa8', edgecolor='white')
    ax.set(title=col, ylabel='Rows')
for ax in list(axes.flat)[len(numeric_cols):]:
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / '02_numeric_distributions.png', dpi=180, bbox_inches='tight')
plt.show()

# 3. Prepare labels and model-specific pipelines

train = train_raw.dropna(subset=['label']).copy()
test = test_raw.dropna(subset=['label']).copy()
assert set(train['label'].unique()) == {'no', 'yes'}
assert set(test['label'].unique()) == {'no', 'yes'}
X_train, y_train = train[feature_cols], train['label']
X_test, y_test = test[feature_cols], test['label']
print('Usable rows:', len(train), 'training;', len(test), 'test')

def make_preprocessor(scale_numeric):
    num_steps = [('impute', SimpleImputer(strategy='median'))]
    if scale_numeric:
        num_steps.append(('scale', StandardScaler()))
    cat_steps = [
        ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('encode', OneHotEncoder(handle_unknown='ignore')),
    ]
    return ColumnTransformer([
        ('numeric', Pipeline(num_steps), numeric_cols),
        ('categorical', Pipeline(cat_steps), categorical_cols),
    ])

logistic_pipe = Pipeline([
    ('prep', make_preprocessor(scale_numeric=True)),
    ('model', LogisticRegression(solver='liblinear', max_iter=2000, random_state=RANDOM_STATE)),
])
forest_pipe = Pipeline([
    ('prep', make_preprocessor(scale_numeric=False)),
    ('model', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=2)),
])

# 4. Tune with stratified cross-validation on train.csv only

from sklearn.metrics import make_scorer

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
positive_f1 = make_scorer(f1_score, pos_label='yes')

logistic_search = GridSearchCV(
    logistic_pipe,
    param_grid={
        'model__C': [0.1, 1.0, 10.0],
        'model__class_weight': [None, 'balanced'],
    },
    scoring=positive_f1, cv=cv, n_jobs=2, refit=True, error_score='raise',
)
forest_search = GridSearchCV(
    forest_pipe,
    param_grid={
        'model__n_estimators': [120, 200],
        'model__max_depth': [12, None],
        'model__min_samples_leaf': [2],
        'model__class_weight': ['balanced_subsample'],
    },
    scoring=positive_f1, cv=cv, n_jobs=2, refit=True, error_score='raise',
)

logistic_search.fit(X_train, y_train)
print('Logistic Regression best CV F1:', round(logistic_search.best_score_, 4))
print('Logistic Regression params:', logistic_search.best_params_)
forest_search.fit(X_train, y_train)
print('Random Forest best CV F1:', round(forest_search.best_score_, 4))
print('Random Forest params:', forest_search.best_params_)

cv_summary = pd.DataFrame([
    {'model': 'Logistic Regression', 'cv_f1_yes': logistic_search.best_score_,
     'best_params': str(logistic_search.best_params_)},
    {'model': 'Random Forest', 'cv_f1_yes': forest_search.best_score_,
     'best_params': str(forest_search.best_params_)},
])
cv_summary.to_csv(OUTPUT_DIR / '03_cv_summary.csv', index=False)
display(cv_summary)

# 5. Evaluate once on test.csv

models = {
    'Logistic Regression': logistic_search.best_estimator_,
    'Random Forest': forest_search.best_estimator_,
}
metrics_rows = []
confusions = {}
pr_data = {}
predictions = {}

for name, model in models.items():
    pred = model.predict(X_test)
    yes_col = list(model.classes_).index('yes')
    probability_yes = model.predict_proba(X_test)[:, yes_col]
    predictions[name] = (pred, probability_yes)
    confusions[name] = confusion_matrix(y_test, pred, labels=['no', 'yes'])
    precision_curve, recall_curve, _ = precision_recall_curve(
        y_test, probability_yes, pos_label='yes'
    )
    pr_data[name] = (recall_curve, precision_curve)
    metrics_rows.append({
        'model': name,
        'accuracy': accuracy_score(y_test, pred),
        'precision_yes': precision_score(y_test, pred, pos_label='yes'),
        'recall_yes': recall_score(y_test, pred, pos_label='yes'),
        'f1_yes': f1_score(y_test, pred, pos_label='yes'),
        'roc_auc': roc_auc_score((y_test == 'yes').astype(int), probability_yes),
        'average_precision': average_precision_score(
            (y_test == 'yes').astype(int), probability_yes
        ),
    })
    print('\n', name)
    print(classification_report(y_test, pred, labels=['no', 'yes'], zero_division=0))

metrics_df = pd.DataFrame(metrics_rows).set_index('model')
metrics_df.to_csv(OUTPUT_DIR / '04_test_metrics.csv')
print('Always-predict-no accuracy on test:', round((y_test == 'no').mean(), 4))
print('Yes prevalence on test:', round((y_test == 'yes').mean(), 4))
display(metrics_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, matrix) in zip(axes, confusions.items()):
    ax.imshow(matrix, cmap='Blues')
    ax.set(title=name, xlabel='Predicted label', ylabel='True label',
           xticks=[0, 1], yticks=[0, 1], xticklabels=['no', 'yes'],
           yticklabels=['no', 'yes'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{matrix[i, j]:,}', ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
fig.tight_layout()
fig.savefig(OUTPUT_DIR / '05_confusion_matrices.png', dpi=180, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(6, 4.5))
for name, (recall_curve, precision_curve) in pr_data.items():
    ax.plot(recall_curve, precision_curve, label=name)
ax.axhline((y_test == 'yes').mean(), color='gray', linestyle='--', label='Yes prevalence')
ax.set(title='Precision-recall on held-out test data', xlabel='Recall for yes',
       ylabel='Precision for yes', xlim=(0, 1), ylim=(0, 1))
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / '06_precision_recall.png', dpi=180, bbox_inches='tight')
plt.show()

# 6. Inspect errors and save the evidence

error_counts = []
for name, (pred, probability_yes) in predictions.items():
    true_arr = y_test.to_numpy()
    fp = int(((true_arr == 'no') & (pred == 'yes')).sum())
    fn = int(((true_arr == 'yes') & (pred == 'no')).sum())
    error_counts.append({'model': name, 'false_positives': fp, 'false_negatives': fn})
    inspect = X_test.copy()
    inspect['true_label'] = true_arr
    inspect['predicted_label'] = pred
    inspect['probability_yes'] = probability_yes
    inspect = inspect[inspect['true_label'] != inspect['predicted_label']]
    inspect = inspect.sort_values('probability_yes', ascending=False)
    inspect.head(20).to_csv(
        OUTPUT_DIR / f'07_error_examples_{name.lower().replace(" ", "_")}.csv',
        index=False,
    )
error_df = pd.DataFrame(error_counts).set_index('model')
error_df.to_csv(OUTPUT_DIR / '07_error_counts.csv')
display(error_df)
print('Results saved in:', OUTPUT_DIR.resolve())

# 7. Your review before writing
