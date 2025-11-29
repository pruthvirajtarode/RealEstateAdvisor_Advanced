# src/train_xgboost.py
import os
import sys
import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn import __version__ as skl_version
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier, XGBRegressor
from preprocess import load_and_preprocess

# ----------------------- Paths -----------------------
BASE = os.path.dirname(os.path.abspath(__file__))   # src/
DATA_RAW = os.path.normpath(os.path.join(BASE, "..", "data", "india_housing_prices.csv"))
DATA_PROCESSED = os.path.normpath(os.path.join(BASE, "..", "data", "india_housing_prices_processed.csv"))
MODEL_DIR = os.path.normpath(os.path.join(BASE, "..", "models"))
os.makedirs(MODEL_DIR, exist_ok=True)

print("Reading data from:", DATA_RAW)
print("Processed will be saved to:", DATA_PROCESSED)
print("Models will be saved to:", MODEL_DIR)
print("Python:", sys.version.splitlines()[0])
print("scikit-learn:", skl_version)

# ----------------------- Load & preprocess -----------------------
# reuse your preprocess function (it already saves processed CSV)
df = load_and_preprocess(DATA_RAW, save_processed=DATA_PROCESSED)
print("Loaded rows:", len(df))

# ----------------------- REBUILD CLASS LABEL (balanced, realistic) -----------------------
# If Good_Investment column is missing or all zeros, rebuild with a sensible rule
# This creates a mix of 0/1 labels so classifier can learn.
if 'Good_Investment' not in df.columns or df['Good_Investment'].nunique() == 1:
    print("Rebuilding 'Good_Investment' label to ensure both classes are present...")
    # Ensure Price_per_SqFt exists
    if 'Price_per_SqFt' not in df.columns:
        df['Price_per_SqFt'] = (df.get('Price_in_Lakhs', 0) * 100000) / df.get('Size_in_SqFt', 1)
    # sensible rule: lower than median price_per_sqft, decent BHK, and nearby amenities
    median_pps = df['Price_per_SqFt'].median()
    df['Good_Investment'] = (
        (df['Price_per_SqFt'] <= median_pps) &
        (df.get('BHK', 3).fillna(3) >= 2) &
        (df.get('Nearby_Schools', 0).fillna(0) >= 2) &
        (df.get('Nearby_Hospitals', 0).fillna(0) >= 1)
    ).astype(int)
    print("Label distribution after rebuild:\n", df['Good_Investment'].value_counts(normalize=False))

# ----------------------- Features / Targets -----------------------
features = [
    'City','Locality','Property_Type','BHK','Size_in_SqFt','Price_in_Lakhs',
    'Price_per_SqFt','Age_of_Property','Nearby_Schools','Nearby_Hospitals',
    'Public_Transport_Accessibility','Parking_Space','Furnished_Status'
]
for f in features:
    if f not in df.columns:
        df[f] = 0

X = df[features].copy()
y_cls = df['Good_Investment'].copy()
y_reg = df['Future_Price_5Y'].copy() if 'Future_Price_5Y' in df.columns else (df['Price_in_Lakhs'] * ((1+0.08)**5))

numeric = [
    'BHK','Size_in_SqFt','Price_in_Lakhs','Price_per_SqFt','Age_of_Property',
    'Nearby_Schools','Nearby_Hospitals','Public_Transport_Accessibility','Parking_Space'
]
cat = ['City','Locality','Property_Type','Furnished_Status']

# ----------------------- Robust cleaning -----------------------
# numeric -> coerce and replace inf with NaN
X.loc[:, numeric] = (
    X[numeric]
    .apply(pd.to_numeric, errors='coerce')
    .replace([np.inf, -np.inf], np.nan)
)
# fill numeric with medians (computed per column)
medians = X[numeric].median()
X.loc[:, numeric] = X[numeric].fillna(medians)

# categorical -> fill missing with mode (string)
for c in cat:
    if c in X.columns:
        try:
            X.loc[:, c] = X[c].fillna(df[c].mode()[0]).astype(str)
        except Exception:
            X.loc[:, c] = X[c].fillna("Unknown").astype(str)

# ensure no infinite values remain
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

# Clean labels for classifier
y_cls = y_cls.fillna(0).astype(int)
y_cls = y_cls.clip(lower=0, upper=1)

# Regression y: fillna with current price if missing
y_reg = y_reg.fillna(df.get('Price_in_Lakhs', 0))

# Final safety check
if X.isnull().any().any():
    print("Warning: NaNs remain in X after cleaning — filling with 0")
    X = X.fillna(0)

# ----------------------- Train/test split -----------------------
X_train, X_test, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.20, random_state=42, stratify=y_cls if y_cls.nunique()>1 else None
)

X_train_r, X_test_r, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

# ----------------------- OneHotEncoder compatibility -----------------------
def make_ohe():
    ver_parts = tuple(int(x) for x in skl_version.split('.')[:2])
    if ver_parts >= (1, 2):
        # sklearn >= 1.2 supports sparse_output
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    else:
        # older sklearn uses sparse
        return OneHotEncoder(handle_unknown='ignore', sparse=False)

ohe = make_ohe()

# ----------------------- Preprocessor -----------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric),
    ('cat', ohe, cat)
], remainder='drop')

# ----------------------- Sanity transform check -----------------------
# Fit-transform on small sample to detect NaN/Inf after encoding
X_sample = X_train.head(min(5000, len(X_train)))
Xt_sample = preprocessor.fit_transform(X_sample)
try:
    import scipy.sparse as sp
    if sp.issparse(Xt_sample):
        Xt_check = Xt_sample.todense()
    else:
        Xt_check = np.asarray(Xt_sample)
except Exception:
    Xt_check = np.asarray(Xt_sample)

if np.isnan(Xt_check).any() or np.isinf(Xt_check).any():
    print("Warning: NaN/Inf detected in features AFTER preprocessing sample — these will be sanitized before fit.")

# ----------------------- Define models (balanced) -----------------------
clf_xgb = XGBClassifier(
    objective='binary:logistic',
    base_score=0.5,
    n_estimators=160,
    learning_rate=0.06,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
# avoid passing use_label_encoder param explicitly for newest xgboost versions

reg_xgb = XGBRegressor(
    n_estimators=160,
    learning_rate=0.06,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)

clf = Pipeline([('pre', preprocessor), ('xgb', clf_xgb)])
reg = Pipeline([('pre', preprocessor), ('xgb', reg_xgb)])

# ----------------------- Clean data again before fit -----------------------
def safe_fit_pipeline(pipeline, X_tr, y_tr, X_te=None, y_te=None):
    """
    Fit pipeline while guarding against NaN/Inf in transformed arrays.
    If transform produces NaN/Inf, we replace them with 0.
    """
    # Fit preprocessor separately to inspect transformed arrays
    pre = pipeline.named_steps['pre']
    pre.fit(X_tr)
    Xt = pre.transform(X_tr)
    Xv = pre.transform(X_te) if X_te is not None else None

    # convert sparse -> array for cleaning if necessary
    try:
        import scipy.sparse as sp
        if sp.issparse(Xt):
            Xt = Xt.todense()
        if Xv is not None and sp.issparse(Xv):
            Xv = Xv.todense()
    except Exception:
        pass

    Xt = np.asarray(Xt, dtype=float)
    if Xv is not None:
        Xv = np.asarray(Xv, dtype=float)

    # replace NaN/Inf with 0
    if np.isnan(Xt).any() or np.isinf(Xt).any():
        Xt = np.nan_to_num(Xt, nan=0.0, posinf=0.0, neginf=0.0)
    if Xv is not None and (np.isnan(Xv).any() or np.isinf(Xv).any()):
        Xv = np.nan_to_num(Xv, nan=0.0, posinf=0.0, neginf=0.0)

    # Create a shallow clone of pipeline with preprocessor replaced by a passthrough (we already transformed)
    from sklearn.base import clone
    pipeline_cloned = clone(pipeline)
    # replace preprocessor with passthrough
    pipeline_cloned.steps[0] = ('pre', 'passthrough')
    # Now call fit on the final estimator with Xt
    pipeline_cloned.fit(Xt, y_tr)
    return pipeline_cloned, Xt, Xv

# ----------------------- Training with MLflow -----------------------
mlflow.set_experiment("RealEstateAdvisor")

# --- CLASSIFIER TRAIN ---
with mlflow.start_run(run_name="xgb_classification_balanced"):
    print("Training classifier (balanced)...")
    # Fit safely
    try:
        clf_trained, Xt_train, Xt_val = safe_fit_pipeline(clf, X_train, y_train_cls, X_test, y_test_cls)
    except Exception as e:
        print("Error in safe_fit_pipeline (classifier):", e)
        raise

    # Predictions (use transformed arrays)
    preds = clf_trained.predict(Xt_val if Xt_val is not None else Xt_train)
    try:
        proba = clf_trained.predict_proba(Xt_val)[:, 1] if Xt_val is not None else clf_trained.predict_proba(Xt_train)[:,1]
    except Exception:
        proba = None

    acc = accuracy_score(y_test_cls, preds) if (Xt_val is not None) else accuracy_score(y_train_cls, preds)
    mlflow.log_metric("cls_accuracy", float(acc))
    if proba is not None:
        try:
            roc = roc_auc_score(y_test_cls, proba)
            mlflow.log_metric("cls_roc_auc", float(roc))
        except Exception:
            pass

    # Save final classifier pipeline (with fitted preprocessor)
    fitted_pre = clf.named_steps['pre']
    from sklearn.pipeline import Pipeline as SKPipeline
    final_clf_pipeline = SKPipeline([('pre', fitted_pre), ('xgb', clf_trained.named_steps['xgb'])])
    joblib.dump(final_clf_pipeline, os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
    print("Saved classifier:", os.path.join(MODEL_DIR, "xgb_classifier.pkl"))

# --- REGRESSOR TRAIN ---
with mlflow.start_run(run_name="xgb_regression_balanced"):
    print("Training regressor (balanced)...")
    try:
        reg_trained, Xtr_reg, Xval_reg = safe_fit_pipeline(reg, X_train_r, y_train_reg, X_test_r, y_test_reg)
    except Exception as e:
        print("Error in safe_fit_pipeline (regressor):", e)
        raise

    preds_r = reg_trained.predict(Xval_reg if Xval_reg is not None else Xtr_reg)
    rmse = mean_squared_error(y_test_reg, preds_r) if (Xval_reg is not None) else mean_squared_error(y_train_reg, preds_r)
    r2 = r2_score(y_test_reg, preds_r) if (Xval_reg is not None) else r2_score(y_train_reg, preds_r)

    mlflow.log_metric("reg_rmse", float(rmse))
    mlflow.log_metric("reg_r2", float(r2))

    # Save final regressor pipeline (with fitted preprocessor)
    fitted_pre_r = reg.named_steps['pre']
    final_reg_pipeline = Pipeline([('pre', fitted_pre_r), ('xgb', reg_trained.named_steps['xgb'])])
    joblib.dump(final_reg_pipeline, os.path.join(MODEL_DIR, "xgb_regressor.pkl"))
    print("Saved regressor:", os.path.join(MODEL_DIR, "xgb_regressor.pkl"))

print("\nTraining finished. Models saved to:", MODEL_DIR)
