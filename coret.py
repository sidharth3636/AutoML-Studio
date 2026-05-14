import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)

from xgboost import XGBClassifier, XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier


# =====================================================
# SEARCH SPACES FOR BAYESIAN OPTIMIZATION
# =====================================================

CLASSIFICATION_SEARCH_SPACES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=500),
        {
            "C": Real(0.01, 10.0, prior="log-uniform"),
            "solver": Categorical(["lbfgs", "saga"]),
        }
    ),
    "Random Forest": (
        RandomForestClassifier(),
        {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 5),
        }
    ),
    "Extra Trees": (
        ExtraTreesClassifier(),
        {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 5),
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(),
        {
            "n_estimators": Integer(50, 300),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.6, 1.0),
        }
    ),
    "XGBoost": (
        XGBClassifier(tree_method="hist", eval_metric="logloss"),
        {
            "n_estimators": Integer(50, 300),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_alpha": Real(0.0, 1.0),
            "reg_lambda": Real(0.5, 2.0),
        }
    ),
}

REGRESSION_SEARCH_SPACES = {
    "Linear Regression": (
        LinearRegression(),
        {}   # no hyperparameters — skipped during Bayesian search
    ),
    "Ridge": (
        Ridge(),
        {
            "alpha": Real(0.01, 100.0, prior="log-uniform"),
        }
    ),
    "Lasso": (
        Lasso(),
        {
            "alpha": Real(0.01, 100.0, prior="log-uniform"),
        }
    ),
    "Random Forest": (
        RandomForestRegressor(),
        {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 5),
        }
    ),
    "Extra Trees": (
        ExtraTreesRegressor(),
        {
            "n_estimators": Integer(50, 300),
            "max_depth": Integer(3, 20),
            "min_samples_split": Integer(2, 10),
            "min_samples_leaf": Integer(1, 5),
        }
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(),
        {
            "n_estimators": Integer(50, 300),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.6, 1.0),
        }
    ),
    "XGBoost": (
        XGBRegressor(tree_method="hist"),
        {
            "n_estimators": Integer(50, 300),
            "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            "max_depth": Integer(3, 10),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.6, 1.0),
            "reg_alpha": Real(0.0, 1.0),
            "reg_lambda": Real(0.5, 2.0),
        }
    ),
}


# =====================================================
# TIME-SAFE SPLIT
# =====================================================

def time_safe_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]
    return X_train, X_test, y_train, y_test


# =====================================================
# SUCCESSIVE HALVING — PHASE 1
# Train all models on increasing data fractions.
# Eliminate weak models at each round.
# Returns top-k survivors ranked by final score.
# =====================================================

def successive_halving(
    models_dict,        # {name: model_instance}
    X_train, y_train,
    X_val,   y_val,
    problem_type,
    top_k=2,
    halving_rounds=3    # number of elimination rounds
):
    """
    Round 1: Train all models on 1/halving_rounds of data  → keep top half
    Round 2: Train survivors on 2/halving_rounds of data   → keep top half
    ...
    Final  : Train survivors on full data                  → return top_k
    """

    print(f"\n{'='*55}")
    print(f"  ⚡ SUCCESSIVE HALVING — {problem_type.upper()}")
    print(f"  Models : {list(models_dict.keys())}")
    print(f"  Rounds : {halving_rounds}  |  Final top-k : {top_k}")
    print(f"{'='*55}")

    candidates = list(models_dict.items())   # [(name, model), ...]
    n_samples  = len(X_train)

    for rnd in range(1, halving_rounds + 1):

        # Fraction of training data used in this round
        fraction  = rnd / halving_rounds
        n_use     = max(100, int(n_samples * fraction))

        # On last round use all data
        if rnd == halving_rounds:
            n_use = n_samples

        X_sub = X_train.iloc[:n_use]
        y_sub = y_train.iloc[:n_use] if hasattr(y_train, "iloc") else y_train[:n_use]

        print(f"\n  📊 Round {rnd}/{halving_rounds} — "
              f"{n_use} samples ({int(fraction*100)}%) — "
              f"{len(candidates)} model(s)")

        round_scores = []

        for name, model in candidates:

            try:
                model.fit(X_sub, y_sub)
                preds = model.predict(X_val)

                if problem_type == "Classification":
                    if isinstance(y_val, pd.DataFrame):
                        score = np.mean([
                            accuracy_score(y_val.iloc[:, i], preds[:, i])
                            for i in range(y_val.shape[1])
                        ])
                    else:
                        score = accuracy_score(y_val, preds)
                else:
                    if isinstance(y_val, pd.DataFrame):
                        score = np.mean([
                            np.sqrt(mean_squared_error(y_val.iloc[:, i], preds[:, i]))
                            for i in range(y_val.shape[1])
                        ])
                    else:
                        score = np.sqrt(mean_squared_error(y_val, preds))

                metric_label = "Acc" if problem_type == "Classification" else "RMSE"
                print(f"     {name:<30} {metric_label}: {round(score, 4)}")
                round_scores.append((name, score, model))

            except Exception as e:
                print(f"     {name} — failed: {e}")

        # Sort: higher is better for classification, lower for regression
        if problem_type == "Classification":
            round_scores.sort(key=lambda x: x[1], reverse=True)
        else:
            round_scores.sort(key=lambda x: x[1])

        # Eliminate bottom half (keep at least top_k)
        keep_n    = max(top_k, len(round_scores) // 2)
        survivors = round_scores[:keep_n]

        print(f"\n  ✅ Survivors after round {rnd}: "
              f"{[s[0] for s in survivors]}")

        # Rebuild candidates from survivors for next round
        candidates = [(name, model) for name, _, model in survivors]

        # Stop early if already at top_k
        if len(candidates) <= top_k:
            break

    # Final ranking — return top_k
    final_ranking = [(name, score) for name, score, _ in survivors[:top_k]]
    top_models    = {name: model for name, model in candidates[:top_k]}

    print(f"\n  🏅 Top-{top_k} models selected for Bayesian tuning:")
    for rank, (name, score) in enumerate(final_ranking, 1):
        metric = "Acc" if problem_type == "Classification" else "RMSE"
        print(f"     {rank}. {name}  ({metric}: {round(score, 4)})")

    return top_models, final_ranking


# =====================================================
# BAYESIAN OPTIMIZATION — PHASE 2
# Run BayesSearchCV only on the top-k survivors.
# =====================================================

def optimize_model(name, base_model, search_space, X_train, y_train,
                   problem_type, usable_cores, n_iter=20, cv=3):

    # Models with no tunable params (e.g. Linear Regression) — skip
    if not search_space:
        print(f"  ⏭  {name} — no search space, skipping Bayesian tuning")
        base_model.fit(X_train, y_train)
        return base_model, {}, None

    scoring = "accuracy" if problem_type == "Classification" else "neg_root_mean_squared_error"

    print(f"\n  🔎 Bayesian Search: {name}  ({n_iter} iters, {cv}-fold CV)")

    opt = BayesSearchCV(
        base_model,
        search_space,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=usable_cores,
        random_state=42,
        refit=True,
        verbose=0,
    )

    opt.fit(X_train, y_train)

    print(f"     Best CV score : {opt.best_score_, 4}")
    print(f"     Best params   : {dict(opt.best_params_)}")

    return opt.best_estimator_, dict(opt.best_params_), opt.best_score_


# =====================================================
# AUTO LOG-TRANSFORM HELPER
# =====================================================

def auto_log_transform(y, label="target"):
    """
    Decides whether to log-transform a regression target and applies it.

    Rules (ALL three must be true to transform):
      1. All values are strictly positive  — log(0) and log(-ve) are undefined.
      2. Absolute skew > 1                 — distribution is meaningfully skewed.
      3. More than one order of magnitude  — values span a wide numeric range.

    Parameters
    ----------
    y     : pd.Series or pd.DataFrame  — target column(s)
    label : str                        — name used in print messages

    Returns
    -------
    y_out          : transformed target (same type as input)
    log_flags      : dict  {col_name: True/False}  — which columns were transformed
    transform_info : dict  {col_name: {"skew": float, "range": (min, max)}}
    """

    # Normalise to DataFrame so single-target and multi-target share one code path
    if isinstance(y, pd.Series):
        y_df     = y.to_frame()
        was_series = True
    else:
        y_df     = y.copy()
        was_series = False

    log_flags      = {}
    transform_info = {}
    y_out          = y_df.copy()

    for col in y_df.columns:
        col_vals   = y_df[col].dropna()
        skewness   = col_vals.skew()
        all_pos    = (col_vals > 0).all()
        col_min    = col_vals.min()
        col_max    = col_vals.max()

        # Check if values span more than one order of magnitude
        wide_range = (col_max / col_min) > 10 if (all_pos and col_min > 0) else False

        should_transform = all_pos and (abs(skewness) > 1) and wide_range

        log_flags[col]      = should_transform
        transform_info[col] = {"skew": round(skewness, 3), "range": (col_min, col_max)}

        if should_transform:
            y_out[col] = np.log(y_df[col])
            print(f"  🔁 '{col}' → log-transformed  "
                  f"(skew={skewness:.2f}, range=[{col_min:.3e}, {col_max:.3e}])")
        else:
            reason = []
            if not all_pos:    reason.append("has non-positive values")
            if abs(skewness) <= 1: reason.append(f"skew={skewness:.2f} ≤ 1")
            if not wide_range: reason.append("range not wide enough")
            print(f"  ⏭  '{col}' → no transform  ({', '.join(reason)})")

    # Return same type as input
    if was_series:
        col = y_df.columns[0]
        return y_out[col], log_flags, transform_info

    return y_out, log_flags, transform_info


def reverse_log_transform(preds, log_flags, target_columns):
    """
    Reverses log-transform on predictions for columns that were transformed.
    Always returns a numpy array aligned with target_columns order.

    Parameters
    ----------
    preds          : np.ndarray  — raw model predictions (n_samples,) or (n_samples, n_targets)
    log_flags      : dict        — {col_name: True/False} from auto_log_transform
    target_columns : list        — ordered list of target column names

    Returns
    -------
    np.ndarray of the same shape as preds, with exp() applied where needed.
    """

    preds_out = preds.copy().astype(float)

    if preds_out.ndim == 1:
        # Single target
        col = target_columns[0]
        if log_flags.get(col, False):
            preds_out = np.exp(preds_out)
    else:
        # Multi-target: one column per index
        for i, col in enumerate(target_columns):
            if log_flags.get(col, False):
                preds_out[:, i] = np.exp(preds_out[:, i])

    return preds_out


# =====================================================
# MAIN AUTOML FUNCTION
# =====================================================

def run_basic_automl(
    csv_path,
    target,
    problem_type,
    cpu_percent=100,
    time_series_mode=False,
    optimize=True,          # False → skip Bayesian search (just halving + baseline)
    n_iter=20,              # Bayesian iterations per surviving model
    cv=3,                   # CV folds during Bayesian search
    halving_rounds=3,       # Successive halving elimination rounds
    top_k=2                 # Number of top models to pass to Bayesian tuning
):

    print("📥 Loading dataset...")
    data = pd.read_csv(csv_path)

    if len(data) > 100000:
        print("Large dataset detected. Sampling 100,000 rows.")
        data = data.sample(100000, random_state=42)

    total_cores  = os.cpu_count()
    usable_cores = max(1, int((cpu_percent / 100) * total_cores))
    print(f"\n🖥  Using {usable_cores}/{total_cores} cores")

    # --------------------------------------------------
    # TARGET HANDLING
    # --------------------------------------------------

    if isinstance(target, list):
        if len(target) == 1:
            X = data.drop(columns=target)
            y = data[target[0]]
        else:
            X = data.drop(columns=target)
            y = data[target]
    else:
        X = data.drop(columns=[target])
        y = data[target]

    is_multi_output = isinstance(y, pd.DataFrame)

    # --------------------------------------------------
    # AUTO LOG-TRANSFORM (Regression only)
    # --------------------------------------------------

    log_flags      = {}
    transform_info = {}
    target_cols_list = target if isinstance(target, list) else [target]

    if problem_type == "Regression":
        print("\n🔍 Checking target(s) for auto log-transform...")
        y, log_flags, transform_info = auto_log_transform(y, label="target")
        any_transformed = any(log_flags.values())
        if any_transformed:
            print("   ✅ Log-transform applied. Predictions will be exp()-reversed before metrics.")
        else:
            print("   ✅ No log-transform needed.")

    # --------------------------------------------------
    # DATA SPLIT
    # --------------------------------------------------

    if time_series_mode:
        X_train_full, X_test, y_train_full, y_test = time_safe_split(X, y)
    else:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Validation split used during successive halving
    X_train_sh, X_val, y_train_sh, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # BUILD INITIAL MODEL POOL (default params, n_jobs set)
    # --------------------------------------------------

    search_spaces = (
        CLASSIFICATION_SEARCH_SPACES
        if problem_type == "Classification"
        else REGRESSION_SEARCH_SPACES
    )

    def make_model(base_model):
        """Wrap in MultiOutput if needed, set n_jobs."""
        if hasattr(base_model, "n_jobs"):
            base_model.set_params(n_jobs=usable_cores)
        if is_multi_output:
            wrapper = MultiOutputClassifier if problem_type == "Classification" else MultiOutputRegressor
            return wrapper(base_model)
        return base_model

    initial_models = {
        name: make_model(base_model)
        for name, (base_model, _) in search_spaces.items()
    }

    start_time  = time.time()
    best_params = {}

    # ==================================================
    # PHASE 1 — SUCCESSIVE HALVING
    # Train all models on growing data fractions.
    # Eliminate weak performers each round.
    # Output: top_k survivors.
    # ==================================================

    top_models, halving_ranking = successive_halving(
        models_dict    = initial_models,
        X_train        = X_train_sh,
        y_train        = y_train_sh,
        X_val          = X_val,
        y_val          = y_val,
        problem_type   = problem_type,
        top_k          = top_k,
        halving_rounds = halving_rounds
    )

    # ==================================================
    # PHASE 2 — BAYESIAN OPTIMIZATION on top_k survivors
    # ==================================================

    print(f"\n{'='*55}")
    print(f"  🧠 BAYESIAN OPTIMIZATION on Top-{top_k} Survivors")
    print(f"{'='*55}")

    final_results = []

    for name in top_models.keys():

        _, search_space = search_spaces[name]

        # Get a fresh base model (not the halving-fitted one)
        base_model_fresh, _ = search_spaces[name]

        if optimize and not is_multi_output:

            tuned_model, params, _ = optimize_model(
                name           = name,
                base_model     = base_model_fresh,
                search_space   = search_space,
                X_train        = X_train_full,
                y_train        = y_train_full,
                problem_type   = problem_type,
                usable_cores   = usable_cores,
                n_iter         = n_iter,
                cv             = cv
            )
            best_params[name] = params

        else:
            # Multi-output or optimize=False → just retrain on full data
            tuned_model = top_models[name]
            tuned_model.fit(X_train_full, y_train_full)
            best_params[name] = {}

        # Evaluate on test set
        preds = tuned_model.predict(X_test)

        # Reverse log-transform predictions (and y_test) before metrics
        # so all reported scores are in the original interpretable scale.
        if problem_type == "Regression" and any(log_flags.values()):
            preds_eval  = reverse_log_transform(preds, log_flags, target_cols_list)
            y_test_eval = reverse_log_transform(
                y_test.values if isinstance(y_test, pd.DataFrame) else y_test.values.reshape(-1, 1),
                log_flags, target_cols_list
            )
            if isinstance(y_test, pd.DataFrame):
                y_test_eval = pd.DataFrame(y_test_eval, columns=target_cols_list)
            else:
                y_test_eval = y_test_eval.flatten()
        else:
            preds_eval  = preds
            y_test_eval = y_test

        if problem_type == "Classification":
            if is_multi_output:
                score = np.mean([
                    accuracy_score(y_test_eval.iloc[:, i], preds_eval[:, i])
                    for i in range(y_test.shape[1])
                ])
            else:
                score = accuracy_score(y_test_eval, preds_eval)
            print(f"\n  {name} — Test Accuracy : {score}")
        else:
            if is_multi_output:
                score = np.mean([
                    np.sqrt(mean_squared_error(y_test_eval.iloc[:, i], preds_eval[:, i]))
                    for i in range(y_test.shape[1])
                ])
            else:
                score = np.sqrt(mean_squared_error(y_test_eval, preds_eval))
            print(f"\n  {name} — Test RMSE : {score}")

        final_results.append((name, score, tuned_model))

    # --------------------------------------------------
    # RANK FINAL RESULTS
    # --------------------------------------------------

    if problem_type == "Classification":
        final_results.sort(key=lambda x: x[1], reverse=True)
    else:
        final_results.sort(key=lambda x: x[1])

    best_name, best_score, best_model = final_results[0]

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    preds = best_model.predict(X_test)

    # Reverse log-transform for final summary metrics
    if problem_type == "Regression" and any(log_flags.values()):
        preds_eval  = reverse_log_transform(preds, log_flags, target_cols_list)
        y_test_eval = reverse_log_transform(
            y_test.values if isinstance(y_test, pd.DataFrame) else y_test.values.reshape(-1, 1),
            log_flags, target_cols_list
        )
        if isinstance(y_test, pd.DataFrame):
            y_test_eval = pd.DataFrame(y_test_eval, columns=target_cols_list)
        else:
            y_test_eval = y_test_eval.flatten()
    else:
        preds_eval  = preds
        y_test_eval = y_test

    mae = rmse = r2 = None

    print(f"\n{'='*55}")
    print(f"  🏆 BEST MODEL  : {best_name}")
    print(f"  {'Accuracy' if problem_type == 'Classification' else 'RMSE'} : {best_score}")
    print(f"  Best Params  : {best_params.get(best_name, {})}")

    if problem_type == "Regression":
        mae  = mean_absolute_error(y_test_eval, preds_eval)
        rmse = np.sqrt(mean_squared_error(y_test_eval, preds_eval))
        r2   = r2_score(y_test_eval, preds_eval)
        print(f"  MAE  : {mae}")
        print(f"  RMSE : {rmse}")
        print(f"  R²   : {r2}")

    print(f"{'='*55}")

    total_time = round(time.time() - start_time, 2)
    print(f"\n⏱  Total time: {total_time}s")

    return (
        (best_name, best_score),
        [(name, score) for name, score, _ in final_results],
        best_params,
        best_model,
        mae,
        rmse,
        r2,
        halving_ranking,                  # ← successive halving leaderboard
        top_models,                       # ← the top-k models that were tuned
        best_params.get(best_name, {}),   # ← best model's params specifically
        total_time
    )
