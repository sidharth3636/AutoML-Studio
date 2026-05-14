import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import IsolationForest
import os
import numpy as np
import random
import re
from imblearn.over_sampling import SMOTE


# =====================================================
# DATE FEATURE ENGINEERING
# =====================================================

def detect_and_engineer_dates(df, threshold=0.8):
    """
    Detects every temporal column and engineers features from it.

    For each detected date/datetime column the function will:
      1. Parse the column with errors='coerce' to standardise all date formats.
      2. Always produce:  {col}_year, {col}_month, {col}_day
      3. If time information is present (any non-midnight hour exists):
             {col}_time  — stored as total seconds since midnight (int)
      4. Drop the original column.
      5. Normalise all newly created numeric columns with MinMaxScaler
         (each column independently scaled to [0, 1]).

    Handles three column shapes
    ───────────────────────────
    DATE + TIME  │ "5/5/2025 14:30"  → year, month, day, time
    DATE ONLY    │ "5/5/2025"        → year, month, day
    TIME ONLY    │ "14:30:00"        → time  (year/month/day skipped — all NaT)

    Implementation notes
    ────────────────────
    FIX 1 — pandas 3.0 dtype change
        pd.api.types.is_string_dtype() handles both "object" (pandas <3)
        and "str" / "string" (pandas >=3) correctly.

    FIX 2 — NaN rows inflate detection ratios
        Both pattern_ratio and valid_ratio are computed only on non-null rows
        so sparse columns (many NaNs) are still detected correctly.

    FIX 3 — Live iteration bug
        Snapshot df.columns with list() before the loop to avoid iterator
        corruption when columns are dropped/added inside the loop.

    FIX 4 — Time-only columns
        Added time_only_pattern so pure time strings ("12:00:00") are
        recognised even though they contain no date separator.
    """

    # Matches the DATE part: "5/5/2025", "2025-05-05", etc.
    date_pattern = r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"

    # Matches TIME-ONLY values: "12:00:00", "1:00:00 AM", "23:59:59"
    time_only_pattern = r"^\d{1,2}:\d{2}(:\d{2})?(\s*(AM|am|PM|pm))?$"

    # FIX 3: Snapshot columns before loop
    cols_to_process = list(df.columns)

    for col in cols_to_process:

        # FIX 1: is_string_dtype works across pandas versions
        if not pd.api.types.is_string_dtype(df[col]):
            continue

        # Guard: column may have been dropped in a previous iteration
        if col not in df.columns:
            continue

        temp = df[col].astype(str).str.strip()

        # FIX 2: Work only on non-null rows for detection
        non_null_mask  = df[col].notna()
        non_null_count = non_null_mask.sum()

        if non_null_count == 0:
            continue   # entirely empty column — skip

        temp_non_null = temp[non_null_mask]

        # Classify the column
        date_ratio      = temp_non_null.str.contains(date_pattern,     regex=True).mean()
        time_only_ratio = temp_non_null.str.match(time_only_pattern).mean()

        is_datetime  = date_ratio      >= 0.5
        is_time_only = time_only_ratio >= 0.5

        if not is_datetime and not is_time_only:
            continue   # not a temporal column — skip

        # ── Step 1: Parse with errors='coerce' ───────────────────────────────
        # errors='coerce' converts any unparseable value to NaT, enforcing a
        # single standard datetime representation across all rows.
        parsed = pd.to_datetime(temp, errors="coerce", dayfirst=True)

        # FIX 2 (continued): valid_ratio over non-null rows only
        valid_ratio = parsed[non_null_mask].notna().mean()

        if valid_ratio < threshold:
            continue   # too many unparseable non-null values — skip

        # Temporarily store parsed column
        df[col] = parsed

        # ── Step 2: Determine whether time information is meaningful ─────────
        # "has_time" is True when the hour is non-zero for at least one row,
        # i.e. the timestamps are not all midnight placeholders.
        has_time = (df[col].dt.hour != 0).any()

        if is_datetime:
            if has_time:
                print(f"📅 Detected DateTime Column: '{col}' (date + time)")
            else:
                print(f"📅 Detected Date Column: '{col}' (date only)")
        else:
            print(f"⏰ Detected Time-Only Column: '{col}'")

        # Collect names of the new columns so we can normalise them later
        new_cols = []

        # ── Step 3a: Date components (skipped for pure time-only columns) ────
        if is_datetime:
            df[f"{col}_year"]  = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"]   = df[col].dt.day
            new_cols += [f"{col}_year", f"{col}_month", f"{col}_day"]
            print(f"   ↳ year, month, day : added")

        # ── Step 3b: Time column (seconds since midnight) ────────────────────
        # Stored as a single integer so it is easy to normalise and use in
        # any ML model without further processing.
        if has_time or is_time_only:
            df[f"{col}_time"] = (
                df[col].dt.hour   * 3600
                + df[col].dt.minute * 60
                + df[col].dt.second
            )
            new_cols.append(f"{col}_time")
            print(f"   ↳ time             : added  (seconds since midnight)")

        # ── Step 4: Drop the original date column ────────────────────────────
        df.drop(columns=[col], inplace=True)
        print(f"   ↳ original column '{col}' dropped")

        # ── Step 5: Normalise the newly created columns with MinMaxScaler ────
        # Each column is scaled independently to [0, 1].
        # NaNs (from NaT rows) are temporarily filled with the column median
        # for scaling, then restored to NaN afterwards.
        if new_cols:
            scaler = MinMaxScaler()
            for nc in new_cols:
                col_data   = df[nc].astype(float)
                median_val = col_data.median()
                filled     = col_data.fillna(median_val).values.reshape(-1, 1)
                scaled     = scaler.fit_transform(filled).flatten()
                # Re-apply NaN mask
                scaled[col_data.isna().values] = np.nan
                df[nc] = scaled
            print(f"   ↳ normalised (MinMax [0,1]): {new_cols}")

    return df


# =====================================================
# NUMERIC TEXT DETECTION
# =====================================================

def _encode_category_col(series, col_name, encoder_dir=None):
    """
    Encode a single object Series using the appropriate strategy:

      - 2 unique values  → Binary   (LabelEncoder)
      - 3–10 unique      → One-Hot  (get_dummies)
      - 11–50 unique     → Label    (LabelEncoder, saved to disk)
      - >50 unique       → Frequency encoding

    Returns a DataFrame of one or more encoded columns.
    Saves encoders to encoder_dir when provided.
    """

    # Fill NaN with the placeholder string so they get their own bucket
    s = series.fillna("_MISSING_").astype(str)

    # Collapse rare values (< 1 % of rows) into "Other"
    freq_map = s.value_counts(normalize=True)
    rare     = freq_map[freq_map < 0.01].index
    s        = s.replace(rare, "Other")

    unique_vals = s.nunique()

    if unique_vals == 2:
        print(f"      [{col_name}] → Binary Encoding  ({unique_vals} unique)")
        le = LabelEncoder()
        encoded = pd.Series(le.fit_transform(s), index=series.index, name=col_name)
        print(f"         Label mapping: { {cls: idx for idx, cls in enumerate(le.classes_)} }")
        if encoder_dir:
            with open(os.path.join(encoder_dir, f"{col_name}_encoder.pkl"), "wb") as f:
                pickle.dump(le, f)
        return encoded.to_frame()

    elif unique_vals <= 10:
        print(f"      [{col_name}] → One-Hot Encoding  ({unique_vals} unique)")
        dummies = pd.get_dummies(s, prefix=col_name, dtype=int)
        dummies.index = series.index
        return dummies

    elif unique_vals <= 50:
        print(f"      [{col_name}] → Label Encoding    ({unique_vals} unique)")
        le = LabelEncoder()
        encoded = pd.Series(le.fit_transform(s), index=series.index, name=col_name)
        print(f"         Label mapping: { {cls: idx for idx, cls in enumerate(le.classes_)} }")
        if encoder_dir:
            with open(os.path.join(encoder_dir, f"{col_name}_encoder.pkl"), "wb") as f:
                pickle.dump(le, f)
        return encoded.to_frame()

    else:
        print(f"      [{col_name}] → Frequency Encoding ({unique_vals} unique)")
        freq_vals = s.value_counts(normalize=True)
        encoded   = s.map(freq_vals)
        encoded.name = col_name
        return encoded.to_frame()


def detect_and_convert_numeric_text(df, threshold=0.7, encoder_dir=None):
    """
    Handles three cases for object columns that contain numeric-looking values:

    CASE 1 — Purely numeric strings (e.g. "501", "5000"):
        → Convert the whole column to float directly.

    CASE 2 — Mixed: some rows are numeric strings, some are pure text
        (e.g. company_size has "501", "5000" AND "Big Tech", "Startup"):
        → Split into TWO columns:
              {col}_numeric  : extracted floats for numeric rows; text rows
                               receive unique continuation integers so the
                               column is fully filled and labels stay separable.
              {col}_category : text label for text rows, NaN elsewhere.
                               Immediately encoded with the best-fit strategy
                               (binary / one-hot / label / frequency) and
                               replaced by the resulting encoded column(s).
        The original column is dropped.

    CASE 3 — Numeric ratio below threshold:
        → Leave untouched; the Smart Encoding block handles it downstream.
    """

    # Column name patterns that must never be treated as numeric
    ID_LIKE = re.compile(r'(id|code|zip|phone|postal|pin|mobile)', re.IGNORECASE)

    # FIX: pandas 3.0 uses dtype "str" not "object"
    object_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]

    for col in object_cols:

        if ID_LIKE.search(col):
            print(f"⏭️  Skipping '{col}' — looks like an ID/code column")
            continue

        extracted = df[col].astype(str).str.extract(r'(\d+\.?\d*)')
        numeric_ratio = extracted.notnull().sum()[0] / len(df)

        # No numeric content at all → leave for Smart Encoding
        if numeric_ratio == 0:
            continue

        is_numeric_row = extracted[0].notnull()
        is_text_row    = ~is_numeric_row

        # ── CASE 1: Every row is a numeric string ─────────────────────────
        if is_text_row.sum() == 0:
            print(f"🔄 Converting '{col}' to numeric (all values numeric)")
            df[col] = extracted[0].astype(float)
            continue

        # ── CASE 2: Mixed numeric + text rows ─────────────────────────────
        print(f"\n⚡ Mixed column detected: '{col}'")
        print(f"   Splitting into '{col}_numeric' + '{col}_category' …")

        # --- numeric sub-column ------------------------------------------
        num_col = extracted[0].astype(float).copy()

        # --- text sub-column (NaN where the row was already numeric) ------
        text_col = df[col].where(is_text_row, other=np.nan)

        # Give every distinct text label its own unique integer so
        # {col}_numeric has zero NaNs and labels remain separable.
        unique_labels = text_col.dropna().unique()
        max_existing  = num_col.dropna().max() if num_col.notna().any() else 0

        label_to_num = {
            lbl: max_existing + (i + 1)
            for i, lbl in enumerate(unique_labels)
        }
        print(f"   Text-label → continuation-int mapping: {label_to_num}")

        for lbl, assigned in label_to_num.items():
            num_col[text_col == lbl] = assigned

        num_col_name  = f"{col}_numeric"
        text_col_name = f"{col}_category"

        # --- encode the category sub-column right now ---------------------
        print(f"   Encoding '{text_col_name}':")
        encoded_df = _encode_category_col(text_col, text_col_name, encoder_dir=encoder_dir)

        # --- insert everything at the original column position ------------
        insert_pos = df.columns.get_loc(col)

        # Drop original first, then re-insert so index arithmetic is stable
        df.drop(columns=[col], inplace=True)

        # Insert numeric column
        df.insert(insert_pos, num_col_name, num_col.values)

        # Insert encoded category column(s) immediately after numeric col
        for offset, enc_col in enumerate(encoded_df.columns, start=1):
            df.insert(insert_pos + offset, enc_col, encoded_df[enc_col].values)

        print(f"   ✅ '{col}' → '{num_col_name}' (float) + "
              f"{list(encoded_df.columns)} (encoded)")

    return df


# =====================================================
# AUTO IMPUTATION SELECTION
# =====================================================

def auto_select_best_imputer(df, numeric_cols, sample_fraction=0.1):

    print("🔍 Testing Imputation Methods...")

    df_copy = df[numeric_cols].copy()

    non_null_idx = df_copy.dropna().index.tolist()

    if len(non_null_idx) < 20:
        return "smart"

    sample_size = max(10, int(len(non_null_idx) * sample_fraction))

    sample_idx = random.sample(non_null_idx, sample_size)

    original_values = df_copy.loc[sample_idx].copy()

    df_copy.loc[sample_idx] = np.nan

    errors = {}

    temp = df_copy.copy()

    for col in numeric_cols:

        if abs(temp[col].skew()) < 1:
            temp[col].fillna(temp[col].mean(), inplace=True)
        else:
            temp[col].fillna(temp[col].median(), inplace=True)

    errors["smart"] = np.mean(np.abs(temp.loc[sample_idx] - original_values).values)

    try:
        knn = KNNImputer(n_neighbors=5)
        knn_data = knn.fit_transform(df_copy)
        knn_subset = knn_data[df_copy.index.get_indexer(sample_idx)]
        errors["knn"] = np.mean(np.abs(knn_subset - original_values.values))
    except:
        errors["knn"] = np.inf

    try:
        iterative = IterativeImputer(random_state=42)
        it_data = iterative.fit_transform(df_copy)
        it_subset = it_data[df_copy.index.get_indexer(sample_idx)]
        errors["iterative"] = np.mean(np.abs(it_subset - original_values.values))
    except:
        errors["iterative"] = np.inf

    best_method = min(errors, key=errors.get)

    # Normalise — map anything unexpected back to "smart"
    if best_method not in ("smart", "knn", "iterative"):
        best_method = "smart"

    print(f"   Best imputer selected: {best_method} "
          f"(errors → {', '.join(f'{k}: {v:.4f}' for k, v in errors.items())})")

    return best_method


# =====================================================
# SMOTE IMBALANCE HANDLER
# =====================================================

def apply_smote_if_needed(feature_df, target_df, target_columns):
    """
    Applies SMOTE per target column only when class imbalance exceeds the 60:40 ratio.

    Rules:
      - Single target  : if minority ratio < 40%  -> apply SMOTE
      - Multiple targets: per-column check;
                          if imbalance gap > 20%  -> apply SMOTE, else skip
      - Prints imbalance stats and SMOTE decisions to the terminal.
    """

    print("\n Checking Class Imbalance for SMOTE...")

    is_multi_target = len(target_columns) > 1

    X = feature_df.copy()
    combined_target = target_df.copy()

    smote_applied = False

    for col in target_columns:

        y = combined_target[col]

        value_counts = y.value_counts()
        total          = len(y)
        minority_count = value_counts.iloc[-1]
        majority_count = value_counts.iloc[0]
        minority_ratio = minority_count / total
        imbalance_gap  = majority_count / total - minority_ratio

        print(f"\n  Target: '{col}'")
        print(f"     Class distribution : {value_counts.to_dict()}")
        print(f"     Minority ratio      : {minority_ratio * 100:.1f}%")
        print(f"     Majority ratio      : {majority_count / total * 100:.1f}%")

        if is_multi_target:
            threshold_met = imbalance_gap > 0.20
            rule_desc = "multi-target rule (gap > 20% -> apply SMOTE)"
        else:
            threshold_met = minority_ratio < 0.40
            rule_desc = "single-target rule (minority < 40% -> apply SMOTE)"

        if not threshold_met:
            print(f"     Imbalance within acceptable range - SMOTE skipped ({rule_desc})")
            continue

        print(f"     Imbalance detected - Applying SMOTE ({rule_desc})")

        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)

            after_counts = pd.Series(y_res).value_counts().to_dict()
            print(f"     SMOTE complete. New distribution: {after_counts}")

            X = pd.DataFrame(X_res, columns=X.columns)

            new_idx = pd.RangeIndex(len(X_res))
            new_target = pd.DataFrame(index=new_idx, columns=combined_target.columns)

            for other_col in combined_target.columns:
                if other_col == col:
                    new_target[other_col] = y_res.values if hasattr(y_res, "values") else y_res
                else:
                    orig_vals = combined_target[other_col].values
                    pad_size  = len(X_res) - len(orig_vals)
                    mode_val  = combined_target[other_col].mode()[0]
                    new_target[other_col] = np.concatenate([orig_vals, np.full(pad_size, mode_val)])

            combined_target = new_target
            smote_applied = True

        except Exception as e:
            print(f"     SMOTE failed for '{col}': {e} - skipping")

    if not smote_applied:
        print("\n  No SMOTE applied - all targets are within balance threshold.")

    return X, combined_target


# =====================================================
# OUTLIER DETECTION AND REMOVAL
# =====================================================

def handle_outliers(df, method="none", contamination=0.05):
    """
    Detects and removes outlier rows from a fully-numeric DataFrame.

    Supported methods
    -----------------
    "none"             : No outlier removal — returns df unchanged.

    "iqr"              : Inter-Quartile Range rule.
                         A row is an outlier if ANY numeric column has a value
                         outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
                         Robust to skewed distributions; good default for
                         tabular data with well-behaved numeric columns.

    "zscore"           : Standard Z-score rule.
                         A row is an outlier if ANY numeric column has
                         |z| > 3  (i.e. more than 3 standard deviations
                         from the mean).
                         Works best when columns are roughly normal.

    "isolation_forest" : Isolation Forest (sklearn).
                         Fits an ensemble of random trees and isolates
                         anomalies by how quickly they can be separated.
                         Handles high-dimensional data well, captures
                         multivariate outliers that IQR/Z-score miss, and
                         does not assume any particular distribution.
                         contamination controls the expected fraction of
                         outliers (default 0.05 = 5%).

    Parameters
    ----------
    df            : pd.DataFrame  — should contain only numeric columns.
    method        : str           — one of "none", "iqr", "zscore",
                                    "isolation_forest".
    contamination : float         — fraction of outliers expected in the
                                    data; used only by isolation_forest.
                                    Must be in (0, 0.5].

    Returns
    -------
    Cleaned pd.DataFrame with outlier rows dropped and index reset.
    """

    method = method.strip().lower()

    if method == "none":
        print("\n skipping Outlier removal (method='none')")
        return df

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        print("\n No numeric columns found — skipping outlier removal")
        return df

    rows_before  = len(df)
    outlier_mask = pd.Series(False, index=df.index)   # True = outlier row

    # IQR method
    if method == "iqr":
        print("\n Outlier Detection: IQR method")

        for col in numeric_cols:
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue   # constant column — no outliers possible

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_mask |= (df[col] < lower) | (df[col] > upper)

        print("   Bounds: Q1 - 1.5xIQR  /  Q3 + 1.5xIQR  per column")

    # Z-score method
    elif method == "zscore":
        print("\n Outlier Detection: Z-score method  (threshold = 3 sigma)")

        for col in numeric_cols:
            std = df[col].std()

            if std == 0:
                continue   # constant column — skip

            z_scores = (df[col] - df[col].mean()) / std
            outlier_mask |= z_scores.abs() > 3

        print("   Threshold: |z| > 3 standard deviations")

    # Isolation Forest
    elif method == "isolation_forest":
        print(f"\n Outlier Detection: Isolation Forest  (contamination={contamination})")

        clf = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

        # Fill any residual NaNs with column median before fitting
        X = df[numeric_cols].copy()
        for col in numeric_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        preds = clf.fit_predict(X)          # +1 = inlier, -1 = outlier
        outlier_mask = pd.Series(preds == -1, index=df.index)

        print("   Model: IsolationForest | n_estimators=100 | random_state=42")

    else:
        print(f"\n Unknown outlier method '{method}' — skipping. "
              f"Valid options: 'none', 'iqr', 'zscore', 'isolation_forest'")
        return df

    # Remove detected outliers
    rows_removed = int(outlier_mask.sum())
    df_clean = df[~outlier_mask].reset_index(drop=True)

    print(f"   Rows before : {rows_before}")
    print(f"   Outliers    : {rows_removed}  ({rows_removed / rows_before * 100:.1f}%)")
    print(f"   Rows after  : {len(df_clean)}")

    return df_clean


# =====================================================
# MAIN PREPROCESS FUNCTION
# =====================================================

def label_encode_large_dataset(
    input_csv,
    target_columns=None,
    output_csv=None,
    encoder_dir=None,
    cat_threshold=0.15,
    outlier_method="none",
    outlier_contamination=0.05,
    problem_type="Regression"
):

    print("📥 Loading dataset...")

    df = pd.read_csv(input_csv)

    # --------------------------------------------------
    # FIX 1: Normalize column names — strip whitespace
    # from both the dataframe columns AND the user-supplied
    # target_columns list so typos like "risk _category"
    # still match "risk_category" in the CSV.
    # --------------------------------------------------
    df.columns = df.columns.str.strip()

    if target_columns is None:
        target_columns = []

    # Strip whitespace from user-supplied target column names
    target_columns = [col.strip() for col in target_columns]

    # --------------------------------------------------
    # FIX 2: Validate that target columns actually exist
    # in the dataframe and raise a clear error immediately
    # instead of silently skipping encoding later.
    # --------------------------------------------------
    missing = [col for col in target_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Target column(s) {missing} not found in CSV.\n"
            f"   Available columns: {df.columns.tolist()}"
        )

    base_name = os.path.splitext(os.path.basename(input_csv))[0]

    if output_csv is None:
        output_csv = f"{base_name}_processed.csv"

    if encoder_dir is None:
        encoder_dir = f"{base_name}_encoders"

    os.makedirs(encoder_dir, exist_ok=True)

    target_df = df[target_columns].copy() if target_columns else pd.DataFrame()
    feature_df = df.drop(columns=target_columns, errors="ignore")

    # =====================================================
    # TARGET ENCODING
    # =====================================================

    if len(target_columns) > 0:

        if problem_type.lower() == "classification":

            print("\n🎯 Encoding Target Column(s) for Classification")

            for col in target_columns:

                # --------------------------------------------------
                # FIX 3: Use pd.api.types.is_numeric_dtype instead
                # of dtype == "object" — covers string, category,
                # and any other non-numeric pandas dtype reliably.
                # --------------------------------------------------
                if not pd.api.types.is_numeric_dtype(target_df[col]):

                    print(f"  {col} → Label Encoded (object → numeric)")

                    le = LabelEncoder()

                    target_df[col] = le.fit_transform(target_df[col].astype(str))

                    print(f"  Class mapping: { {cls: idx for idx, cls in enumerate(le.classes_)} }")

                    with open(os.path.join(encoder_dir, f"{col}_target_encoder.pkl"), "wb") as f:
                        pickle.dump(le, f)

                else:

                    print(f"  {col} already numeric → skipping")

        else:  # Regression — leave target completely untouched

            print("\n🎯 Regression task detected → Target column(s) left untouched")

            for col in target_columns:
                print(f"  {col} → unchanged (dtype: {target_df[col].dtype})")

    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    feature_df = detect_and_engineer_dates(feature_df)

    # --------------------------------------------------
    # FIX 5: Drop rows where ALL engineered date columns are NaN.
    # These rows had unparseable / missing timestamps and carry no
    # temporal context — keeping them adds noise without signal.
    # Only runs if at least one date column was actually created.
    # --------------------------------------------------
    date_eng_cols = [c for c in feature_df.columns
                     if c.endswith(("_year", "_month", "_day", "_time"))]
    if date_eng_cols:
        rows_before = len(feature_df)
        nat_mask    = feature_df[date_eng_cols].isna().all(axis=1)
        if nat_mask.any():
            feature_df = feature_df[~nat_mask].reset_index(drop=True)
            # Keep target_df aligned with feature_df
            if len(target_df) > 0:
                target_df = target_df[~nat_mask].reset_index(drop=True)
            rows_dropped = rows_before - len(feature_df)
            print(f"\n🗑  Dropped {rows_dropped} rows with no parseable date "
                  f"({rows_dropped/rows_before*100:.1f}% of data)")
        else:
            print("\n✅ No NaT-only date rows found — no rows dropped")

    # --------------------------------------------------
    # FIX 6: Drop zero-variance date columns produced by date engineering.
    # A date component that is constant across all rows (e.g. time_year
    # when the dataset covers only one year) becomes all-zeros after
    # MinMaxScaling — it contributes no information and can mislead models.
    # --------------------------------------------------
    zero_var_date_cols = [c for c in date_eng_cols
                          if c in feature_df.columns
                          and feature_df[c].nunique(dropna=True) <= 1]
    if zero_var_date_cols:
        feature_df.drop(columns=zero_var_date_cols, inplace=True)
        print(f"🗑  Dropped zero-variance date column(s): {zero_var_date_cols}")

    feature_df = detect_and_convert_numeric_text(feature_df, encoder_dir=encoder_dir)

    # --------------------------------------------------
    # FIX 4: Refresh column lists AFTER feature engineering
    # since new columns may have been added or removed.
    # --------------------------------------------------
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # FIX: pandas 3.0 uses dtype "str" not "object"
    categorical_cols = [c for c in feature_df.columns if pd.api.types.is_string_dtype(feature_df[c])]

    # =====================================================
    # SMART ENCODING
    # =====================================================

    # Note: columns produced by detect_and_convert_numeric_text are already
    # encoded inline, so they will no longer appear in categorical_cols here.
    print("\n🔤 Smart Encoding Categorical Columns")

    # Track label-encoded columns so they are excluded from MinMax scaling.
    # Label-encoded integers have no ordinal meaning — scaling them would
    # imply a numeric relationship between categories that doesn't exist.
    label_encoded_cols = []

    for col in categorical_cols:

        feature_df[col] = feature_df[col].astype(str)

        freq = feature_df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        feature_df[col] = feature_df[col].replace(rare, "Other")

        unique_vals = feature_df[col].nunique()

        if unique_vals == 2:

            print(f"{col} → Binary Encoding")

            le = LabelEncoder()
            feature_df[col] = le.fit_transform(feature_df[col])
            print(f"   Label mapping: { {cls: idx for idx, cls in enumerate(le.classes_)} }")

        elif unique_vals <= 10:

            print(f"{col} → One Hot Encoding")

            dummies = pd.get_dummies(feature_df[col], prefix=col, dtype=int)

            feature_df = pd.concat([feature_df.drop(columns=[col]), dummies], axis=1)

        elif unique_vals <= 50:

            print(f"{col} → Label Encoding")

            le = LabelEncoder()

            feature_df[col] = le.fit_transform(feature_df[col])

            print(f"   Label mapping: { {cls: idx for idx, cls in enumerate(le.classes_)} }")

            label_encoded_cols.append(col)  # track to exclude from scaling

            with open(os.path.join(encoder_dir, f"{col}_encoder.pkl"), "wb") as f:
                pickle.dump(le, f)

        else:

            print(f"{col} → Frequency Encoding")

            freq = feature_df[col].value_counts(normalize=True)

            feature_df[col] = feature_df[col].map(freq)

    # =====================================================
    # SMART IMPUTATION
    # =====================================================

    # Refresh numeric cols after encoding — all cols are numeric now
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if feature_df[numeric_cols].isnull().any().any():

        print("\n🩹 Missing values detected — running Smart Imputation...")

        best_method = auto_select_best_imputer(feature_df, numeric_cols)

        if best_method in ("smart", "median", "mean"):
            for col in numeric_cols:
                if feature_df[col].isnull().any():
                    if abs(feature_df[col].skew()) < 1:
                        feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
                    else:
                        feature_df[col] = feature_df[col].fillna(feature_df[col].median())

        elif best_method == "knn":
            knn = KNNImputer(n_neighbors=5)
            feature_df[numeric_cols] = knn.fit_transform(feature_df[numeric_cols])

        elif best_method == "iterative":
            iterative = IterativeImputer(random_state=42)
            feature_df[numeric_cols] = iterative.fit_transform(feature_df[numeric_cols])

        print(f"   ✅ Imputation done using: {best_method}")

    else:
        print("\n✅ No missing values found — skipping imputation")

    # =====================================================
    # OUTLIER REMOVAL
    # =====================================================

    if outlier_method != "none":
        print(f"\n🚨 Outlier Removal (method='{outlier_method}')")
        # Re-attach target so row indices stay aligned after dropping outliers
        combined = pd.concat([feature_df, target_df], axis=1)
        combined = handle_outliers(combined, method=outlier_method, contamination=outlier_contamination)
        feature_df = combined[feature_df.columns].reset_index(drop=True)
        if len(target_columns) > 0:
            target_df = combined[target_columns].reset_index(drop=True)
    else:
        print("\n⏭️  Outlier removal skipped (method='none')")

    # =====================================================
    # SCALING
    # =====================================================

    # Refresh numeric_cols again after imputation
    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # FIX 5 — Scale low-cardinality numeric columns (e.g. latitude, longitude)
    # The old guard `nunique() > 20` incorrectly skipped legitimate numeric
    # features that happen to have few unique values (e.g. coordinate grids).
    # Option B: skip only true binary 0/1 columns by checking actual values,
    # regardless of cardinality. nunique() > 2 filters out constant/binary
    # columns; the set check ensures only exact {0,1} columns are excluded.
    scale_cols = [
        c for c in numeric_cols
        if feature_df[c].nunique() > 2
        and set(feature_df[c].dropna().unique()) != {0, 1}
        and c not in label_encoded_cols  # skip label-encoded categoricals
    ]

    if scale_cols:

        scaler = MinMaxScaler()

        feature_df[scale_cols] = scaler.fit_transform(feature_df[scale_cols])

        with open(os.path.join(encoder_dir, "minmax_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        print(f"\n📊 Scaled columns: {scale_cols}")

    # =====================================================
    # SMOTE - CLASSIFICATION ONLY
    # =====================================================

    if problem_type.lower() == "classification" and len(target_columns) > 0:
        feature_df, target_df = apply_smote_if_needed(feature_df, target_df, target_columns)

    final_df = pd.concat([feature_df, target_df], axis=1)

    final_df.to_csv(output_csv, index=False)

    print("\n✅ Preprocessing Completed")
    print(f"   Output saved to  : {output_csv}")
    print(f"   Encoders saved to: {encoder_dir}/")

    return output_csv