"""Machine learning model building utilities for training and hyperparameter tuning."""

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def load_in_df(filepath):
    """Loads a CSV file into a DataFrame with date parsing."""
    full_df = pd.read_csv(filepath)
    full_df["date"] = pd.to_datetime(full_df["date"])
    return full_df


def extract_train_and_test(df, train_split=0.8):
    """Splits DataFrame into training and test sets based on date cutoff."""
    df_length = len(df)
    train_length = int(df_length * (train_split))
    date_cutoff = df["date"].iloc[-train_length]
    train_df = df[df["date"] <= date_cutoff]
    test_df = df[df["date"] > date_cutoff]
    return train_df, test_df


def extract_X_and_Y(full_df, cols_to_drop, include_odds=False, date=None, X_only=False):
    """Extracts feature matrix X and encoded target y from DataFrame."""
    if date is not None:
        subset_df = full_df[full_df["date"] > date]
    else:
        subset_df = full_df.copy()
    if not include_odds:
        cols_to_drop += ["AVG_BETH", "AVG_BETD", "AVG_BETA"]
    processed_cols_to_drop = [col_name for col_name in cols_to_drop if col_name in list(subset_df.columns)]
    subset_df = subset_df.drop(processed_cols_to_drop, axis=1)
    subset_df.dropna(inplace=True)
    if X_only:
        return subset_df
    y = subset_df.pop("FTR")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return subset_df, y_encoded


def find_cat_and_numerical_cols(X):
    """Identifies categorical and numerical columns in feature matrix."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.to_list()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.to_list()
    return num_cols, cat_cols


def create_pipeline(preprocessor, model):
    """Creates a scikit-learn pipeline with preprocessing and model steps."""
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline


def fit_model(X_train, y_train, X_val, y_val, pipeline):
    """Fits a pipeline model and returns predictions comparison DataFrame."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    comparison_df = pd.DataFrame({"Actual": y_val, "Predicted": preds})
    print(classification_report(y_val, preds))
    return comparison_df


def create_model_from_pipeline(preprocessor, model, X_train, y_train, X_val, y_val):
    """Creates and fits a complete pipeline model with validation."""
    pipeline = create_pipeline(preprocessor, model)
    df = fit_model(X_train, y_train, X_val, y_val, pipeline)
    return df


def fit_model_from_df(train_df, cols_to_drop, model, val_size=0.2, include_odds=False):
    """Fits a model directly from DataFrame with automatic preprocessing."""
    X_train, y_train = extract_X_and_Y(train_df, cols_to_drop, include_odds=include_odds)
    num_cols, cat_cols = find_cat_and_numerical_cols(X_train)
    training_size = len(X_train)
    index_cut_off = int(training_size * (1 - val_size))
    X_train_act = X_train[:index_cut_off]
    y_train_act = y_train[:index_cut_off]
    X_val_act = X_train[index_cut_off:]
    y_val_act = y_train[index_cut_off:]
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )
    df = create_model_from_pipeline(preprocessor, model, X_train_act, y_train_act, X_val_act, y_val_act)
    return df


def grid_search_from_df(train_df, test_df, cols_to_drop, model, param_grid, n_splits, include_odds=False):
    """Performs grid search with time series cross-validation on DataFrame."""
    X_train, y_train = extract_X_and_Y(train_df, cols_to_drop, include_odds=include_odds)
    X_test, y_test = extract_X_and_Y(test_df, cols_to_drop, include_odds=include_odds)
    num_cols, cat_cols = find_cat_and_numerical_cols(X_train)
    preprocessor = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)]
    )
    pipeline = create_pipeline(preprocessor, model)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",  # or 'f1_macro' for class balance
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)
    preds = grid_search.predict(X_test)
    print(classification_report(y_test, preds))
    accuracy = accuracy_score(y_test, preds)
    return grid_search, accuracy, pipeline


def find_best_hyperparams_and_features(
    train_df, test_df, lst_of_cols_to_drop, model, param_grid, n_splits, include_odds=False
):
    """Finds best hyperparameters and feature combinations via grid search."""
    results_dic = {}
    for idx, cols_to_drop in enumerate(lst_of_cols_to_drop):
        grid_search, accuracy, pipeline = grid_search_from_df(
            train_df, test_df, cols_to_drop, model, param_grid, n_splits, include_odds=include_odds
        )
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        results_dic[f"Dropped Cols {idx}"] = (best_model, best_params, accuracy)
    return results_dic


def find_best_model_and_further_info(train_df, test_df, lst_of_cols_to_drop, model_dic, n_splits, include_odds=False):
    """Evaluates multiple models and returns best performing configuration."""
    best_results_dic = {}
    for model_name, model_info in model_dic.items():
        model = model_info[0]
        param_grid = model_info[1]
        results_dic = find_best_hyperparams_and_features(
            train_df, test_df, lst_of_cols_to_drop, model, param_grid, n_splits, include_odds=include_odds
        )
        results_lst = list(results_dic.items())
        results_lst.sort(key=lambda x: x[1][2])
        best_result = results_lst[-1]
        best_results_dic[f"{model_name} {best_result[0]}"] = best_result[1]
    return best_results_dic
