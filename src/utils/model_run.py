import joblib

from utils.model_building.model_build import (
    extract_train_and_test,
    find_best_hyperparams_and_features,
    find_best_model_and_further_info,
    grid_search_from_df,
)
from utils.model_building.model_evaluation import get_ev_of_predictions, get_overall_profit_of_past_predictions
from utils.model_building.model_params import *
from utils.overall_data_process import extract_data_for_model

countries = ["England"]
train_seasons = [f"{year}_{year + 1}" for year in range(15, 26)]
test_seasons = ["24_25", "25_26"]
needed_cols = [
    "Div",
    "Div_Name",
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HR",
    "AR",
    "AVG_BETH",
    "AVG_BETD",
    "AVG_BETA",
    "MAX_BETH",
    "MAX_BETD",
    "MAX_BETA",
    "MAX_BETH_COMP",
    "MAX_BETD_COMP",
    "MAX_BETA_COMP",
]
country_divisions = {"England": ["E0", "E1", "E2", "E3", "E4", "EC"]}

## Train Model
######################################################################################################################
clean_past_df = extract_data_for_model(countries, train_seasons, needed_cols, country_divisions, data="Train")
train_df, test_df = extract_train_and_test(clean_past_df)
results_dic = find_best_model_and_further_info(train_df, test_df, lst_of_cols_to_drop, model_dic, 4)

grid, accuracy, pipeline = grid_search_from_df(
    train_df, test_df, lvl1_unneeded_cols, xgb_model, xgb_param_grid, 4, include_odds=True
)

# results_dic = find_best_hyperparams_and_features(train_df, test_df, lst_of_cols_to_drop, xgb_model, xgb_param_grid, 4,
#                                                include_odds=True)

best_model = grid.best_estimator_

joblib.dump(best_model, "data/best_model.pkl")

best_model_params = grid.best_params_

## Evaluate on Upcoming Matches
#########################################################################################################################
best_model = joblib.load("data/best_model.pkl")

clean_upcoming_df = extract_data_for_model(countries, test_seasons, needed_cols, country_divisions)
preds_train_df, na_test_df = extract_train_and_test(clean_upcoming_df, train_split=1)
df = get_ev_of_predictions(preds_train_df, best_model, lvl1_unneeded_cols, include_odds=True)

## Evaluate on Previous matches
#########################################################################################################################
best_model = joblib.load("data/best_model.pkl")
countries = ["England"]
seasons = ["24_25", "25_26"]
clean_past_df = extract_data_for_model(countries, seasons, needed_cols, country_divisions, data="Train")
ev_df = get_ev_of_predictions(clean_past_df, best_model, lvl1_unneeded_cols, include_odds=True)
ev_df.head()
get_overall_profit_of_past_predictions(
    clean_past_df, lvl1_unneeded_cols, best_model, acceptable_ev=0.05, use_probs=True, include_odds=True, stake=10
)
get_overall_profit_of_past_predictions(clean_past_df, lvl1_unneeded_cols, best_model, include_odds=True, stake=10)
