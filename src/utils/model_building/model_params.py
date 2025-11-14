from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

betting_company_maps = {
    "B365": "Bet365",
    "BF": "Betfair",
    "1XB": "1XBet",
    "BW": "Bet&Win",
    "WH": "William Hill",
    "BV": "Betvictor",
    "CL": "Coral",
    "LB": "Ladbrokes",
    "BFD": "Betfred",
    "BMGM": "BetMGM",
    "VC": "VC Bet",
    "IW": "Interwetten",
}
division_maps = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
}
prediction_maps = {"H": "Home Win", "D": "Draw", "A": "Away Win"}

lvl1_unneeded_cols = [
    "date",
    "home_team",
    "away_team",
    "season",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HR",
    "AR",
    "MAX_BETH",
    "MAX_BETD",
    "MAX_BETA",
    "MAX_BETH_COMP",
    "MAX_BETD_COMP",
    "MAX_BETA_COMP",
]
lvl2_unneeded_cols = lvl1_unneeded_cols + ["HT_RC_1", "HT_RD", "AT_RD", "AT_RC_1"]
lvl3_unneeded_cols = lvl2_unneeded_cols + [
    "HT_AW%_5",
    "HT_AL%_5",
    "HT_AAGS_5",
    "HT_AAGC_5",
    "AT_HW%_5",
    "AT_HL%_5",
    "AT_HAGS_5",
    "AT_HAGC_5",
]
lvl4_unneeded_cols = lvl3_unneeded_cols + ["HT_OAGS_20", "HT_OAGC_20", "AT_OAGS_20", "AT_OAGC_20"]
lvl5_unneeded_cols = lvl4_unneeded_cols + ["HT_HAGS_5", "HT_HAGC_5", "AT_AAGS_5", "AT_AAGC_5"]
lvl6_unneeded_cols = lvl1_unneeded_cols + ["HT_OAGS_20", "HT_OAGC_20", "AT_OAGS_20", "AT_OAGC_20"]
lvl7_unneeded_cols = lvl5_unneeded_cols + [
    "HT_LC_20",
    "HT_OW%_20",
    "HT_OL%_20",
    "HT_HW%_5",
    "HT_HL%_5",
    "AT_LC_20",
    "AT_OW%_20",
    "AT_OL%_20",
    "AT_AW%_5",
    "AT_AL%_5",
]

lst_of_cols_to_drop = [
    lvl1_unneeded_cols,
    lvl2_unneeded_cols,
    lvl3_unneeded_cols,
    lvl4_unneeded_cols,
    lvl5_unneeded_cols,
    lvl6_unneeded_cols,
    lvl7_unneeded_cols,
]

rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

rf_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__max_depth": [5, 10, 20, None],
    "model__max_features": ["sqrt", "log2"],
}
xgb_model = XGBClassifier(
    eval_metric="mlogloss",  # appropriate for multi-class
    random_state=42,
    tree_method="hist",  # 'gpu_hist' if you have GPU
    n_jobs=-1,  # use all CPU cores
)
xgb_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [3, 5, 7, 9],
}
lgb_model = LGBMClassifier(objective="multiclass", random_state=42, n_jobs=-1)

lgb_param_grid = {
    "model__n_estimators": [200, 500, 800],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__max_depth": [-1, 5, 10],  # -1 = no limit
    "model__num_leaves": [31, 63, 127],
}

cat_model = CatBoostClassifier(loss_function="MultiClass", random_seed=42)

cat_param_grid = {
    "model__iterations": [300, 600, 900],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__depth": [4, 6, 8],
}

model_dic = {
    "xgb": (xgb_model, xgb_param_grid),
    "rf": (rf_model, rf_param_grid),
    "lgb": (lgb_model, lgb_param_grid),
    "catboost": (cat_model, cat_param_grid),
}
