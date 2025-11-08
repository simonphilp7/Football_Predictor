import pandas as pd
from utils.model_building.model_build import extract_X_and_Y
from utils.model_building.model_params import betting_company_maps, prediction_maps, division_maps

def get_betting_vals_and_preds_no_calibration(test_df, cols_to_drop, best_model,
                                              use_probs=False, include_odds=False):
    betting_cols = ['MAX_BETH', 'MAX_BETD', 'MAX_BETA']
    clean_test_df = test_df[~test_df['MAX_BETH'].isna()]
    new_cols_to_drop = [col for col in cols_to_drop if col not in betting_cols]
    X_test, y_test = extract_X_and_Y(clean_test_df, new_cols_to_drop, include_odds=include_odds)
    betting_vals = X_test[betting_cols]
    betting_vals_lst = [(row.MAX_BETH, row.MAX_BETD, row.MAX_BETA) for row in betting_vals.itertuples()]
    X_test = X_test.drop(betting_cols, axis=1)
    if use_probs:
        y_preds = best_model.predict_proba(X_test)
    else:
        y_preds = best_model.predict(X_test)
    return betting_vals_lst, list(y_preds), list(y_test)


def get_betting_vals_and_preds_simple(test_df, cols_to_drop, model, include_odds=False, X_only=False):
    betting_cols = ['MAX_BETH', 'MAX_BETD', 'MAX_BETA']
    clean_test_df = test_df[~test_df['MAX_BETH'].isna()]
    new_cols_to_drop = [col for col in cols_to_drop if col not in betting_cols]
    if X_only:
        X_test = extract_X_and_Y(clean_test_df, new_cols_to_drop, include_odds=include_odds, X_only=True)
        y_test = None
    else:
        X_test, y_test = extract_X_and_Y(clean_test_df, new_cols_to_drop, include_odds=include_odds)
    betting_vals = X_test[betting_cols]
    betting_vals_lst = [(row.MAX_BETH, row.MAX_BETD, row.MAX_BETA) for row in betting_vals.itertuples()]
    X_test = X_test.drop(betting_cols, axis=1)
    y_prob_preds = model.predict_proba(X_test)
    y_preds = model.predict(X_test)
    return betting_vals_lst, list(y_prob_preds), list(y_preds), y_test

def profit_of_prediction(odds, prediction, actual):
    if prediction != actual:
        return -1
    else:
        if prediction == 0:
            return odds[2] - 1
        elif prediction == 1:
            return odds[1] - 1
        else:
            return odds[0] - 1


def calculate_ev(odds, prediction, prob):
    if prediction == 0:
        prediction_odds = odds[2]
    elif prediction == 1:
        prediction_odds = odds[1]
    else:
        prediction_odds = odds[0]
    ev = (prob * prediction_odds) - 1
    return ev


def profit_of_probability_predictions(odds, predictions, actual, acceptable_ev):
    max_prob = max(predictions)
    prediction = list(predictions).index(max_prob)
    if calculate_ev(odds, prediction, max_prob) >= acceptable_ev:
        return profit_of_prediction(odds, prediction, actual)
    else:
        return None


def calculate_evs(odds, predictions):
    max_prob = max(predictions)
    prediction = list(predictions).index(max_prob)
    return calculate_ev(odds, prediction, max_prob)


def reformat_y(number):
    if number == 0:
        return 'A'
    elif number == 1:
        return 'D'
    else:
        return 'H'


def extract_best_odds_and_company_of_prediction(row):
    prediction = row['Prediction']
    if prediction == 'H':
        return pd.Series([row['MAX_BETH'], betting_company_maps[row['MAX_BETH_COMP'][:-1]]])
    elif prediction == 'D':
        return pd.Series([row['MAX_BETD'], betting_company_maps[row['MAX_BETD_COMP'][:-1]]])
    else:
        return pd.Series([row['MAX_BETA'], betting_company_maps[row['MAX_BETA_COMP'][:-1]]])


def get_ev_of_predictions(test_df, model, cols_to_drop, include_odds=False):
    betting_vals_lst, y_prob_preds, y_preds, y_test = get_betting_vals_and_preds_simple(test_df, cols_to_drop, model,
                                                                                X_only=True,
                                                                                include_odds=include_odds)
    test_df.dropna(inplace=True)
    formatted_y_preds = [reformat_y(y_pred) for y_pred in y_preds]
    evs = [calculate_evs(odds, predictions) for odds, predictions in zip(betting_vals_lst, y_prob_preds)]
    test_df['Prediction'] = formatted_y_preds
    test_df['Prediction_EV'] = evs
    test_df[['Prediction_Odds', 'Prediction_Company']] = test_df.apply(
        lambda row: extract_best_odds_and_company_of_prediction(row), axis=1)
    test_df['div'] = test_df['div'].map(division_maps)
    test_df['Prediction'] = test_df['Prediction'].map(prediction_maps)
    output_df = test_df[['date', 'div', 'home_team', 'away_team', 'Prediction', 'Prediction_EV', 'Prediction_Odds',
                         'Prediction_Company']]
    sorted_output_df = output_df.sort_values(by='Prediction_EV', ascending=False)
    return sorted_output_df


def get_average_profit_of_past_predictions(test_df, cols_to_drop, model, acceptable_ev=0, top_n_evs=False, n_evs=None,
                                     use_probs=False, include_odds=False):
    betting_vals_lst, y_prob_preds, y_preds, y_test = get_betting_vals_and_preds_simple(test_df, cols_to_drop, model,
                                                                                include_odds=include_odds)
    if use_probs:
        if top_n_evs:
            evs = [calculate_evs(odds, predictions) for odds, predictions in zip(betting_vals_lst, y_prob_preds)]
            evs.sort(reverse=True)
            acceptable_ev = evs[n_evs - 1]

        initial_profits_lst = [profit_of_probability_predictions(odds, predictions, actual, acceptable_ev) for
                               odds, predictions, actual in
                               zip(betting_vals_lst, y_prob_preds, list(y_test))]
        profits_lst = list(filter(lambda x: x is not None, initial_profits_lst))
    else:
        profits_lst = [profit_of_prediction(odds, prediction, actual) for odds, prediction, actual in
                       zip(betting_vals_lst, y_preds, list(y_test))]
    no_of_bets = len(profits_lst)
    return (sum(profits_lst) / len(profits_lst)), no_of_bets

def get_overall_profit_of_past_predictions(test_df, cols_to_drop, model, acceptable_ev=0,use_probs=False,include_odds=False, stake=1):
    average_profit, no_of_bets = get_average_profit_of_past_predictions(test_df, cols_to_drop, model, acceptable_ev=acceptable_ev,
                                                        use_probs=use_probs, include_odds=include_odds)
    print(no_of_bets)
    total_profit = no_of_bets*average_profit*stake
    return total_profit
