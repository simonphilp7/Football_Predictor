import math
import pandas as pd
from datetime import datetime
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def create_correct_bet_column_groups(df):
    with open(r"C:\Repos\Misc\Football_Predictor\data\odds_columns.txt", 'r') as f:
        txt = f.readlines()
    bet_columns = [line.split('=')[0].strip() for line in txt]
    correct_bet_columns = [col_name for col_name in bet_columns if col_name in df.columns]
    correct_bet_columns.sort(key=lambda col_name: df[col_name].isna().sum())
    correct_bet_column_groups = [[correct_bet_columns[idx], correct_bet_columns[idx + 1], correct_bet_columns[idx + 2]]
                                 for idx in range(0, len(correct_bet_columns), 3)]
    return correct_bet_column_groups

def extract_max_and_avg_odds(lst_of_odds):
    if not (lst_of_odds):
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    non_empty_odds = [odds for odds in lst_of_odds if not (math.isnan(odds[0][0]))]
    avg_home_odds = round(sum([odds[0][0] for odds in non_empty_odds]) / len(non_empty_odds), 2)
    avg_draw_odds = round(sum([odds[0][1] for odds in non_empty_odds]) / len(non_empty_odds), 2)
    avg_away_odds = round(sum([odds[0][2] for odds in non_empty_odds]) / len(non_empty_odds), 2)
    max_home_odds = max([odds[0][0] for odds in non_empty_odds])
    max_draw_odds = round(max([odds[0][1] for odds in non_empty_odds]), 2)
    max_away_odds = round(max([odds[0][2] for odds in non_empty_odds]), 2)
    max_home_odds_comp = non_empty_odds[[odds[0][0] for odds in non_empty_odds].index(max_home_odds)][1][0]
    max_draw_odds_comp = non_empty_odds[[odds[0][1] for odds in non_empty_odds].index(max_draw_odds)][1][1]
    max_away_odds_comp = non_empty_odds[[odds[0][2] for odds in non_empty_odds].index(max_away_odds)][1][2]
    return pd.Series(
        [avg_home_odds, avg_draw_odds, avg_away_odds, round(max_home_odds, 2), round(max_draw_odds, 2),
         round(max_away_odds, 2),
         max_home_odds_comp, max_draw_odds_comp, max_away_odds_comp])


def find_odds(row, bet_column_groups):
    lst_of_odds = []
    for bet_column_group in bet_column_groups:
        try:
            home_odds = row[bet_column_group[0]]
            draw_odds = row[bet_column_group[1]]
            loss_odds = row[bet_column_group[2]]
            lst_of_odds.append(((float(home_odds), float(draw_odds), float(loss_odds)), bet_column_group))
        except Exception:
            pass
    try:
        return extract_max_and_avg_odds(lst_of_odds)
    except Exception:
        return pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def add_odds(row, bet_column_groups):
    try:
        raise (Exception)
        avg_home_odds = row['AvgH']
        avg_draw_odds = row['AvgD']
        avg_away_odds = row['AvgA']
        max_home_odds = row['MaxH']
        max_draw_odds = row['MaxD']
        max_away_odds = row['MaxA']
        series = pd.Series([avg_home_odds, avg_draw_odds, avg_away_odds, max_home_odds, max_draw_odds, max_away_odds])
        if series.isna().any():
            return find_odds(row, bet_column_groups)
        else:
            return pd.Series([avg_home_odds, avg_draw_odds, avg_away_odds, max_home_odds, max_draw_odds, max_away_odds])
    except Exception:
        return find_odds(row, bet_column_groups)


def parse_date(date_str):
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):  # try 4-digit and 2-digit year
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return pd.NaT


def data_cleaning(df, test_data=False):
    df['Date'] = df['Date'].apply(parse_date)
    if test_data:
        return df
    int_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HR', 'AR']
    for col in int_cols:
        df[col] = df[col].astype("Int64")
    return df


def clean_data(path, needed_cols, export=False, export_filepath=None, df=None, test_data=False):
    if df is None:
        df = pd.read_csv(path)
    new_df = df.copy()
    groups = create_correct_bet_column_groups(new_df)
    new_df[['AVG_BETH', 'AVG_BETD', 'AVG_BETA', 'MAX_BETH', 'MAX_BETD', 'MAX_BETA', 'MAX_BETH_COMP', 'MAX_BETD_COMP',
            'MAX_BETA_COMP']] = new_df.apply(
        lambda row: add_odds(row, groups), axis=1)
    possessed_needed_cols = [col_name for col_name in needed_cols if col_name in list(new_df.columns)]
    full_sample_df = new_df[possessed_needed_cols]
    full_sample_df = data_cleaning(full_sample_df, test_data=test_data)
    full_sample_df.sort_values(by='Date', ascending=False, inplace=True)
    if export:
        full_sample_df.to_csv(export_filepath, index=False)
    return full_sample_df


path_to_english = r"C:\Repos\Misc\Football_Predictor\data\England.csv"
needed_cols = ['Div', 'Div_Name', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST',
               'AST', 'HR', 'AR', 'AVG_BETH', 'AVG_BETD', 'AVG_BETA', 'MAX_BETH', 'MAX_BETD', 'MAX_BETA',
               'MAX_BETH_COMP', 'MAX_BETD_COMP',
               'MAX_BETA_COMP']
# seasons = [f'{year}_{year + 1}' for year in range(15, 26)]

# df = (clean_data(path_to_english, needed_cols, export=True,
#               export_filepath=r"C:\Repos\Misc\Football_Predictor\data\sample_set.csv"))
