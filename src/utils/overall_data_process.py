import pandas as pd
from datetime import datetime
from utils.data_download.download_data import get_recent_results, get_upcoming_matches, country_divisions
from utils.data_processing.data_cleaning import clean_data
from utils.data_processing.feature_engineering import add_all_features_to_df
from utils.data_processing.final_clean import predictions_col_names, final_clean


def get_data(countries, seasons):
    recent_df = get_recent_results(countries, seasons)
    upcoming_df = get_upcoming_matches(countries, country_divisions)
    return recent_df, upcoming_df

def extract_data_for_model(countries, seasons, needed_cols, data='Preds'):
    past_df, upcoming_df = get_data(countries, seasons)
    clean_past_df = clean_data('', needed_cols, export=False, export_filepath=None, df=past_df)
    if data == 'Train':
        clean_past_df = add_all_features_to_df(clean_past_df,
                                               r"C:\Repos\Misc\Football_Predictor\data\team_locations.pkl",
                                               20, 5)
        final_past_df = final_clean(clean_past_df, predictions_col_names)
        return final_past_df
    elif data == 'Preds':
        clean_upcoming_df = clean_data('', needed_cols, export=False, export_filepath=None, df=upcoming_df,
                                       test_data=True)
        today = pd.Timestamp(datetime.now().date()) # current time with timezone awareness if applicable
        upcoming_filtered_df = clean_upcoming_df[clean_upcoming_df['Date'] >= today]
        clean_upcoming_df = add_all_features_to_df(clean_past_df,
                                                   r"C:\Repos\Misc\Football_Predictor\data\team_locations.pkl", 20, 5,
                                                   test_df=upcoming_filtered_df)
        final_upcoming_df = final_clean(clean_upcoming_df, predictions_col_names)
        return final_upcoming_df

countries = ['England']
seasons = ['24_25', '25_26']
needed_cols = ['Div', 'Div_Name', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST',
               'AST', 'HR', 'AR', 'AVG_BETH', 'AVG_BETD', 'AVG_BETA', 'MAX_BETH', 'MAX_BETD', 'MAX_BETA',
               'MAX_BETH_COMP', 'MAX_BETD_COMP', 'MAX_BETA_COMP']

#clean_upcoming_df = extract_data_for_model(countries, seasons, needed_cols)
#clean_past_df = extract_data_for_model(countries, seasons, needed_cols, data='Train')
#clean_upcoming_df.head()
