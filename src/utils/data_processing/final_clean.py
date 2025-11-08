full_dataset_col_names = {'Date': 'date', 'Div': 'div', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
                          'Season': 'season', 'FTR': 'FTR', 'FTHG': 'FTHG', 'FTAG': 'FTAG', 'HS': 'HS', 'AS': 'AS',
                          'HST': 'HST',
                          'AST': 'AST', 'HR': 'HR', 'AR': 'AR', 'AVG_BETH': 'AVG_BETH', 'AVG_BETD': 'AVG_BETD',
                          'AVG_BETA': 'AVG_BETA',
                          'MAX_BETH': 'MAX_BETH', 'MAX_BETD': 'MAX_BETD', 'MAX_BETA': 'MAX_BETA',
                          'HT_overall_league_change': 'HT_LC_20', 'HT_overall_prop_of_wins': 'HT_OW%_20',
                          'HT_overall_prop_of_losses': 'HT_OL%_20', 'HT_overall_avg_goals_scored': 'HT_OAGS_20',
                          'HT_overall_avg_goals_conceded': 'HT_OAGC_20', 'HT_home_prop_of_wins': 'HT_HW%_5',
                          'HT_home_prop_of_losses': 'HT_HL%_5', 'HT_home_avg_goals_scored': 'HT_HAGS_5',
                          'HT_home_avg_goals_conceded': 'HT_HAGC_5', 'HT_away_prop_of_wins': 'HT_AW%_5',
                          'HT_away_prop_of_losses': 'HT_AL%_5', 'HT_away_avg_goals_scored': 'HT_AAGS_5',
                          'HT_away_avg_goals_conceded': 'HT_AAGC_5', 'HT_red_cards_in_last_match': 'HT_RC_1',
                          'HT_days_between_games': 'HT_RD', 'AT_overall_league_change': 'AT_LC_20',
                          'AT_overall_prop_of_wins': 'AT_OW%_20',
                          'AT_overall_prop_of_losses': 'AT_OL%_20', 'AT_overall_avg_goals_scored': 'AT_OAGS_20',
                          'AT_overall_avg_goals_conceded': 'AT_OAGC_20', 'AT_home_prop_of_wins': 'AT_HW%_5',
                          'AT_home_prop_of_losses': 'AT_HL%_5', 'AT_home_avg_goals_scored': 'AT_HAGS_5',
                          'AT_home_avg_goals_conceded': 'AT_HAGC_5', 'AT_away_prop_of_wins': 'AT_AW%_5',
                          'AT_away_prop_of_losses': 'AT_AL%_5', 'AT_away_avg_goals_scored': 'AT_AAGS_5',
                          'AT_away_avg_goals_conceded': 'AT_AAGC_5', 'AT_red_cards_in_last_match': 'AT_RC_1',
                          'AT_days_between_games': 'AT_RD', 'away_team_dist_travelled': 'AT_DT'}

predictions_col_names = {'Date': 'date', 'Div': 'div', 'HomeTeam': 'home_team', 'AwayTeam': 'away_team', 'FTR': 'FTR',
                         'AVG_BETH': 'AVG_BETH', 'AVG_BETD': 'AVG_BETD',
                         'AVG_BETA': 'AVG_BETA',
                         'MAX_BETH': 'MAX_BETH', 'MAX_BETD': 'MAX_BETD', 'MAX_BETA': 'MAX_BETA',
                         'MAX_BETH_COMP': 'MAX_BETH_COMP', 'MAX_BETD_COMP': 'MAX_BETD_COMP',
                         'MAX_BETA_COMP': 'MAX_BETA_COMP',
                         'HT_overall_league_change': 'HT_LC_20', 'HT_overall_prop_of_wins': 'HT_OW%_20',
                         'HT_overall_prop_of_losses': 'HT_OL%_20', 'HT_overall_avg_goals_scored': 'HT_OAGS_20',
                         'HT_overall_avg_goals_conceded': 'HT_OAGC_20', 'HT_home_prop_of_wins': 'HT_HW%_5',
                         'HT_home_prop_of_losses': 'HT_HL%_5', 'HT_home_avg_goals_scored': 'HT_HAGS_5',
                         'HT_home_avg_goals_conceded': 'HT_HAGC_5', 'HT_away_prop_of_wins': 'HT_AW%_5',
                         'HT_away_prop_of_losses': 'HT_AL%_5', 'HT_away_avg_goals_scored': 'HT_AAGS_5',
                         'HT_away_avg_goals_conceded': 'HT_AAGC_5', 'HT_red_cards_in_last_match': 'HT_RC_1',
                         'HT_days_between_games': 'HT_RD', 'AT_overall_league_change': 'AT_LC_20',
                         'AT_overall_prop_of_wins': 'AT_OW%_20',
                         'AT_overall_prop_of_losses': 'AT_OL%_20', 'AT_overall_avg_goals_scored': 'AT_OAGS_20',
                         'AT_overall_avg_goals_conceded': 'AT_OAGC_20', 'AT_home_prop_of_wins': 'AT_HW%_5',
                         'AT_home_prop_of_losses': 'AT_HL%_5', 'AT_home_avg_goals_scored': 'AT_HAGS_5',
                         'AT_home_avg_goals_conceded': 'AT_HAGC_5', 'AT_away_prop_of_wins': 'AT_AW%_5',
                         'AT_away_prop_of_losses': 'AT_AL%_5', 'AT_away_avg_goals_scored': 'AT_AAGS_5',
                         'AT_away_avg_goals_conceded': 'AT_AAGC_5', 'AT_red_cards_in_last_match': 'AT_RC_1',
                         'AT_days_between_games': 'AT_RD', 'away_team_dist_travelled': 'AT_DT'}


def final_clean(df, new_col_names):
    cols_to_keep = list(new_col_names.keys())
    processed_cols_to_keep = [col_name for col_name in cols_to_keep if col_name in list(df.columns)]
    filtered_df = df[processed_cols_to_keep]
    renamed_df = filtered_df.rename(columns=new_col_names)
    return renamed_df

# tst_df = final_clean(clean_upcoming_df, predictions_col_names)
# renamed_df = final_clean(r"C:\Repos\Misc\Football_Predictor\data\full_sample_set_2.csv", col_names)
# renamed_df.to_csv(r"C:\Repos\Misc\Football_Predictor\data\cleaned_full_sample_set.csv", index=False)
# renamed_df.columns
