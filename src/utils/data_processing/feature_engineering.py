import pickle

import numpy as np
import pandas as pd
import streamlit as st
from agents import Agent, Runner, set_default_openai_key
from geopy.distance import great_circle
from geopy.geocoders import Nominatim


class TooFewGames(Exception):
    pass


api_key = st.secrets["keys"]["api_key"]
set_default_openai_key(api_key)


def is_over_two_seasons(df):
    if df["Season"].iloc[0] != df["Season"].iloc[-1]:
        return 1
    else:
        return 0


def is_league_change(df):
    recent_league = df["Div"].iloc[0]
    old_league = df["Div"].iloc[-1]
    if recent_league == old_league:
        return 0
    recent_div_num = int(recent_league[1]) if recent_league[1] != "C" else 5
    old_div_num = int(old_league[1]) if old_league[1] != "C" else 5
    if recent_div_num < old_div_num:
        return 1
    else:
        return -1


def return_n_game_info(df, n):
    reduced_df = df.head(n)
    if len(reduced_df) < 3:
        raise TooFewGames("")
    new_season = is_over_two_seasons(reduced_df)
    league_change = is_league_change(reduced_df)
    return new_season, league_change, reduced_df


def find_most_recent_n_games(df, team, date, n, games="All"):
    filtered_df = df[(df["Date"] < date) & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))]
    filtered_home_df = filtered_df[filtered_df["HomeTeam"] == team]
    filtered_away_df = filtered_df[filtered_df["AwayTeam"] == team]

    if games == "All":
        return return_n_game_info(filtered_df, n)
    elif games == "Home":
        return return_n_game_info(filtered_home_df, n)
    elif games == "Away":
        return return_n_game_info(filtered_away_df, n)
    else:
        raise Exception("Incorrect games argument")


def calculate_form_stats_for_home_or_away(df, type):
    results = df["FTR"].to_list()
    home_goals = df["FTHG"].to_list()
    away_goals = df["FTAG"].to_list()
    no_of_games = len(results)
    if type == "Home":
        no_of_wins = results.count("H")
        no_of_losses = results.count("A")
        avg_goals_scored = round(sum(home_goals) / len(home_goals), 2)
        avg_goals_conceded = round(sum(away_goals) / len(away_goals), 2)
    elif type == "Away":
        no_of_wins = results.count("A")
        no_of_losses = results.count("H")
        avg_goals_scored = round(sum(away_goals) / len(away_goals), 2)
        avg_goals_conceded = round(sum(home_goals) / len(home_goals), 2)
    else:
        raise Exception("Incorrect type argument")
    return no_of_games, no_of_wins / no_of_games, no_of_losses / no_of_games, avg_goals_scored, avg_goals_conceded


def calculate_form_stats(df, team, games="All"):
    if games == "All":
        home_df = df[df["HomeTeam"] == team]
        away_df = df[df["AwayTeam"] == team]
        if (home_df.empty) or (away_df.empty):
            raise TooFewGames("")
        no_of_home_games, prop_of_home_wins, prop_of_home_losses, avg_home_goals_scored, avg_home_goals_conceded = (
            calculate_form_stats_for_home_or_away(home_df, "Home")
        )
        no_of_away_games, prop_of_away_wins, prop_of_away_losses, avg_away_goals_scored, avg_away_goals_conceded = (
            calculate_form_stats_for_home_or_away(away_df, "Away")
        )
        prop_of_wins = round(
            (prop_of_home_wins * (no_of_home_games / (no_of_home_games + no_of_away_games)))
            + (prop_of_away_wins * (no_of_away_games / (no_of_home_games + no_of_away_games))),
            2,
        )
        prop_of_losses = round(
            (prop_of_home_losses * (no_of_home_games / (no_of_home_games + no_of_away_games)))
            + (prop_of_away_losses * (no_of_away_games / (no_of_home_games + no_of_away_games))),
            2,
        )
        avg_goals_scored = round(
            (avg_home_goals_scored * (no_of_home_games / (no_of_home_games + no_of_away_games)))
            + (avg_away_goals_scored * (no_of_away_games / (no_of_home_games + no_of_away_games))),
            2,
        )
        avg_goals_conceded = round(
            (avg_home_goals_conceded * (no_of_home_games / (no_of_home_games + no_of_away_games)))
            + (avg_away_goals_conceded * (no_of_away_games / (no_of_home_games + no_of_away_games))),
            2,
        )
    elif games == "Home":
        no_of_games, prop_of_wins, prop_of_losses, avg_goals_scored, avg_goals_conceded = (
            calculate_form_stats_for_home_or_away(df, "Home")
        )
    elif games == "Away":
        no_of_games, prop_of_wins, prop_of_losses, avg_goals_scored, avg_goals_conceded = (
            calculate_form_stats_for_home_or_away(df, "Away")
        )
    else:
        raise Exception("Incorrect games argument")
    return prop_of_wins, prop_of_losses, avg_goals_scored, avg_goals_conceded


def no_of_red_cards_in_last_match(last_match, team):
    if last_match["HomeTeam"] == team:
        return int(last_match["HR"])
    elif last_match["AwayTeam"] == team:
        return int(last_match["AR"])
    else:
        raise Exception("Team not in last match: red_cards_in_last_match")


def extract_form_stats(df, team, date, n, games="All"):
    try:
        new_season, league_change, filtered_df = find_most_recent_n_games(df, team, date, n, games=games)
    except TooFewGames:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        prop_of_wins, prop_of_losses, avg_goals_scored, avg_goals_conceded = calculate_form_stats(
            filtered_df, team, games=games
        )
    except TooFewGames:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    return new_season, league_change, prop_of_wins, prop_of_losses, avg_goals_scored, avg_goals_conceded


def extract_all_form_info(df, team, date, overall_n, home_and_away_n):
    col_names = [
        "_new_season",
        "_league_change",
        "_prop_of_wins",
        "_prop_of_losses",
        "_avg_goals_scored",
        "_avg_goals_conceded",
    ]
    overall_form_info = extract_form_stats(df, team, date, overall_n)
    overall_form_info_dic = {"overall" + col_name: info for col_name, info in zip(col_names, overall_form_info)}
    home_form_info = extract_form_stats(df, team, date, home_and_away_n, games="Home")
    home_form_info_dic = {"home" + col_name: info for col_name, info in zip(col_names, home_form_info)}
    away_form_info = extract_form_stats(df, team, date, home_and_away_n, games="Away")
    away_form_info_dic = {"away" + col_name: info for col_name, info in zip(col_names, away_form_info)}
    all_form_info_dic = overall_form_info_dic | home_form_info_dic | away_form_info_dic
    return all_form_info_dic


def extract_last_match_info(df, team, date):
    filtered_df = df[(df["Date"] < date) & ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))]
    if filtered_df.empty:
        return {"red_cards_in_last_match": np.nan, "days_between_games": np.nan}
    last_match = filtered_df.iloc[0]
    try:
        red_cards_in_last_match = no_of_red_cards_in_last_match(last_match, team)
        days_between_games = abs((last_match["Date"] - date).days)
        return {"red_cards_in_last_match": red_cards_in_last_match, "days_between_games": days_between_games}
    except Exception:
        return {"red_cards_in_last_match": np.nan, "days_between_games": np.nan}


def extract_all_team_info(df, team, date, overall_n, home_and_away_n):
    all_form_info_dic = extract_all_form_info(df, team, date, overall_n, home_and_away_n)
    last_match_info_dic = extract_last_match_info(df, team, date)
    all_team_info_dic = all_form_info_dic | last_match_info_dic
    return all_team_info_dic


def find_location_of_team(agent, geolocator, team):
    result = Runner.run_sync(agent, team)
    location = geolocator.geocode(result.final_output + ", United Kingdom")
    point = location.point
    return point


def locate_each_team(df):
    teams = list(df["HomeTeam"].unique())
    agent = Agent(
        name="Assistant",
        instructions="I'm going to give you the names of English football teams. These might be nicknames or shortened names. Please provide the town/city with which they are based. Provide just the location and nothing more.",
    )
    geolocator = Nominatim(user_agent="my_geocoder")
    team_locations = {team: find_location_of_team(agent, geolocator, team) for team in teams}
    return team_locations


def compute_travel_distance(home_team, away_team, team_locations):
    home_location = team_locations[home_team]
    away_location = team_locations[away_team]
    distance = great_circle(home_location, away_location)
    return distance.km


def extract_all_match_info_dic(df, team_locations, home_team, away_team, date, overall_n, home_and_away_n):
    all_home_team_info_dic = extract_all_team_info(df, home_team, date, overall_n, home_and_away_n)
    all_away_team_info_dic = extract_all_team_info(df, away_team, date, overall_n, home_and_away_n)
    all_home_team_info_dic = {f"HT_{k}": v for k, v in all_home_team_info_dic.items()}
    all_away_team_info_dic = {f"AT_{k}": v for k, v in all_away_team_info_dic.items()}
    away_team_dist_travelled = compute_travel_distance(home_team, away_team, team_locations)
    full_match_dic = (
        all_home_team_info_dic | all_away_team_info_dic | {"away_team_dist_travelled": away_team_dist_travelled}
    )
    return full_match_dic


def extract_all_info_for_match(match_row, df, team_locations, overall_n, home_and_away_n):
    home_team = match_row["HomeTeam"]
    away_team = match_row["AwayTeam"]
    date = match_row["Date"]
    full_match_dic = extract_all_match_info_dic(
        df, team_locations, home_team, away_team, date, overall_n, home_and_away_n
    )
    return pd.Series(full_match_dic)


def add_all_features_to_df(info_df, path_to_team_locations, overall_n, home_and_away_n, test_df=None):
    with open(path_to_team_locations, "rb") as f:  # 'rb' = read binary
        team_locations = pickle.load(f)
    if test_df is None:
        df = info_df
        new_df = df.apply(
            lambda row: extract_all_info_for_match(row, info_df, team_locations, overall_n, home_and_away_n), axis=1
        )
    else:
        df = test_df
        new_df = test_df.apply(
            lambda row: extract_all_info_for_match(row, info_df, team_locations, overall_n, home_and_away_n), axis=1
        )
    needed_cols = [
        "Div",
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
        "Div_Name",
        "Season",
    ]
    processed_needed_cols = [col_name for col_name in needed_cols if col_name in list(df.columns)]
    original_df = df[processed_needed_cols]
    full_df = pd.concat([original_df, new_df], axis=1)
    return full_df
