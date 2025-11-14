"""Streamlit application utilities for displaying football match predictions and information."""

from pathlib import Path

import joblib
import streamlit as st

from utils.model_building.model_build import extract_train_and_test
from utils.model_building.model_evaluation import get_ev_of_predictions
from utils.model_building.model_params import *
from utils.overall_data_process import extract_data_for_model


class StreamlitApp:
    def __init__(self):
        """Initializes the StreamlitApp with custom CSS styling and page configuration."""
        # Add custom CSS for fonts/colors
        st.markdown(
            """
            <style>
            .main {background-color: #f8f9fa;}
            .stChatMessage {font-size: 1.1em;}
            .stApp {font-family: 'Segoe UI', Arial, sans-serif;}
            .sidebar .sidebar-content {background: #e3e6ea;}
            .chat-message-box {
                background: #fff;
                border-radius: 10px;
                border: 1px solid #e3e6ea;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
                padding: 1rem 1.2rem;
                margin-bottom: 0.7rem;
                margin-top: 0.2rem;
            }
            .chat-message-user {
                border-left: 4px solid #10a37f;
            }
            .chat-message-assistant {
                border-left: 4px solid #1a73e8;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.set_page_config(page_title="Football Predictor", layout="wide")

    def add_basic_login(self):
        """Adds password-based authentication to the Streamlit app."""
        AUTHORIZED_PASSWORD = st.secrets["passwords"]["app"]  # change this to whatever you like

        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False

        if not st.session_state.authenticated:
            st.title("🔒 Login")
            password = st.text_input("Enter password", type="password")
            if st.button("Login"):
                if password == AUTHORIZED_PASSWORD:
                    st.session_state.authenticated = True
                else:
                    st.error("❌ Wrong password")
            st.stop()

    def set_out_base_app(self):
        """Configures the base application layout with sidebar navigation and pages."""
        st.sidebar.image("data/football.png", width=64)
        st.sidebar.markdown("""
        ## Welcome!
        - View recent predictions by the Football Predictor.
        - Switch to the Information page for more details.
        - Visit the Data Source link below to see the data used.
        """)
        # Add a link at the bottom left of the sidebar
        st.sidebar.markdown(
            f"""<div style="position: fixed; left: 1.5rem; bottom: 1.5rem; z-index: 100;">
                <a href={st.secrets["links"]["data_source"]} target="_blank" style="color: #888; font-size: 0.95em; text-decoration: none;">Data Source</a>
            </div>""",
            unsafe_allow_html=True,
        )
        st.set_page_config(page_title="Football Predictor", layout="wide")
        prediction_page = st.Page(self.set_out_predictions_page, title="Football Predictor", icon=":material/sports_soccer:")
        info_page = st.Page(self.set_out_info_page, title="Information", icon=":material/info:")
        pg = st.navigation([prediction_page, info_page], position="sidebar")
        pg.run()

    def set_out_predictions_page(self):
        """Displays the predictions page with upcoming match predictions."""
        st.title("Football Predictor - Upcoming Match Predictions")
        self.produce_predictions()

    def set_out_info_page(self):
        """Displays the information page with project details and author contact information."""
        st.title("Information")

        st.markdown("""
                ## About this site

                This application provides predicted outcomes for upcoming football matches using a machine learning model trained on historical
                match data. The predictions are refreshed automatically with the latest fixtures and betting odds. For each match the app shows
                a predicted result together with an Expected Value (EV) figure — the EV represents the estimated return on a £1 bet when
                comparing the model's probability to the market odds.
                """)

        st.markdown("""
                ## How the model was built

                - Data source: historical match data (e.g. Football-Data.co.uk) was used as the primary input.
                - Feature engineering: recent form, home/away performance, travel distance, head-to-head, and aggregated rolling statistics
                    were derived to give the model contextual signals about teams and fixtures.
                - Model training: the processed dataset was used to train a supervised learning model (XGBoost). Hyperparameters were selected using cross-validation and evaluation on held-out seasons.
                - Evaluation & output: predictions are calibrated to probabilities and transformed into an Expected Value (EV) metric by
                    comparing model probability vs. implied probability from the odds.
                """)

        st.markdown("""
                ## Links & resources

                - Data source: see the **Data Source** link in the bottom-left of the sidebar.
                - Kaggle dataset (showing engineered features): [Kaggle dataset](https://www.kaggle.com/datasets/simonphilp/english-football-data)  
                - Blog / write-up: [Modeling football predictions — blog post](#)  
                - Code repository: [GitHub repo](#)
                """)

        st.markdown("---")

        # Photo + contact details
        cols = st.columns([0.28, 0.72])
        with cols[0]:
            img_path = Path("data/profile_pic.jpeg")
            if img_path.exists():
                st.image(str(img_path), caption="Author", use_container_width=True)
            else:
                st.markdown(
                    """
                                        <div style='width:180px;height:180px;background:#eef2f5;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#6c757d'>
                                            <div style='text-align:center'>
                                                <div style='font-size:20px'>📸</div>
                                                <div style='margin-top:6px'>Photo<br>placeholder</div>
                                            </div>
                                        </div>
                                        """,
                    unsafe_allow_html=True,
                )

        with cols[1]:
            st.subheader("Contact")
            st.markdown(
                """
                                **Simon Philp**  
                                Email: [simonphilp27@gmail.com](mailto:simonphilp27@gmail.com)   
                                LinkedIn: [Profile](https://www.linkedin.com/in/simon-philp-b057891b9/)  

                                If you'd like to collaborate, suggest improvements, or get access to the model/code, please use the links above or
                                contact me directly.
                                """
            )

    def produce_predictions(self):
        """Generates and caches match predictions using the trained model."""
        countries = ["England"]
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

        best_model = joblib.load("data/best_model.pkl")

        if "model_results" not in st.session_state:
            clean_upcoming_df = extract_data_for_model(countries, test_seasons, needed_cols, country_divisions)
            if clean_upcoming_df.empty:
                st.session_state["model_results"] = None
            else:
                # If model not loaded, do not attempt prediction
                if best_model is None:
                    st.session_state["model_results"] = None
                else:
                    preds_train_df, na_test_df = extract_train_and_test(clean_upcoming_df, train_split=1)
                    df = get_ev_of_predictions(preds_train_df, best_model, lvl1_unneeded_cols, include_odds=True)
                    st.session_state["model_results"] = df

        if st.session_state["model_results"] is None:
            st.write("No upcoming matches within the specified range. Come back another day for new predictions!")
        else:
            df = st.session_state["model_results"]
            self.format_df(df)

    def format_df(self, df):
        """Formats and displays the predictions dataframe with custom column configurations."""
        column_config = {
            "date": st.column_config.DatetimeColumn("Date", format="MMMM D"),
            "div": "League",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "Prediction_EV": st.column_config.NumberColumn("Expected Value", format="%.2f"),
            "Prediction_Odds": st.column_config.NumberColumn("Best Odds", format="%.2f"),
            "Prediction_Company": "Odds Company",
        }
        st.dataframe(df, column_config=column_config, hide_index=True)
