# Football Predictor

A machine learning-powered Streamlit application that predicts outcomes of upcoming football matches and calculates expected value (EV) for betting decisions.

## Overview

This project uses historical football match data, betting odds, and engineered features (team form, travel distance, rest days, etc.) to train classification models that predict match outcomes (Home Win, Draw, Away Win). The application provides:

- **Match Predictions**: Predicted outcomes for upcoming fixtures
- **Expected Value Calculations**: EV metrics comparing model probabilities to market odds
- **Best Odds**: Recommendations for the best available betting odds
- **Interactive Dashboard**: Streamlit-based web interface for viewing predictions

## Features

### Data Pipeline
- **Web Scraping**: Automated downloading of match data and betting odds from Football-Data.co.uk
- **Data Cleaning**: Processing of raw data including date parsing, type conversion, and odds aggregation
- **Feature Engineering**: 
  - Recent form statistics (overall, home, and away)
  - Rolling averages for goals scored/conceded
  - League position changes
  - Travel distance between teams
  - Days between matches
  - Red cards in previous matches

### Machine Learning
- **Multiple Model Support**: XGBoost, Random Forest, LightGBM, and CatBoost
- **Hyperparameter Tuning**: Grid search with time series cross-validation
- **Feature Selection**: Automated testing of different feature combinations
- **Probability Calibration**: Converting predictions to probabilities for EV calculation

### Expected Value Analysis
- **EV Calculation**: Compares model probability to implied probability from odds
- **Best Odds Identification**: Finds the highest available odds across multiple bookmakers
- **Profit Tracking**: Historical profit analysis for model validation

## Project Structure

```
Football_Predictor/
├── data/                           # Data storage
│   ├── profile_pic.jpeg
│   ├── football.png            
│   ├── team_locations.pkl
│   ├── odds_columns.txt
│   └── best_model.pkl             # Trained model
├── src/
│   ├── streamlit_app.py           # Main application entry point
│   └── utils/
│       ├── overall_data_process.py    # Data pipeline orchestration
│       ├── model_run.py               # Model training script
│       ├── data_download/
│       │   └── download_data.py       # Web scraping utilities
│       ├── data_processing/
│       │   ├── data_cleaning.py       # Raw data cleaning
│       │   ├── feature_engineering.py # Feature creation
│       │   └── final_clean.py         # Final data preparation
│       ├── model_building/
│       │   ├── model_build.py         # Model training
│       │   ├── model_evaluation.py    # Evaluation metrics
│       │   └── model_params.py        # Model configurations
│       └── streamlit_app/
│           └── streamlit_app_utils.py # UI components
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Football_Predictor.git
cd Football_Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Streamlit secrets (create `.streamlit/secrets.toml`):
```toml
[passwords]
app = "your_password_here"

[keys]
api_key = "your_openai_api_key"

[links]
data_source = "https://www.football-data.co.uk/"
```

## Usage

### Training a Model

Run the model training script:
```bash
python src/utils/model_run.py
```

This will:
- Download historical match data
- Process and engineer features
- Train multiple models with hyperparameter tuning
- Save the best model to `data/best_model.pkl`

### Running the Streamlit App

Start the application:
```bash
streamlit run src/streamlit_app.py
```

The app provides:
- **Football Predictor Page**: View upcoming match predictions sorted by EV
- **Information Page**: Learn about the model, methodology, and contact information

### Making Predictions

```python
from utils.overall_data_process import extract_data_for_model
from utils.model_building.model_evaluation import get_ev_of_predictions
import joblib

# Load trained model
model = joblib.load("data/best_model.pkl")

# Get upcoming matches data
countries = ["England"]
seasons = ["24_25"]
needed_cols = [...]  # See model_run.py for full list
country_divisions = {"England": ["E0", "E1", "E2", "E3", "E4", "EC"]}

upcoming_df = extract_data_for_model(countries, seasons, needed_cols, 
                                     country_divisions, data="Preds")

# Generate predictions
predictions = get_ev_of_predictions(upcoming_df, model, cols_to_drop, 
                                   include_odds=True)
```

## Model Details

### Features
The model uses 40+ engineered features including:
- **Overall Form** (last 20 games): Win rate, loss rate, goals scored/conceded, league position change
- **Home Form** (last 5 home games): Home-specific performance metrics
- **Away Form** (last 5 away games): Away-specific performance metrics
- **Recent Match Info**: Red cards, days since last match
- **Travel Distance**: Geographic distance between team locations
- **Betting Odds**: Average and maximum odds from multiple bookmakers

### Training
- **Cross-Validation**: Time series split to preserve temporal ordering
- **Evaluation Metric**: Classification accuracy and profit on test set
- **Best Model Selection**: Automatic selection based on test accuracy

### Performance
The model is evaluated on:
- Classification accuracy on held-out test sets
- Expected Value (EV) of predictions
- Historical profit when following model recommendations

## Data Sources

- **Match Results**: [Football-Data.co.uk](https://www.football-data.co.uk/)
- **Betting Odds**: Multiple bookmakers including Bet365, Betfair, William Hill, etc.
- **Geographic Data**: Team locations via Nominatim geocoding

## Contributing

Contributions are welcome! Areas for improvement:
- Additional leagues and competitions
- More sophisticated feature engineering
- Alternative model architectures (neural networks)
- Live odds integration
- Automated model retraining

## License

This project is for educational purposes. Always gamble responsibly.

## Author

**Simon Philp**
- Email: simonphilp27@gmail.com
- LinkedIn: [Profile](https://www.linkedin.com/in/simon-philp-b057891b9/)

## Acknowledgments

- Football-Data.co.uk for providing historical match data
- The scikit-learn, XGBoost, and Streamlit communities
