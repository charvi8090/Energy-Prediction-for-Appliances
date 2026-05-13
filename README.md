# Energy_Regression
- HuggingFace:  https://huggingface.co/spaces/Charvi8090/Energy_Prediction
# Household Energy Prediction

A machine learning project that predicts household appliance energy consumption using environmental and weather data. This project explores various regression algorithms to forecast energy usage based on temperature, humidity, and outdoor weather conditions.

## Overview

This project analyzes energy consumption patterns in a household by examining the relationship between appliances energy usage and various environmental factors including indoor temperature/humidity readings from multiple rooms and outdoor weather conditions.

## Dataset

The dataset (`energydata_complete.csv`) contains energy consumption data collected at 10-minute intervals, featuring:

- **Target Variable**: 
  - `Appliances` - Energy consumption of appliances (Wh)
  - `lights` - Energy consumption of lights (Wh)

- **Features**:
  - **Indoor Climate** (9 rooms): Temperature (T1-T9) and Relative Humidity (RH_1 to RH_9)
  - **Outdoor Weather**: Temperature, atmospheric pressure, humidity, wind speed, visibility, and dew point
  - **Time**: Date and timestamp information
  - **Random Variables**: rv1, rv2 (for baseline comparison)

## Project Structure

```
Household_energy_prediction.ipynb    # Main analysis notebook
energydata_complete.csv              # Dataset (not included in repo)
README.md                            # Project documentation
```

## Technologies Used

**Core Libraries:**
- Python 3.x
- NumPy & Pandas - Data manipulation
- Matplotlib & Seaborn - Data visualization
- Missingno - Missing data analysis

**Machine Learning:**
- Scikit-learn - ML algorithms and preprocessing
- XGBoost - Gradient boosting
- LightGBM - Light gradient boosting
- SciPy & Statsmodels - Statistical analysis

## Machine Learning Models

The project implements and compares multiple regression algorithms:

### Linear Models
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net

### Tree-Based Models
- Decision Tree Regressor
- Random Forest Regressor
- Extra Trees Regressor
- Gradient Boosting Regressor
- XGBoost
- LightGBM

### Ensemble Methods
- Bagging Regressor
- Stacking Regressor

## Methodology

1. **Data Loading & Exploration**
   - Load energy consumption data
   - Exploratory data analysis (EDA)
   - Statistical summaries and visualizations

2. **Data Preprocessing**
   - Handle missing values
   - Feature engineering from datetime
   - Correlation analysis
   - Feature selection

3. **Feature Scaling**
   - StandardScaler for normalizing features
   - Train-test split (maintaining temporal order)

4. **Model Training & Evaluation**
   - Train multiple ML models
   - Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
   - Model comparison using MSE and R² metrics

5. **Model Selection**
   - Performance comparison across all models
   - Selection based on test set performance

## Results

The models were evaluated using Mean Squared Error (MSE) and R² Score:

| Model | Train MSE | Test MSE | Train R² | Test R² |
|-------|-----------|----------|----------|---------|
| Linear Regression | 0.0382 | 0.0467 | 0.1800 | -0.0001 |
| Ridge | 0.0382 | 0.0391 | 0.1800 | 0.1622 |
| Lasso | 0.0382 | 0.0391 | 0.1800 | 0.1624 |
| Elastic Net | 0.0382 | 0.0391 | 0.1800 | 0.1624 |
| Decision Tree | 0.0112 | 0.0197 | 0.7601 | 0.5769 |
| **Extra Trees Regressor*** | ~0 | 0.0123 | 1.0000 | **0.7365** |

**Best Model: Extra Trees Regressor with hyperparameter tuning**
- Test R² Score: **0.736** (73.6% variance explained)
- Test MSE: **0.0123**

### Hyperparameter Optimization

The Extra Trees Regressor achieved optimal performance with:
- `n_estimators`: 1400
- `max_depth`: 70
- `max_features`: 'log2'
- `criterion`: 'squared_error'
- `bootstrap`: False

## Key Insights

- Tree-based ensemble methods significantly outperform linear models
- Indoor temperature and humidity sensors provide strong predictive power
- Extra Trees Regressor achieved the best generalization to unseen data
- The model successfully captures 73.6% of the variance in energy consumption patterns

## Installation & Usage

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn missingno
pip install scikit-learn xgboost lightgbm scipy statsmodels
```

### Running the Notebook
1. Clone this repository
2. Install required dependencies
3. Place the dataset (`energydata_complete.csv`) in the working directory
4. Open and run `Household_energy_prediction.ipynb` in Jupyter Notebook or Google Colab

```python
# For Google Colab users
from google.colab import drive
drive.mount('/content/drive')
```

## Future Improvements

- Implement deep learning models (LSTM, GRU) for time-series forecasting
- Add feature importance analysis and interpretation
- Create a real-time prediction dashboard
- Incorporate external factors (holidays, occupancy patterns)
- Deploy the model as a web application or API
- Experiment with time-series cross-validation
- Add forecast intervals and uncertainty quantification

## License

This project is available under the MIT License.

## Acknowledgments

- Dataset source: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- Inspired by energy efficiency and smart home research

## Contact

Feel free to reach out for questions, suggestions, or collaboration opportunities!

---

**Note**: This is an educational/research project demonstrating machine learning techniques for energy consumption prediction.
