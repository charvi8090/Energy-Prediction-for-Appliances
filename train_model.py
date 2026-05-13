"""
Model Training Script for Household Energy Prediction
This script trains an Extra Trees Regressor model and saves it for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load the energy dataset"""
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully! Shape: {df.shape}")
    return df

def prepare_features(df):
    """Prepare features and target variable"""
    print("\nPreparing features...")
    
    # Drop date column and target variable
    X = df.drop(['date', 'Appliances'], axis=1, errors='ignore')
    y = df['Appliances']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y

def train_model(X_train, y_train, quick_mode=False):
    """Train the Extra Trees Regressor model with hyperparameter tuning"""
    print("\nTraining model...")
    
    if quick_mode:
        print("Running in QUICK MODE (fewer hyperparameters)...")
        # Simplified hyperparameters for faster training
        h_param = {
            'n_estimators': [100, 200],
            'max_depth': [50, 70],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [False]
        }
        cv = 3
    else:
        print("Running in FULL MODE (comprehensive tuning)...")
        # Full hyperparameter grid from the notebook
        h_param = {
            'n_estimators': [100, 500, 1400],
            'max_depth': [50, 70, 100],
            'max_features': ['sqrt', 'log2'],
            'criterion': ['squared_error'],
            'bootstrap': [False, True]
        }
        cv = 5
    
    # Initialize the base model
    base_model = ExtraTreesRegressor(random_state=42, n_jobs=-1)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=h_param,
        cv=cv,
        scoring='neg_mean_squared_error',
        verbose=2,
        n_jobs=-1
    )
    
    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score (neg MSE): {grid_search.best_score_:.6f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate the model on training and test sets"""
    print("\nEvaluating model...")
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Training MSE:   {train_mse:.6f}")
    print(f"Training R²:    {train_r2:.6f}")
    print(f"Test MSE:       {test_mse:.6f}")
    print(f"Test R²:        {test_r2:.6f}")
    print("="*50)
    
    return {
        'train_mse': train_mse,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_r2': test_r2
    }

def save_model(model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """Save the trained model and scaler"""
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Saving scaler to {scaler_path}...")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Model and scaler saved successfully!")

def main():
    """Main training pipeline"""
    print("="*50)
    print("HOUSEHOLD ENERGY PREDICTION - MODEL TRAINING")
    print("="*50)
    
    # Configuration
    DATA_PATH = 'energydata_complete.csv'  # Update this path as needed
    QUICK_MODE = True  # Set to False for full hyperparameter tuning
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    try:
        # Step 1: Load data
        df = load_data(DATA_PATH)
        
        # Step 2: Prepare features
        X, y = prepare_features(df)
        
        # Step 3: Train-test split
        print(f"\nSplitting data (test_size={TEST_SIZE})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")
        
        # Step 4: Feature scaling
        print("\nScaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Step 5: Train model
        model = train_model(X_train_scaled, y_train, quick_mode=QUICK_MODE)
        
        # Step 6: Evaluate model
        metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Step 7: Save model
        save_model(model, scaler)
        
        print("\n✅ Training completed successfully!")
        print("\nYou can now run the Streamlit app with: streamlit run app.py")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find data file '{DATA_PATH}'")
        print("Please ensure the dataset is in the same directory as this script.")
        print("You can download it or update the DATA_PATH variable.")
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()