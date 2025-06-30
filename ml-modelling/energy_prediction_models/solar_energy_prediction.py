import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SolarEnergyPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess weather and energy data"""
        print("Loading data...")
        
        # Load datasets
        weather_df = pd.read_csv('../data/weather_features.csv')
        energy_df = pd.read_csv('../data/energy_data.csv')
        
        # Convert time columns to datetime
        weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])
        energy_df['time'] = pd.to_datetime(energy_df['time'])
        
        # Merge datasets on time
        merged_df = pd.merge(weather_df, energy_df[['time', 'generation solar']], 
                           left_on='dt_iso', right_on='time', how='inner')
        
        # Extract time-based features
        merged_df['hour'] = merged_df['dt_iso'].dt.hour
        merged_df['month'] = merged_df['dt_iso'].dt.month
        merged_df['is_daytime'] = ((merged_df['hour'] >= 6) & (merged_df['hour'] <= 18)).astype(int)
        
        # Create solar-specific features
        merged_df['solar_angle'] = self._calculate_solar_angle(merged_df['hour'], merged_df['month'])
        merged_df['cloud_impact'] = merged_df['clouds_all'] / 100.0
        
        # Handle missing values
        merged_df = merged_df.dropna()
        
        # Select features for solar prediction
        solar_features = [
            'temp', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
            'clouds_all', 'hour', 'month', 'is_daytime', 'solar_angle', 'cloud_impact'
        ]
        
        # Filter for daytime hours
        solar_df = merged_df[merged_df['is_daytime'] == 1].copy()
        
        # Prepare X and y
        X = solar_df[solar_features]
        y = solar_df['generation solar']
        
        print(f"Final dataset shape: {X.shape}")
        return X, y, solar_features
    
    def _calculate_solar_angle(self, hour, month):
        """Calculate approximate solar angle"""
        hour_factor = np.sin(np.pi * (hour - 6) / 12)
        hour_factor = np.maximum(hour_factor, 0)
        seasonal_factor = np.sin(np.pi * (month - 1) / 6)
        seasonal_factor = (seasonal_factor + 1) / 2
        return hour_factor * seasonal_factor
    
    def train_model(self, X, y):
        """Train the solar energy prediction model"""
        print("Training solar energy prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        return X_test, y_test, y_pred, feature_importance
    
    def plot_results(self, X_test, y_test, y_pred, feature_importance):
        """Plot model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Predictions vs Actual
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Solar Generation')
        axes[0, 0].set_ylabel('Predicted Solar Generation')
        axes[0, 0].set_title('Solar Energy Predictions vs Actual')
        
        # Feature importance
        top_features = feature_importance.head(10)
        axes[0, 1].barh(top_features['feature'], top_features['importance'])
        axes[0, 1].set_title('Top 10 Feature Importance')
        axes[0, 1].set_xlabel('Importance')
        
        # Residuals
        residuals = y_test - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        
        # Distribution of predictions
        axes[1, 1].hist(y_pred, bins=30, alpha=0.7, label='Predicted')
        axes[1, 1].hist(y_test, bins=30, alpha=0.7, label='Actual')
        axes[1, 1].set_xlabel('Solar Generation')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Predictions vs Actual')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('solar_energy_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run solar energy prediction"""
    print("=== Solar Energy Prediction Model ===")
    
    # Initialize predictor
    predictor = SolarEnergyPredictor()
    
    # Load and preprocess data
    X, y, features = predictor.load_and_preprocess_data()
    
    # Train model
    X_test, y_test, y_pred, feature_importance = predictor.train_model(X, y)
    
    # Plot results
    predictor.plot_results(X_test, y_test, y_pred, feature_importance)
    
    print("\nSolar energy prediction model training completed!")

if __name__ == "__main__":
    main() 