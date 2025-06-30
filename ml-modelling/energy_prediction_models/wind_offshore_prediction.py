import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

class WindOffshorePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess weather and energy data for wind offshore prediction"""
        print("Loading data for wind offshore prediction...")
        
        # Load datasets
        weather_df = pd.read_csv('../data/weather_features.csv')
        energy_df = pd.read_csv('../data/energy_data.csv')
        
        # Convert time columns to datetime
        weather_df['dt_iso'] = pd.to_datetime(weather_df['dt_iso'])
        energy_df['time'] = pd.to_datetime(energy_df['time'])
        
        # Merge datasets on time
        merged_df = pd.merge(weather_df, energy_df[['time', 'generation wind offshore']], 
                           left_on='dt_iso', right_on='time', how='inner')
        
        # Extract time-based features
        merged_df['hour'] = merged_df['dt_iso'].dt.hour
        merged_df['day'] = merged_df['dt_iso'].dt.day
        merged_df['month'] = merged_df['dt_iso'].dt.month
        merged_df['year'] = merged_df['dt_iso'].dt.year
        merged_df['day_of_week'] = merged_df['dt_iso'].dt.dayofweek
        merged_df['season'] = merged_df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                                      3: 'spring', 4: 'spring', 5: 'spring',
                                                      6: 'summer', 7: 'summer', 8: 'summer',
                                                      9: 'autumn', 10: 'autumn', 11: 'autumn'})
        
        # Create wind-specific features
        merged_df['wind_speed_squared'] = merged_df['wind_speed'] ** 2  # Wind power is proportional to v³
        merged_df['wind_speed_cubed'] = merged_df['wind_speed'] ** 3
        merged_df['wind_direction_sin'] = np.sin(np.radians(merged_df['wind_deg']))
        merged_df['wind_direction_cos'] = np.cos(np.radians(merged_df['wind_deg']))
        
        # Create seasonal wind patterns
        merged_df['is_winter'] = (merged_df['month'].isin([12, 1, 2])).astype(int)
        merged_df['is_spring'] = (merged_df['month'].isin([3, 4, 5])).astype(int)
        merged_df['is_summer'] = (merged_df['month'].isin([6, 7, 8])).astype(int)
        merged_df['is_autumn'] = (merged_df['month'].isin([9, 10, 11])).astype(int)
        
        # Create time-of-day features for wind patterns
        merged_df['is_night'] = ((merged_df['hour'] >= 22) | (merged_df['hour'] <= 6)).astype(int)
        merged_df['is_morning'] = ((merged_df['hour'] >= 6) & (merged_df['hour'] <= 12)).astype(int)
        merged_df['is_afternoon'] = ((merged_df['hour'] >= 12) & (merged_df['hour'] <= 18)).astype(int)
        merged_df['is_evening'] = ((merged_df['hour'] >= 18) & (merged_df['hour'] <= 22)).astype(int)
        
        # Create pressure gradient features (important for wind)
        merged_df['pressure_gradient'] = merged_df['pressure'].diff().fillna(0)
        
        # Handle missing values
        merged_df = merged_df.dropna()
        
        # Select features for wind offshore prediction
        wind_features = [
            'temp', 'temp_min', 'temp_max', 'pressure', 'humidity',
            'wind_speed', 'wind_deg', 'rain_1h', 'rain_3h', 'snow_3h',
            'clouds_all', 'hour', 'day', 'month', 'year', 'day_of_week',
            'wind_speed_squared', 'wind_speed_cubed', 'wind_direction_sin', 'wind_direction_cos',
            'is_winter', 'is_spring', 'is_summer', 'is_autumn',
            'is_night', 'is_morning', 'is_afternoon', 'is_evening',
            'pressure_gradient'
        ]
        
        # Encode categorical variables
        categorical_cols = ['weather_main', 'weather_description', 'season']
        for col in categorical_cols:
            if col in merged_df.columns:
                le = LabelEncoder()
                merged_df[f'{col}_encoded'] = le.fit_transform(merged_df[col].astype(str))
                self.label_encoders[col] = le
                wind_features.append(f'{col}_encoded')
        
        # Prepare X and y
        X = merged_df[wind_features]
        y = merged_df['generation wind offshore']
        
        # Remove outliers
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        X = X[mask]
        y = y[mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Target variable range: {y.min():.2f} - {y.max():.2f}")
        print(f"Average wind offshore generation: {y.mean():.2f}")
        
        return X, y, wind_features
    
    def train_multiple_models(self, X, y):
        """Train multiple models and select the best one"""
        print("Training multiple wind offshore prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred
            }
            
            print(f"{name} - MSE: {mse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")
        
        # Select best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best R² score: {results[best_model_name]['r2']:.3f}")
        
        return results, X_test, y_test
    
    def analyze_wind_patterns(self, X, y):
        """Analyze wind patterns and their relationship with generation"""
        print("Analyzing wind patterns...")
        
        # Create analysis dataframe
        analysis_df = pd.DataFrame({
            'wind_speed': X['wind_speed'],
            'wind_deg': X['wind_deg'],
            'pressure': X['pressure'],
            'temp': X['temp'],
            'generation': y
        })
        
        # Wind speed vs generation
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.scatter(analysis_df['wind_speed'], analysis_df['generation'], alpha=0.5)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Wind Offshore Generation (MW)')
        plt.title('Wind Speed vs Generation')
        
        # Wind direction vs generation
        plt.subplot(2, 3, 2)
        plt.scatter(analysis_df['wind_deg'], analysis_df['generation'], alpha=0.5)
        plt.xlabel('Wind Direction (degrees)')
        plt.ylabel('Wind Offshore Generation (MW)')
        plt.title('Wind Direction vs Generation')
        
        # Pressure vs generation
        plt.subplot(2, 3, 3)
        plt.scatter(analysis_df['pressure'], analysis_df['generation'], alpha=0.5)
        plt.xlabel('Pressure (hPa)')
        plt.ylabel('Wind Offshore Generation (MW)')
        plt.title('Pressure vs Generation')
        
        # Temperature vs generation
        plt.subplot(2, 3, 4)
        plt.scatter(analysis_df['temp'], analysis_df['generation'], alpha=0.5)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Wind Offshore Generation (MW)')
        plt.title('Temperature vs Generation')
        
        # Wind speed distribution
        plt.subplot(2, 3, 5)
        plt.hist(analysis_df['wind_speed'], bins=30, alpha=0.7)
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Frequency')
        plt.title('Wind Speed Distribution')
        
        # Generation distribution
        plt.subplot(2, 3, 6)
        plt.hist(analysis_df['generation'], bins=30, alpha=0.7)
        plt.xlabel('Wind Offshore Generation (MW)')
        plt.ylabel('Frequency')
        plt.title('Generation Distribution')
        
        plt.tight_layout()
        plt.savefig('wind_offshore_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print correlation analysis
        correlation = analysis_df.corr()['generation'].sort_values(ascending=False)
        print("\nFeature correlations with generation:")
        print(correlation)
    
    def plot_results(self, results, X_test, y_test):
        """Plot model comparison and results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        
        axes[0, 0].bar(model_names, r2_scores)
        axes[0, 0].set_title('Model R² Scores Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Best model predictions vs actual
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        y_pred_best = results[best_model_name]['y_pred']
        
        axes[0, 1].scatter(y_test, y_pred_best, alpha=0.5)
        axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Wind Offshore Generation')
        axes[0, 1].set_ylabel('Predicted Wind Offshore Generation')
        axes[0, 1].set_title(f'{best_model_name} - Predictions vs Actual')
        
        # Feature importance for best model
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            top_features = feature_importance.head(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance'])
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].set_xlabel('Importance')
        
        # Residuals plot
        residuals = y_test - y_pred_best
        axes[1, 1].scatter(y_pred_best, residuals, alpha=0.5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals Plot')
        
        plt.tight_layout()
        plt.savefig('wind_offshore_prediction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
    
    def predict_wind_generation(self, weather_data):
        """Predict wind offshore generation for new weather data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please run train_multiple_models first.")
        
        # This method should implement the same preprocessing steps as load_and_preprocess_data
        # For now, returning a placeholder
        return self.model.predict(weather_data)

def main():
    """Main function to run wind offshore prediction"""
    print("=== Wind Offshore Energy Prediction Model ===")
    
    # Initialize predictor
    predictor = WindOffshorePredictor()
    
    # Load and preprocess data
    X, y, features = predictor.load_and_preprocess_data()
    
    # Analyze wind patterns
    predictor.analyze_wind_patterns(X, y)
    
    # Train multiple models
    results, X_test, y_test = predictor.train_multiple_models(X, y)
    
    # Plot results
    predictor.plot_results(results, X_test, y_test)
    
    print("\nWind offshore prediction model training completed!")

if __name__ == "__main__":
    main() 