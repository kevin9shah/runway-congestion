
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, Any, List, Optional

# --- Model Explanation ---
"""
Machine Learning Model for Congestion Prediction:

1.  **Model Choice (Random Forest):** We use a `RandomForestRegressor` because it's a powerful,
    non-linear model that works well with tabular data. It's also "explainable," meaning we can
    easily extract feature importances to understand what drives the predictions.

2.  **Training Process:**
    - The model is trained on a dataset of historical features and their corresponding
      `congestion_index`.
    - Since we are starting without historical data, we generate a synthetic dataset that mimics
      real-world scenarios (e.g., high arrival rates lead to high congestion).
    - In a real implementation, this module would continuously collect data and retrain the model
      periodically to adapt to changing patterns.

3.  **Prediction Flow:**
    a.  Receive the latest engineered features for the local airport.
    b.  Use the trained RandomForest model to generate a `local_congestion_prediction`.
    c.  Receive a summary of congestion from neighboring airports (`neighbor_congestion_summary`).
    d.  Adjust the local prediction based on the neighbors' status. This is the core of the swarm
        logic. A simple but effective method is a weighted average:
        `final_prediction = (local_weight * local_prediction) + (neighbor_weight * avg_neighbor_congestion)`
    e.  This `final_prediction` is the node's output for the current time step.

4.  **Feature Importance:** After training, we can inspect the `feature_importances_` attribute of
    the model. This tells us which features (e.g., `on_ground_aircraft_count` vs.
    `arrival_rate_per_hour`) are the most powerful predictors of congestion.
"""

# --- Model Class ---

class PredictionModel:
    def __init__(self, airport_icao: str, model_dir: str = "models"):
        self.airport_icao = airport_icao
        self.model_path = os.path.join(model_dir, f"model_{self.airport_icao}.joblib")
        self.scaler_path = os.path.join(model_dir, f"scaler_{self.airport_icao}.joblib")
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = [
            "arrival_rate_per_hour",
            "departure_rate_per_hour",
            "on_ground_aircraft_count",
            "circling_aircraft_count",
            "avg_landing_spacing_seconds",
            "arrival_departure_imbalance"
        ]
        
        # Ensure the model directory exists
        os.makedirs(model_dir, exist_ok=True)

    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generates a synthetic dataset for initial training."""
        print(f"Generating synthetic training data for {self.airport_icao}...")
        data = {
            "arrival_rate_per_hour": np.random.uniform(0, 50, n_samples),
            "departure_rate_per_hour": np.random.uniform(0, 50, n_samples),
            "on_ground_aircraft_count": np.random.uniform(0, 60, n_samples),
            "circling_aircraft_count": np.random.uniform(0, 25, n_samples),
            "avg_landing_spacing_seconds": np.random.uniform(60, 240, n_samples),
            "arrival_departure_imbalance": np.random.uniform(0.5, 2.5, n_samples),
        }
        df = pd.DataFrame(data)
        
        # Create a plausible congestion_index based on the features
        df['congestion_index'] = (
            0.4 * (df['arrival_rate_per_hour'] / 50) +
            0.4 * (df['on_ground_aircraft_count'] / 60) +
            0.2 * (df['circling_aircraft_count'] / 25)
        )
        # Add some noise
        df['congestion_index'] += np.random.normal(0, 0.05, n_samples)
        df['congestion_index'] = df['congestion_index'].clip(0, 1)
        
        return df

    def train(self, data: Optional[pd.DataFrame] = None, test_size: float = 0.2):
        """
        Trains the RandomForest model and saves it.
        
        Args:
            data: A DataFrame of historical feature data. If None, synthetic data is generated.
            test_size: The proportion of the dataset to include in the test split.
        """
        if data is None:
            data = self._generate_synthetic_data()

        # Clean data: drop rows with NaN in features, fill target NaNs
        data = data.dropna(subset=self.feature_names)
        data['congestion_index'] = data['congestion_index'].fillna(data['congestion_index'].mean())
        
        if data.empty:
            print("Warning: No valid data available for training. Model not trained.")
            return

        X = data[self.feature_names]
        y = data['congestion_index']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)

        print(f"Training model for {self.airport_icao}...")
        self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)
        print(f"Model training complete. R^2 score: {score:.4f}")

        # Save model and scaler
        self.save_model()
        
    def predict(
        self,
        features: Dict[str, Any],
        neighbor_congestion_summary: Optional[Dict[str, float]] = None,
        swarm_influence: float = 0.2
    ) -> Optional[float]:
        """
        Predicts congestion, adjusting for swarm intelligence.

        Args:
            features: A dictionary of the latest engineered features.
            neighbor_congestion_summary: A dictionary with 'average' and 'trend' of neighbors.
            swarm_influence: The weight (0-1) to give to neighbor data.

        Returns:
            The final predicted congestion index (0-1), or None if prediction fails.
        """
        if self.model is None or self.scaler is None:
            print("Error: Model is not trained or loaded.")
            return None

        try:
            # Prepare input: ensure correct order and handle NaNs
            input_df = pd.DataFrame([features], columns=self.feature_names)
            input_df = input_df.fillna(0) # Simple imputation for missing values
            
            # Scale the input features
            input_scaled = self.scaler.transform(input_df)

            # Local prediction
            local_prediction = self.model.predict(input_scaled)[0]

            # Swarm intelligence adjustment
            final_prediction = local_prediction
            if neighbor_congestion_summary and 'average' in neighbor_congestion_summary:
                avg_neighbor_congestion = neighbor_congestion_summary['average']
                final_prediction = (
                    (1 - swarm_influence) * local_prediction +
                    swarm_influence * avg_neighbor_congestion
                )
            
            return np.clip(final_prediction, 0, 1)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Returns the feature importances of the trained model."""
        if self.model is None:
            return None
        
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save_model(self):
        """Saves the trained model and scaler to disk."""
        if self.model and self.scaler:
            try:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                print(f"Model for {self.airport_icao} saved successfully.")
            except Exception as e:
                print(f"Error saving model for {self.airport_icao}: {e}")

    def load_model(self) -> bool:
        """Loads a pre-trained model and scaler from disk."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                print(f"Model for {self.airport_icao} loaded successfully.")
                return True
            except Exception as e:
                print(f"Error loading model for {self.airport_icao}: {e}")
                return False
        return False

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Prediction Model ---")
    
    mock_airport = "KJFK"
    
    # 1. Initialize and train the model
    model = PredictionModel(mock_airport)
    
    # Try to load a pre-existing model, or train a new one if it doesn't exist
    if not model.load_model():
        print("No pre-trained model found. Training a new one with synthetic data.")
        model.train()

    # 2. Display Feature Importance
    importances = model.get_feature_importance()
    if importances:
        print("\nFeature Importances:")
        # Sort for readability
        sorted_importances = sorted(importances.items(), key=lambda item: item[1], reverse=True)
        for feature, importance in sorted_importances:
            print(f"  - {feature}: {importance:.4f}")

    # 3. Make a prediction
    print("\nMaking a sample prediction...")
    sample_features = {
        "arrival_rate_per_hour": 35.0,
        "departure_rate_per_hour": 30.0,
        "on_ground_aircraft_count": 45,
        "circling_aircraft_count": 10,
        "avg_landing_spacing_seconds": 90.0,
        "arrival_departure_imbalance": 1.17,
    }
    
    # Prediction without swarm data
    local_pred = model.predict(sample_features)
    print(f"Prediction (local only): {local_pred:.4f}")

    # Prediction with swarm data (simulating high congestion at neighbors)
    neighbor_data = {"average": 0.85, "trend": 0.1}
    swarm_pred = model.predict(sample_features, neighbor_congestion_summary=neighbor_data)
    print(f"Prediction (with swarm influence): {swarm_pred:.4f}")

    print("\n--- Prediction Model Test Complete ---")
