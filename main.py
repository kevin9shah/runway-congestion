
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

# --- Import Project Modules ---
from src.api import opensky_api
from src.features.extractor import extract_features_from_api_data, get_airport_icao_and_position
from src.ml.prediction_model import PredictionModel
from src.swarm.communication import SwarmCommunicator

# --- Main Airport Node Class ---

class AirportNode:
    """
    Represents a single airport in the distributed network.
    It encapsulates all logic for data fetching, feature engineering,
    prediction, and swarm communication.
    """
    def __init__(self, airport_code: str, neighbors: List[str], use_live_data: bool = False):
        self.icao = get_airport_icao_and_position(airport_code)[0]
        self.neighbors = neighbors
        self.use_live_data = use_live_data
        
        # Instantiate components
        self.model = PredictionModel(self.icao)
        self.communicator = SwarmCommunicator(self.icao)
        
        # State variables
        self.history: List[Dict[str, Any]] = []
        self.current_features: Optional[Dict[str, Any]] = None
        self.current_prediction: float = 0.0
        self.short_term_trend: float = 0.0
        self.raw_data: Optional[Dict[str, Any]] = None
        self.neighbor_summary: Optional[Dict[str, Any]] = None

    def initialize_model(self):
        """Loads a pre-trained model or trains a new one."""
        if not self.model.load_model():
            print(f"No pre-trained model for {self.icao}. Training a new one...")
            self.model.train()
        else:
            print(f"Pre-trained model for {self.icao} loaded.")

    def update(self, all_nodes: Dict[str, 'AirportNode']):
        """
        Executes a single simulation tick for this node.
        1. Fetch data.
        2. Extract features.
        3. Make local prediction.
        4. Query neighbors and adjust prediction.
        5. Update state.
        """
        print(f"\n--- Updating Node: {self.icao} ---")
        
        # 1. Fetch data (using mock data for this simulation)
        self.raw_data = self._get_data_for_features()
        if not self.raw_data:
            print(f"[{self.icao}] Skipping update: Could not retrieve data.")
            return

        # 2. Extract features
        self.current_features = extract_features_from_api_data(
            airport_code=self.icao,
            **self.raw_data
        )
        if not self.current_features:
            print(f"[{self.icao}] Skipping update: Feature extraction failed.")
            return
        
        print(f"[{self.icao}] Features Extracted: Congestion Index = {self.current_features['congestion_index']:.2f}")

        # 3. Query neighbors (before making final prediction)
        self.neighbor_summary = self.communicator.get_neighbor_summary(self.neighbors, all_nodes)
        if self.neighbor_summary:
            print(f"[{self.icao}] Swarm Summary: Avg Neighbor Congestion = {self.neighbor_summary['average']:.2f}")

        # 4. Make prediction (local, then adjusted by swarm)
        predicted_congestion = self.model.predict(
            features=self.current_features,
            neighbor_congestion_summary=self.neighbor_summary
        )
        if predicted_congestion is None:
            print(f"[{self.icao}] Skipping update: Prediction failed.")
            return

        # 5. Update state
        self._update_state(predicted_congestion)
        
        print(f"[{self.icao}] Final State: Predicted Congestion = {self.current_prediction:.2f}, Trend = {self.short_term_trend:.2f}")

    def _get_data_for_features(self) -> Optional[Dict[str, Any]]:
        """
        Fetches data required for feature extraction.
        Switches between live API calls and mock data.
        """
        if self.use_live_data:
            # --- LIVE DATA ---
            end_time = int(time.time())
            start_time = end_time - 3600
            
            states = opensky_api.get_states()
            # In a real system, we'd need to filter states by airport proximity here
            
            arrivals = opensky_api.get_flights_for_airport(self.icao, start_time, end_time, is_arrival=True)
            departures = opensky_api.get_flights_for_airport(self.icao, start_time, end_time, is_arrival=False)
            
            if states is None or arrivals is None or departures is None:
                return None
            return {"states_data": states, "arrivals_data": arrivals, "departures_data": departures}
        
        else:
            # --- MOCK DATA ---
            # Generate plausible random data for each tick to make the simulation dynamic
            num_arrivals = np.random.randint(10, 40)
            num_departures = np.random.randint(10, 40)
            on_ground = np.random.randint(15, 50)
            circling = np.random.randint(2, 15)
            
            mock_states = {'states': [
                ['', '', '', None, 0, 0, 0, 0, True, 0, 0, 0, None, 0, '', False, 0]] * on_ground +
                [['', '', '', None, 0, 0, 0, 2000, False, 80, 0, 0, None, 0, '', False, 0]] * circling
            }
            mock_arrivals = [{'icao24': '', 'firstSeen': 0, 'lastSeen': i * 100} for i in range(num_arrivals)]
            mock_departures = [{'icao24': '', 'firstSeen': 0, 'lastSeen': i * 100} for i in range(num_departures)]
            
            return {"states_data": mock_states, "arrivals_data": mock_arrivals, "departures_data": mock_departures}

    def _update_state(self, new_prediction: float):
        """Updates the node's internal state and history."""
        if self.history:
            last_prediction = self.history[-1]['prediction']
            self.short_term_trend = new_prediction - last_prediction
        else:
            self.short_term_trend = 0.0
        
        self.current_prediction = new_prediction
        
        state_snapshot = {
            "timestamp": int(time.time()),
            "features": self.current_features,
            "prediction": self.current_prediction,
            "trend": self.short_term_trend
        }
        self.history.append(state_snapshot)
    
    def get_congestion_summary(self) -> Optional[Dict[str, Any]]:
        """Provides the public summary for neighbors to consume."""
        if not self.history:
            return None
        return {
            "congestion_index": self.current_prediction,
            "short_term_trend": self.short_term_trend,
            "timestamp": self.history[-1]["timestamp"]
        }

    def get_details(self) -> Optional[Dict[str, Any]]:
        """Provides a detailed snapshot of the node's current state for the API."""
        if not self.history:
            return None
        
        # Sanitize raw_data by summarizing large lists
        sanitized_raw_data = {}
        if self.raw_data:
            for key, value in self.raw_data.items():
                if isinstance(value, dict) and 'states' in value and isinstance(value['states'], list):
                    sanitized_raw_data[key] = f"{len(value['states'])} states entries"
                elif isinstance(value, list):
                    sanitized_raw_data[key] = f"{len(value)} entries"
                else:
                    sanitized_raw_data[key] = value

        return {
            "icao": self.icao,
            "timestamp": self.history[-1]["timestamp"],
            "prediction": self.current_prediction,
            "short_term_trend": self.short_term_trend,
            "raw_data_summary": sanitized_raw_data,
            "extracted_features": self.current_features,
            "swarm_communication": self.neighbor_summary,
        }



# --- Simulation Orchestrator ---

class Simulation:
    def __init__(self, airport_definitions: Dict[str, List[str]], use_live_data: bool = False):
        print("Initializing Swarm Simulation...")
        self.nodes: Dict[str, AirportNode] = {
            code: AirportNode(code, neighbors, use_live_data)
            for code, neighbors in airport_definitions.items()
        }
        # Initialize all models
        for node in self.nodes.values():
            node.initialize_model()

    def run(self, num_ticks: int = 5):
        """Runs the simulation for a specified number of steps."""
        print(f"\n{'='*20} STARTING SIMULATION {'='*20}")
        for i in range(num_ticks):
            print(f"\n{'*'*15} SIMULATION TICK: {i+1}/{num_ticks} {'*'*15}")
            
            # In a real parallel system, these would run concurrently
            for icao, node in self.nodes.items():
                node.update(self.nodes)
            
            # Log the state of the entire system for this tick
            self._log_system_state(i + 1)
            
            if i < num_ticks - 1:
                time.sleep(1) # Pause between ticks for readability
        
        print(f"\n{'='*22} SIMULATION COMPLETE {'='*22}")
        self.print_summary()

    def _log_system_state(self, tick_number: int):
        """Prints a summary of all node states for the current tick."""
        print(f"\n--- System State at Tick {tick_number} ---")
        log_data = []
        for icao, node in self.nodes.items():
            summary = node.get_congestion_summary()
            if summary:
                log_data.append({
                    "ICAO": icao,
                    "Congestion": f"{summary['congestion_index']:.3f}",
                    "Trend": f"{summary['short_term_trend']:.3f}"
                })
        
        if log_data:
            df = pd.DataFrame(log_data)
            print(df.to_string(index=False))

    def print_summary(self):
        """Prints a final summary of the simulation."""
        print("\n--- Final Simulation Summary ---")
        for icao, node in self.nodes.items():
            print(f"\nNode: {icao}")
            # Convert history to DataFrame for nice printing
            if node.history:
                history_df = pd.DataFrame([{
                    'Tick': i + 1,
                    'Congestion': h['prediction'],
                    'Trend': h['trend'],
                    'Arrivals/hr': h['features']['arrival_rate_per_hour'],
                    'On_Ground': h['features']['on_ground_aircraft_count']
                } for i, h in enumerate(node.history)])
                print(history_df.round(3).to_string(index=False))


# --- Main Execution Block ---

if __name__ == "__main__":
    # Define the network topology.
    # Airports and their direct neighbors. This can be based on flight routes.
    AIRPORT_NETWORK = {
        "KJFK": ["KBOS", "KORD", "EGLL"], # New York
        "KLAX": ["KSFO", "KORD", "RJTT"], # Los Angeles
        "EGLL": ["LFPG", "EDDF", "KJFK"], # London
        "EDDF": ["EGLL", "LFPG", "LOWW"], # Frankfurt
        "LFPG": ["EGLL", "EDDF", "LEMD"], # Paris
        "RJTT": ["ZBAA", "VHHH", "KLAX"], # Tokyo
    }

    # Set 'use_live_data' to True to try with the OpenSky API.
    # Note: Live data is rate-limited and may not always be available.
    # The default mock data is more reliable for this demonstration.
    simulation = Simulation(AIRPORT_NETWORK, use_live_data=False)

    # Run the simulation for a few ticks
    simulation.run(num_ticks=3)
