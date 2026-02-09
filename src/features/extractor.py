
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple

# --- Constants ---
# These are heuristics and may need tuning based on the specific airport.
# An aircraft is considered 'near' the airport if it's within this lat/lon box.
AIRPORT_PROXIMITY_BOX = 0.5 # Degrees latitude/longitude
# Time in seconds an aircraft can be on a runway before it's considered 'stuck'.
RUNWAY_OCCUPANCY_THRESHOLD = 180

# --- Feature Engineering Explanation ---
"""
How Runway Congestion is INFERRED Without Explicit Runway Data:

The OpenSky Network API does not provide direct runway usage data. We infer congestion by treating the
airport's airspace and ground as a system and measuring aircraft flow and density.

1.  **Arrival/Departure Rates:** High, sustained rates imply high runway utilization. We measure this
    by counting the number of flights landing and taking off in a given time window (e.g., 1 hour).

2.  **Aircraft On-Ground Count:** A key proxy for takeoff queues. We count aircraft that are on the
    ground but not parked at a gate (inferred by position). We can get this from the '/states/all'
    endpoint, filtering by the airport's bounding box and `on_ground=True`.

3.  **Circling Aircraft Count:** A proxy for landing queues. We identify aircraft that are close to
    the airport, not on the ground, and have a low velocity. This suggests they are in a holding
    pattern.

4.  **Average Landing Spacing:** By analyzing the time difference between consecutive arrivals, we
    can gauge how tightly packed landing operations are. Smaller spacing indicates higher throughput
    and potential stress on the system.

5.  **Delay Proxy (Arrival/Departure Imbalance):** A significant imbalance between arrivals and
    departures can indicate that the airport is prioritizing one over the other to clear a backlog,
a classic sign of congestion management.

6.  **Congestion Index (Composite Metric):** This is a normalized (0-1) value that combines the
    above features into a single, understandable metric. It's a weighted average where a value of 1.0
    represents peak congestion and 0.0 represents an idle airport.
"""

# --- Feature Extraction Functions ---

def get_airport_icao_and_position(airport_code: str) -> Optional[Tuple[str, float, float]]:
    """
    A simple lookup for airport ICAO codes and their approximate lat/lon.
    In a real system, this would come from a comprehensive database.
    """
    # Source: OurAirports.com, filtered for major hubs
    airport_db = {
        "KLAX": ("KLAX", 33.9425, -118.4081),
        "KJFK": ("KJFK", 40.6398, -73.7789),
        "EGLL": ("EGLL", 51.4706, -0.4619),
        "EDDF": ("EDDF", 50.0333, 8.5706),
        "LFPG": ("LFPG", 49.0097, 2.5479),
        "RJTT": ("RJTT", 35.5523, 139.7797),
    }
    return airport_db.get(airport_code.upper())


def extract_features_from_api_data(
    airport_code: str,
    states_data: Dict[str, Any],
    arrivals_data: List[Dict[str, Any]],
    departures_data: List[Dict[str, Any]],
    time_window_hours: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Processes raw API data to calculate congestion features for a single airport.

    Args:
        airport_code: The ICAO code of the airport.
        states_data: JSON response from the OpenSky '/states/all' endpoint.
        arrivals_data: JSON response from the '/flights/arrival' endpoint.
        departures_data: JSON response from the '/flights/departure'endpoint.
        time_window_hours: The duration of the observation window in hours.

    Returns:
        A dictionary of calculated features or None if data is insufficient.
    """
    airport_info = get_airport_icao_and_position(airport_code)
    if not airport_info or not states_data or 'states' not in states_data:
        return None

    _, airport_lat, airport_lon = airport_info
    time_window_sec = time_window_hours * 3600

    # --- Feature 1: Arrival and Departure Rates ---
    arrival_rate_per_hour = len(arrivals_data) / time_window_hours
    departure_rate_per_hour = len(departures_data) / time_window_hours

    # --- Features 2 & 3: On-Ground and Circling Aircraft ---
    on_ground_count = 0
    circling_count = 0
    if states_data['states']:
        states_df = pd.DataFrame(states_data['states'], columns=[
            "icao24", "callsign", "origin_country", "time_position", "last_contact",
            "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
            "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk", "spi",
            "position_source"
        ])
        
        # Filter for aircraft within the airport's proximity box
        lat_min, lat_max = airport_lat - AIRPORT_PROXIMITY_BOX, airport_lat + AIRPORT_PROXIMITY_BOX
        lon_min, lon_max = airport_lon - AIRPORT_PROXIMITY_BOX, airport_lon + AIRPORT_PROXIMITY_BOX
        
        airport_area_df = states_df[
            (states_df['latitude'].between(lat_min, lat_max)) &
            (states_df['longitude'].between(lon_min, lon_max))
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        on_ground_count = airport_area_df['on_ground'].sum()
        
        # Identify circling aircraft: low speed, low altitude, near airport
        circling_df = airport_area_df[
            (~airport_area_df['on_ground']) &
            (airport_area_df['velocity'] < 100) & # meters/sec, approx < 200 knots
            (airport_area_df['baro_altitude'].fillna(0) < 3000) # meters, approx < 10,000 ft
        ]
        circling_count = len(circling_df)

    # --- Feature 4: Average Landing Spacing ---
    avg_landing_spacing_sec = np.nan
    if len(arrivals_data) > 1:
        arrival_times = sorted([flight['lastSeen'] for flight in arrivals_data])
        spacings = np.diff(arrival_times)
        avg_landing_spacing_sec = np.mean(spacings) if len(spacings) > 0 else np.nan

    # --- Feature 5: Delay Proxy (Imbalance) ---
    # Simple ratio; a value > 1 means more arrivals than departures.
    arrival_departure_imbalance = len(arrivals_data) / (len(departures_data) + 1e-6)

    # --- Feature 6: Congestion Index (Composite) ---
    # Normalize features to a 0-1 scale before combining.
    # These max values are estimates and should be tuned based on real data.
    norm_arrival_rate = min(arrival_rate_per_hour / 40.0, 1.0) # Max 40 arrivals/hr
    norm_on_ground = min(on_ground_count / 50.0, 1.0) # Max 50 aircraft on ground
    norm_circling = min(circling_count / 20.0, 1.0) # Max 20 aircraft circling

    # Weights can be adjusted based on their perceived importance.
    congestion_index = np.average(
        [norm_arrival_rate, norm_on_ground, norm_circling],
        weights=[0.4, 0.4, 0.2]
    )

    return {
        "timestamp": int(time.time()),
        "airport_icao": airport_code,
        "arrival_rate_per_hour": arrival_rate_per_hour,
        "departure_rate_per_hour": departure_rate_per_hour,
        "on_ground_aircraft_count": on_ground_count,
        "circling_aircraft_count": circling_count,
        "avg_landing_spacing_seconds": avg_landing_spacing_sec,
        "arrival_departure_imbalance": arrival_departure_imbalance,
        "congestion_index": congestion_index
    }

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Feature Extractor ---")

    # This example requires mock data since we can't guarantee API availability
    # in a static script.
    mock_airport = "KLAX"
    mock_end_time = int(time.time())
    mock_start_time = mock_end_time - 3600

    # Mock data simulates a moderately busy airport.
    mock_states = {'states': [
        # 5 aircraft on the ground near KLAX
        ['a8b4a3', 'N628TS', 'United States', None, mock_end_time - 10, -118.4, 33.9, 100, True, 5, 120, 0, None, 150, '1234', False, 0],
        ['a8b4a4', 'DAL123', 'United States', None, mock_end_time - 15, -118.41, 33.91, 100, True, 0, 0, 0, None, 150, '5678', False, 0],
        ['a8b4a5', 'SWA456', 'United States', None, mock_end_time - 20, -118.39, 33.89, 100, True, 10, 240, 0, None, 150, '4321', False, 0],
        ['a8b4a6', 'AAL789', 'United States', None, mock_end_time - 25, -118.42, 33.92, 100, True, 0, 0, 0, None, 150, '8765', False, 0],
        ['a8b4a7', 'UAL101', 'United States', None, mock_end_time - 30, -118.38, 33.88, 100, True, 8, 180, 0, None, 150, '1122', False, 0],
        # 2 aircraft circling nearby
        ['a8b4a8', 'FDX202', 'United States', None, mock_end_time - 40, -118.5, 33.8, 2000, False, 80, 90, -5, None, 2100, '3344', False, 0],
        ['a8b4a9', 'UPS303', 'United States', None, mock_end_time - 50, -118.3, 34.0, 2500, False, 90, 270, 0, None, 2600, '5566', False, 0],
    ]}
    mock_arrivals = [
        {'icao24': 'a8b4a3', 'firstSeen': mock_start_time + 100, 'lastSeen': mock_end_time - 300},
        {'icao24': 'a8b4a8', 'firstSeen': mock_start_time + 200, 'lastSeen': mock_end_time - 200},
        {'icao24': 'a8b4a9', 'firstSeen': mock_start_time + 300, 'lastSeen': mock_end_time - 100},
    ] * 5 # Simulate 15 arrivals in the hour
    mock_departures = [
        {'icao24': 'a8b4a5', 'firstSeen': mock_start_time + 150, 'lastSeen': mock_end_time - 250},
        {'icao24': 'a8b4a7', 'firstSeen': mock_start_time + 250, 'lastSeen': mock_end_time - 150},
    ] * 6 # Simulate 12 departures

    print(f"\nExtracting features for {mock_airport} using mock data...")
    features = extract_features_from_api_data(
        mock_airport,
        mock_states,
        mock_arrivals,
        mock_departures
    )

    if features:
        print("Successfully extracted features:")
        import json
        print(json.dumps(features, indent=4))
    else:
        print("Failed to extract features.")
    
    print("\n--- Feature Extractor Test Complete ---")
