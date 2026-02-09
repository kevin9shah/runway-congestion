
import requests
import time
from typing import Optional, Dict, Any, List

# --- Constants ---
OPENSKY_API_URL = "https://opensky-network.org/api"
# In a real application, this would be handled securely, e.g., via environment variables
# and a proper OAuth2 flow.
OPENSKY_AUTH = ("user", "password") # Replace with your OpenSky credentials if you have them

# --- API Wrapper Functions ---

def get_states(icao24: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetches the state vectors for one or more aircraft.

    Args:
        icao24: The ICAO24 address of a single aircraft. If None, fetches all states.

    Returns:
        A dictionary containing the API response or None if an error occurs.
    """
    url = f"{OPENSKY_API_URL}/states/all"
    params = {}
    if icao24:
        params["icao24"] = icao24

    try:
        response = requests.get(url, auth=OPENSKY_AUTH, params=params, timeout=15)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching states from OpenSky API: {e}")
        return None

def get_flights_for_airport(
    airport_icao: str,
    start_time: int,
    end_time: int,
    is_arrival: bool = True
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetches arrival or departure flights for a given airport within a time interval.

    Args:
        airport_icao: The ICAO code of the airport.
        start_time: The start of the time interval (Unix timestamp).
        end_time: The end of the time interval (Unix timestamp).
        is_arrival: True to fetch arrivals, False to fetch departures.

    Returns:
        A list of flight dictionaries or None if an error occurs.
    """
    endpoint = "arrivals" if is_arrival else "departures"
    url = f"{OPENSKY_API_URL}/flights/{endpoint}"
    params = {
        "airport": airport_icao,
        "begin": start_time,
        "end": end_time,
    }

    try:
        response = requests.get(url, auth=OPENSKY_AUTH, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {endpoint} for {airport_icao} from OpenSky API: {e}")
        return None

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing OpenSky API Wrapper ---")

    # --- Test 1: Get all states (will be a large amount of data) ---
    print("\nFetching all states (limit 1 for preview)...")
    all_states = get_states()
    if all_states and all_states.get('states'):
        print(f"Successfully fetched states for {len(all_states['states'])} aircraft.")
        print("Example state vector:")
        print(all_states['states'][0])
    else:
        print("Failed to fetch all states or no aircraft are airborne.")

    time.sleep(2) # Respect API rate limits

    # --- Test 2: Get states for a specific aircraft (example ICAO24) ---
    # Note: This ICAO24 might not be valid or airborne.
    example_icao24 = "a8b4a3"
    print(f"\nFetching states for specific aircraft: {example_icao24}...")
    specific_states = get_states(icao24=example_icao24)
    if specific_states and specific_states.get('states'):
        print("Successfully fetched state vector:")
        print(specific_states['states'][0])
    else:
        print(f"Could not fetch states for {example_icao24}. It may not be airborne.")

    time.sleep(2)

    # --- Test 3: Get arrivals for a major airport (e.g., KLAX) ---
    # We look at a 1-hour window from 2 hours ago to 1 hour ago.
    airport_to_check = "KLAX"
    end_ts = int(time.time())
    start_ts = end_ts - 3600 # 1 hour ago
    print(f"\nFetching arrivals for {airport_to_check} in the last hour...")
    arrivals = get_flights_for_airport(airport_to_check, start_ts, end_ts, is_arrival=True)
    if arrivals is not None:
        print(f"Found {len(arrivals)} arrivals for {airport_to_check}.")
        if arrivals:
            print("Example arrival flight:")
            print(arrivals[0])
    else:
        print(f"Failed to fetch arrivals for {airport_to_check}.")

    time.sleep(2)

    # --- Test 4: Get departures for the same airport ---
    print(f"\nFetching departures for {airport_to_check} in the last hour...")
    departures = get_flights_for_airport(airport_to_check, start_ts, end_ts, is_arrival=False)
    if departures is not None:
        print(f"Found {len(departures)} departures for {airport_to_check}.")
        if departures:
            print("Example departure flight:")
            print(departures[0])
    else:
        print(f"Failed to fetch departures for {airport_to_check}.")

    print("\n--- API Wrapper Test Complete ---")
