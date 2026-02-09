
import json
import threading
import time
from flask import Flask, jsonify, render_template

from main import Simulation

# --- Globals ---
app = Flask(__name__)
# Use a lock to ensure thread-safe access to the simulation data
simulation_lock = threading.Lock()
# This dictionary will hold the latest state of our simulation
simulation_data_store = {}
# This dictionary will hold the detailed state for each node
simulation_details_store = {}

# --- Background Simulation Thread ---

def run_simulation_background():
    """
    This function runs the simulation in a background thread and
    continuously updates the global data store with the latest state.
    """
    global simulation_data_store, simulation_details_store
    
    # Define the airport network
    airport_network = {
        "KJFK": ["KBOS", "KORD", "EGLL"],
        "KLAX": ["KSFO", "KORD", "RJTT"],
        "EGLL": ["LFPG", "EDDF", "KJFK"],
        "EDDF": ["EGLL", "LFPG", "LOWW"],
        "LFPG": ["EGLL", "EDDF", "LEMD"],
        "RJTT": ["ZBAA", "VHHH", "KLAX"],
    }
    
    # Initialize the simulation
    simulation = Simulation(airport_network, use_live_data=False)
    
    tick_count = 0
    while True:
        tick_count += 1
        print(f"--- Running Simulation Tick: {tick_count} ---")
        
        # Run one tick of the simulation
        simulation.run(num_ticks=1)
        
        # Update the shared data stores with the latest results
        with simulation_lock:
            for icao, node in simulation.nodes.items():
                summary = node.get_congestion_summary()
                if summary:
                    simulation_data_store[icao] = summary

                details = node.get_details()
                if details:
                    simulation_details_store[icao] = details

        # Wait for a few seconds before the next tick
        time.sleep(15) # Update every 15 seconds

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main dashboard UI."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Provides the latest simulation data as a JSON API endpoint."""
    with simulation_lock:
        # Return a copy of the data
        return jsonify(simulation_data_store)

@app.route('/api/details/<icao>')
def get_details(icao: str):
    """Provides detailed data for a specific airport node."""
    with simulation_lock:
        details = simulation_details_store.get(icao)
        if details:
            return jsonify(details)
        else:
            return jsonify({"error": "No details found for the specified ICAO."}), 404

# --- Main Execution ---

if __name__ == '__main__':
    # Start the simulation in a background thread
    simulation_thread = threading.Thread(target=run_simulation_background, daemon=True)
    simulation_thread.start()
    
    # Start the Flask web server
    # use_reloader=False is important to prevent running the background thread twice
    app.run(host="0.0.0.0", port=5001, debug=True, use_reloader=False)
