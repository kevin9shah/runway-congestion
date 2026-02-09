
# Swarm-Based Distributed Runway Congestion Prediction System

## 1. Project Goal

This project implements a **distributed system for predicting airport runway congestion** using principles of **swarm intelligence**. Each airport in the network acts as an independent, intelligent node that senses its local environment, makes a prediction, and communicates a summarized version of its state to its neighbors.

The system is designed to be **scalable, fault-tolerant, and privacy-preserving**, as nodes do not share raw flight data, only high-level congestion metrics. It uses free and open aviation data from the **OpenSky Network REST API** to simulate a real-world scenario where congestion at one airport can influence its neighbors.

This project serves as a proof-of-concept for a final-year engineering project or a research demonstration in distributed AI systems.

---

## 2. System Architecture

The system is modeled as a network of `AirportNode` objects, where each node represents a major airport. The architecture is decentralized by design.

```
+------------------------+
|   Simulation Manager   |
|       (main.py)        |
+------------------------+
            |
            | Manages & Orchestrates Ticks
            |
+-----------------------------------------------------------------+
|                        AIRPORT NODE (e.g., KJFK)                  |
|-----------------------------------------------------------------|
|                                                                 |
|  +---------------------+  Fetches  +-------------------------+  |
|  |   OpenSky API       +<----------+    Feature Extractor    |  |
|  |   (api/opensky.py)  |           |   (features/extractor.py) |  |
|  +---------------------+---------->+-------------------------+  |
|          ^ Raw Data       | Engineered Features                  |
|          |                v                                      |
|  +---------------------+  Adjusts  +-------------------------+  |
|  | Swarm Communicator  +---------->+  Local ML Model         |  |
|  | (swarm/comm.py)     |           |  (ml/prediction_model.py) |  |
|  +---------------------+<----------+-------------------------+  |
|       ^ | Receives Summary       | Predicts Congestion          |
|       | v                                                       |
|   (To/From Neighbor Nodes)                                      |
|                                                                 |
+-----------------------------------------------------------------+
```

### Key Components:
- **AirportNode**: The core class representing an airport. It encapsulates all other components.
- **OpenSky API Fetcher**: Responsible for fetching live or mock flight data.
- **Feature Extractor**: Processes raw data to calculate features like arrival/departure rates, on-ground aircraft count, and a composite `congestion_index`.
- **Local ML Model**: A `RandomForestRegressor` trained for each airport to predict local congestion based on its features.
- **Swarm Communicator**: Simulates the sharing of summarized data (congestion index and trend) with neighboring nodes.

---

## 3. How to Run the Project

### 3.1. Setup

**Prerequisites:**
- Python 3.7+
- `pip` for package installation

**Installation:**

1.  **Clone the repository (or download the source code):**
    ```bash
    git clone <repository_url>
    cd Backend-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 3.2. Running the Simulation

The main simulation is orchestrated by `main.py`. By default, it runs using **mock data** to ensure consistent and reliable execution without hitting API rate limits.

**To run the simulation:**

```bash
python main.py
```

You will see output logs for each "tick" of the simulation, showing how each airport node updates its state and is influenced by its neighbors.

**Example Output:**
```
=============== STARTING SIMULATION ===============

*************** SIMULATION TICK: 1/3 ***************

--- Updating Node: KJFK ---
[KJFK] Features Extracted: Congestion Index = 0.58
[KJFK] Swarm Summary: Avg Neighbor Congestion = 0.00
[KJFK] Final State: Predicted Congestion = 0.59, Trend = 0.59

--- System State at Tick 1 ---
  ICAO  Congestion  Trend
  KJFK       0.588  0.588
  KLAX       0.608  0.608
  EGLL       0.431  0.431
  ...
```

**To use LIVE data from the OpenSky API:**

1.  **(Optional but Recommended)** Open `src/api/opensky_api.py` and replace the placeholder `OPENSKY_AUTH` credentials with your own if you have an account. This provides higher rate limits.
2.  Open `main.py` and change `use_live_data` to `True`:
    ```python
    # In main.py, at the bottom of the file
    simulation = Simulation(AIRPORT_NETWORK, use_live_data=True)
    ```
3.  Run the simulation again. Note that live data quality can vary, and rate limiting may occur.

---

## 4. Core Concepts Explained

### 4.1. Inferring Runway Congestion

Since the OpenSky API does not provide explicit runway data, we **infer** congestion from several proxies:
- **High Arrival/Departure Rates**: More flights per hour means higher runway usage.
- **High On-Ground Aircraft Count**: A large number of planes on the ground, especially near runways, indicates a queue for takeoff.
- **High Circling Aircraft Count**: Planes holding in the air near an airport signal a queue for landing.
- **Low Landing Spacing**: Aircraft landing very close together in time means the airport is operating at high capacity.

These features are combined into a single, normalized **`congestion_index` (0 to 1)**, which is the primary metric used in the system.

### 4.2. Swarm Intelligence

The "swarm" behavior emerges from how nodes interact:

1.  **Local Sensing**: Each node independently assesses its own congestion.
2.  **Information Sharing**: Nodes do **not** share complex, raw data. They broadcast a very simple message:
    - "My current congestion is `0.85` (very high)."
    - "The trend is `+0.1` (getting worse)."
3.  **Collective Behavior**: A node adjusts its own prediction based on its neighbors' simple messages. If a node's neighbors are all highly congested, it will increase its own congestion prediction, anticipating knock-on delays.

This mimics natural swarms (like bees or ants) where simple, local interactions lead to complex, intelligent global behavior.

---

## 5. System Properties

- **Scalability**: New airports can be added to the `AIRPORT_NETWORK` dictionary in `main.py` without any change to the core logic. The communication overhead remains low as each node only talks to its immediate neighbors.
- **Fault Tolerance**: If a node fails (e.g., cannot fetch data or its process dies), it does not bring down the network. Its neighbors simply stop receiving its summary and continue to operate with the remaining information.
- **Privacy-Preserving**: This is a key design feature. The system demonstrates that useful, system-wide insights can be gained without centralizing sensitive raw data (like specific flight paths or airline information).

---

## 6. Future Scope

This project can be extended in several ways:

- **Real-Time Communication**: Replace the simulated communication with a real network protocol like a REST API on each node or a message broker (e.g., RabbitMQ, NATS).
- **Advanced ML Models**: Implement more sophisticated models, such as LSTMs, to better capture the time-series nature of congestion.
- **Dynamic Neighbor Discovery**: Allow nodes to dynamically discover their most relevant neighbors based on real-time traffic data.
- **Data Persistence**: Store historical feature and prediction data in a time-series database (e.g., InfluxDB, Prometheus) for more robust training and analysis.
- **Visualization Dashboard**: Create a web-based UI to visualize the network graph and the congestion state of each airport in real-time.
- **Track Integration**: Fully integrate the `/tracks` endpoint from the OpenSky API to get more granular data on aircraft movement, improving the accuracy of runway occupancy time estimates.
# runway-congestion
