from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List, Optional
import numpy as np

# Use a forward reference for the type hint to avoid circular import
if TYPE_CHECKING:
    from main import AirportNode # This will be our main node class

# --- Swarm Communication Explanation ---
"""
Swarm Communication Logic:

This module simulates the communication protocol in the distributed system. In a real-world
deployment, this would be a lightweight network layer (e.g., a REST API endpoint on each node
or a shared message bus). For this simulation, it's a direct method call.

1.  **Data Shared (Privacy-Preserving):** Nodes do NOT share raw flight data. They only share a
    small, anonymized summary of their status:
    - `congestion_index`: The current predicted congestion (0-1).
    - `short_term_trend`: Is congestion rising or falling?
    - `timestamp`: When the data was generated.

2.  **Neighbor Discovery:** In our simulation, neighbor relationships are pre-defined. In a dynamic
    system, nodes could broadcast their existence or use a central registry for discovery. Here,
    the main application acts as the registry.

3.  **Communication Flow:**
    a. An `AirportNode` (`node_self`) wants to get data from its neighbors.
    b. It calls `get_neighbor_summary()`, passing in a reference to the global list of all nodes
       and a list of its pre-defined neighbor ICAOs.
    c. The communicator iterates through the neighbor ICAOs, finds the corresponding `AirportNode`
       objects in the global list, and accesses their public `get_congestion_summary()` method.
    d. It aggregates the data from all neighbors (e.g., by averaging) and returns a simple
       summary dictionary.
    e. This summary is then used by `node_self` to adjust its own local prediction.
"""

# --- Communicator Class ---

class SwarmCommunicator:
    def __init__(self, node_icao: str):
        self.node_icao = node_icao

    def get_neighbor_summary(
        self, 
        neighbor_icaos: List[str],
        all_nodes: Dict[str, "AirportNode"]
    ) -> Optional[Dict[str, float]]:
        """
        Gathers and summarizes congestion data from neighboring nodes.

        Args:
            neighbor_icaos: A list of ICAO codes for the neighboring airports.
            all_nodes: A dictionary mapping all ICAO codes to their AirportNode instances.

        Returns:
            A dictionary containing the 'average' congestion and 'trend' of neighbors,
            or None if no neighbor data is available.
        """
        neighbor_congestion_indices: List[float] = []
        neighbor_trends: List[float] = []

        for icao in neighbor_icaos:
            neighbor_node = all_nodes.get(icao)
            if neighbor_node:
                # Access the public summary from the neighbor
                summary = neighbor_node.get_congestion_summary()
                if summary:
                    neighbor_congestion_indices.append(summary['congestion_index'])
                    neighbor_trends.append(summary['short_term_trend'])
        
        if not neighbor_congestion_indices:
            return None

        # Aggregate the data
        avg_congestion = np.mean(neighbor_congestion_indices)
        avg_trend = np.mean(neighbor_trends)

        return {
            "average": avg_congestion,
            "trend": avg_trend
        }

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing Swarm Communicator ---")
    
    # This requires a mock setup of the AirportNode class, which we'll define in main.py.
    # We create a simplified version here for testing purposes.
    class MockAirportNode:
        def __init__(self, icao: str, congestion_index: float, trend: float):
            self.icao = icao
            self._congestion_index = congestion_index
            self._trend = trend
            self.communicator = SwarmCommunicator(self.icao)

        def get_congestion_summary(self) -> Dict[str, float]:
            return {
                "congestion_index": self._congestion_index,
                "short_term_trend": self._trend,
                "timestamp": 1234567890
            }

        def query_neighbors(self, all_nodes_map: Dict[str, 'MockAirportNode'], neighbors: List[str]):
            print(f"\nNode {self.icao} (Congestion: {self._congestion_index:.2f}) is querying its neighbors...")
            summary = self.communicator.get_neighbor_summary(neighbors, all_nodes_map)
            if summary:
                print(f"  -> Neighbor summary received: Average Congestion = {summary['average']:.2f}, Average Trend = {summary['trend']:.2f}")
            else:
                print("  -> No neighbor data found.")
    
    # 1. Create a few mock airport nodes
    node_jfk = MockAirportNode("KJFK", 0.75, 0.1)  # High congestion, rising
    node_bos = MockAirportNode("KBOS", 0.40, -0.05) # Moderate congestion, falling
    node_ord = MockAirportNode("KORD", 0.85, 0.2)  # Very high congestion, rising fast
    node_lax = MockAirportNode("KLAX", 0.60, 0.0)   # High-ish congestion, stable

    # 2. Create the "global" registry of all nodes
    all_nodes_registry = {
        "KJFK": node_jfk,
        "KBOS": node_bos,
        "KORD": node_ord,
        "KLAX": node_lax,
    }

    # 3. Define neighbors for JFK
    jfk_neighbors = ["KBOS", "KORD"] # Boston and Chicago are neighbors

    # 4. Simulate JFK querying its neighbors
    node_jfk.query_neighbors(all_nodes_registry, jfk_neighbors)
    
    # Expected output for JFK's query:
    # Average Congestion = (0.40 + 0.85) / 2 = 0.625
    # Average Trend = (-0.05 + 0.2) / 2 = 0.075

    # 5. Simulate LAX querying its neighbors (let's say only ORD is its neighbor in this test)
    lax_neighbors = ["KORD"]
    node_lax.query_neighbors(all_nodes_registry, lax_neighbors)

    # Expected output for LAX's query:
    # Average Congestion = 0.85
    # Average Trend = 0.2
    
    print("\n--- Swarm Communicator Test Complete ---")
