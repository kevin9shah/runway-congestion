
import unittest
import time
import sys
import os

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.extractor import extract_features_from_api_data

class TestFeatureExtractor(unittest.TestCase):

    def setUp(self):
        """Set up mock data for testing."""
        self.mock_airport = "KLAX"
        mock_end_time = int(time.time())
        mock_start_time = mock_end_time - 3600

        # Mock data simulates a moderately busy airport.
        self.mock_states = {'states': [
            # 5 aircraft on the ground near KLAX
            ['a8b4a3', 'N628TS', 'United States', None, mock_end_time - 10, -118.4, 33.9, 100, True, 5, 120, 0, None, 150, '1234', False, 0],
            ['a8b4a4', 'DAL123', 'United States', None, mock_end_time - 15, -118.41, 33.91, 100, True, 0, 0, 0, None, 150, '5678', False, 0],
            ['a8b4a5', 'SWA456', 'United States', None, mock_end_time - 20, -118.39, 33.89, 100, True, 10, 240, 0, None, 150, '4321', False, 0],
            ['a8b4a6', 'AAL789', 'United States', None, mock_end_time - 25, -118.42, 33.92, 100, True, 0, 0, 0, None, 150, '8765', False, 0],
            ['a8b4a7', 'UAL101', 'United States', None, mock_end_time - 30, -118.38, 33.88, 100, True, 8, 180, 0, None, 150, '1122', False, 0],
            # 2 aircraft circling nearby
            ['a8b4a8', 'FDX202', 'United States', None, mock_end_time - 40, -118.5, 33.8, 2000, False, 80, 90, -5, None, 2100, '3344', False, 0],
            ['a8b4a9', 'UPS303', 'United States', None, mock_end_time - 50, -118.3, 34.0, 2500, False, 90, 270, 0, None, 2600, '5566', False, 0],
            # 1 aircraft far away (should be ignored)
            ['a8b4aa', 'BAW282', 'United Kingdom', None, mock_end_time - 60, -122.3, 37.6, 10000, False, 250, 270, 0, None, 11000, '7700', False, 0],
        ]}
        self.mock_arrivals = [
            {'icao24': 'a8b4a3', 'firstSeen': mock_start_time + 100, 'lastSeen': mock_end_time - 300},
            {'icao24': 'a8b4a8', 'firstSeen': mock_start_time + 200, 'lastSeen': mock_end_time - 200},
            {'icao24': 'a8b4a9', 'firstSeen': mock_start_time + 300, 'lastSeen': mock_end_time - 100},
        ] * 5  # 15 arrivals
        self.mock_departures = [
            {'icao24': 'a8b4a5', 'firstSeen': mock_start_time + 150, 'lastSeen': mock_end_time - 250},
            {'icao24': 'a8b4a7', 'firstSeen': mock_start_time + 250, 'lastSeen': mock_end_time - 150},
        ] * 6  # 12 departures

    def test_feature_extraction(self):
        """
        Test that features are extracted correctly from mock data.
        """
        features = extract_features_from_api_data(
            self.mock_airport,
            self.mock_states,
            self.mock_arrivals,
            self.mock_departures
        )

        # 1. Test that the function returns a dictionary
        self.assertIsInstance(features, dict)

        # 2. Test the specific calculated values
        self.assertEqual(features['airport_icao'], 'KLAX')
        self.assertEqual(features['arrival_rate_per_hour'], 15.0)
        self.assertEqual(features['departure_rate_per_hour'], 12.0)
        
        # We expect 5 on the ground + 2 circling = 7 in proximity.
        # The 8th aircraft is too far away.
        # From these, 5 should be counted as 'on_ground'.
        self.assertEqual(features['on_ground_aircraft_count'], 5)
        # And 2 should be counted as 'circling'.
        self.assertEqual(features['circling_aircraft_count'], 2)

        # 3. Test the congestion index calculation
        # norm_arrival_rate = min(15.0 / 40.0, 1.0) = 0.375
        # norm_on_ground = min(5.0 / 50.0, 1.0) = 0.1
        # norm_circling = min(2.0 / 20.0, 1.0) = 0.1
        # congestion_index = (0.4 * 0.375) + (0.4 * 0.1) + (0.2 * 0.1)
        #                  = 0.15 + 0.04 + 0.02 = 0.21
        self.assertAlmostEqual(features['congestion_index'], 0.21, places=4)

    def test_empty_data(self):
        """
        Test that the function handles empty or invalid data gracefully.
        """
        # Test with no states data
        features = extract_features_from_api_data(
            self.mock_airport,
            {},
            self.mock_arrivals,
            self.mock_departures
        )
        self.assertIsNone(features)

        # Test with empty lists for flights
        features = extract_features_from_api_data(
            self.mock_airport,
            self.mock_states,
            [],
            []
        )
        self.assertIsInstance(features, dict)
        self.assertEqual(features['arrival_rate_per_hour'], 0)
        self.assertEqual(features['departure_rate_per_hour'], 0)
        self.assertTrue('avg_landing_spacing_seconds' not in features or features['avg_landing_spacing_seconds'] is None or self.isNaN(features['avg_landing_spacing_seconds']))


    def isNaN(self, num):
        return num != num
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
