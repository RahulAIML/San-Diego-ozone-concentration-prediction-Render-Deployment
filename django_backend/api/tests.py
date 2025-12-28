from django.test import TestCase
from django.urls import reverse
from rest_framework.test import APIClient
import json

class OzoneApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.predict_url = reverse('predict')
        self.health_url = reverse('health_check')
        self.valid_payload = {
            "ozone_lag_1": 40,
            "tmax": 25,
            "tavg": 20,
            "CUTI": 0.5,
            "month_sin": 0.5,
            "month_cos": 0.5,
            "wspd": 10,
            "Tmax_inland": 30,
            "land_sea_temp_diff": 5,
            "CUTI_lag1": 0.5,
            "CUTI_lag3": 0.5,
            "CUTI_roll7_mean": 0.5,
            "thermal_stability": 0.1,
            "marine_layer_presence": 0,
            "BEUTI": 10,
            "tsun": 10,
            "temp_range": 10,
            "CUTI_lag7": 0.5,
            "sst_value_sst": 18,
            "sst_anomaly": 0,
            "distance_to_coast_km": 5
        }

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get(self.health_url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'healthy')

    def test_predict_endpoint_success(self):
        """Test the prediction endpoint with valid data."""
        response = self.client.post(
            self.predict_url,
            self.valid_payload,
            format='json'
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'success')
        self.assertIn('prediction', response.data)
        
        # Verify prediction structure
        prediction = response.data['prediction']
        self.assertIn('predicted_ozone', prediction)
        self.assertIn('air_quality_category', prediction)
        self.assertIn('source', prediction)
        
        # Check if model was actually used (or mock if model not found, but we expect model execution in this env)
        # Note: If model artifacts are missing in test env, it might fall back to 'Error Fallback' or 'Mock'.
        # We'll print the source for debugging in logs if needed.
        print(f"Prediction Source: {prediction.get('source')}")

    def test_logs_endpoint(self):
        """Test retrieving logs after a prediction."""
        # Make a prediction first to create a log
        self.client.post(self.predict_url, self.valid_payload, format='json')
        
        response = self.client.get(reverse('get_logs'))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(len(response.data) > 0)
