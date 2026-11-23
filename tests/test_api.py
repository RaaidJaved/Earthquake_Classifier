"""API Integration Tests for FastAPI Earthquake Classifier EDA Service."""

import unittest
from starlette.testclient import TestClient
from app.main import app


class TestFastAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")

    def test_eda_summary_endpoint(self):
        response = self.client.get("/api/eda/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_records", data)
        self.assertIn("imbalance_ratio", data)
        self.assertIn("significant_count", data)

    def test_eda_class_balance_endpoint(self):
        response = self.client.get("/api/eda/class-balance")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["labels"]), 2)
        self.assertEqual(len(data["counts"]), 2)

    def test_eda_missingness_endpoint(self):
        response = self.client.get("/api/eda/missingness")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("attributes", data)
        self.assertIn("overall", data)
        self.assertIn("minor_class", data)
        self.assertIn("significant_class", data)

    def test_eda_spatial_endpoint(self):
        response = self.client.get("/api/eda/spatial?limit=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)
        if len(data) > 0:
            self.assertIn("latitude", data[0])
            self.assertIn("longitude", data[0])
            self.assertIn("is_significant", data[0])

    def test_html_dashboard_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Earthquake Classifier", response.text)
        self.assertIn("Class Balance Analysis", response.text)


if __name__ == "__main__":
    unittest.main()
