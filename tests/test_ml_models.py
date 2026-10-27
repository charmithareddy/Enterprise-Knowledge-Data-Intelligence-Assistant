"""
Unit tests for the Machine Learning Predictive Analytics engine.
"""

import unittest
from src.models.ml_engine import enterprise_ml

class TestEnterpriseMLEngine(unittest.TestCase):

    def test_revenue_forecaster(self):
        res = enterprise_ml.predict_revenue(
            marketing_spend_k=5000.0,
            sales_reps=40,
            customer_csat=4.5,
            discount_pct=0.05,
            quarter=4
        )
        self.assertIn("forecasted_revenue_m", res)
        self.assertGreater(res["forecasted_revenue_m"], 15.0)
        self.assertEqual(len(res["confidence_interval"]), 2)
        self.assertGreater(res["confidence_interval"][1], res["confidence_interval"][0])

    def test_retention_classifier(self):
        res = enterprise_ml.predict_retention_risk(
            salary=85000.0,
            performance_rating=3.5,
            remote_days=0,
            tenure_years=1.0,
            overtime_hours_monthly=35.0
        )
        self.assertIn("predicted_risk", res)
        self.assertIn(res["predicted_risk"], ["Low", "Medium", "High"])
        self.assertIn("probabilities", res)

if __name__ == "__main__":
    unittest.main()
