import json
import unittest
from pathlib import Path


class AnalysisInputsTests(unittest.TestCase):
    def load_data(self):
        return json.loads((Path(__file__).resolve().parents[1] / "fixtures/analysis/input.json").read_text())

    def test_partial_judgments(self):
        data = self.load_data()
        self.assertTrue(len(data['judgments']) < len(data['designs']) * len(data['queries']))

    def test_design_family(self):
        data = self.load_data()
        self.assertTrue(len({row['family_id'] for row in data['designs']}) < len(data['designs']))

    def test_late_candidates(self):
        data = self.load_data()
        self.assertTrue(any(row['first_available_at'] > data['queries'][0]['at'] for row in data['designs']))

    def test_source_versions_and_cutoff(self):
        data = self.load_data()
        self.assertEqual(data["schema_version"], 1)
        self.assertTrue(data["source_versions"])
        self.assertIn("+00:00", data["as_of"])
