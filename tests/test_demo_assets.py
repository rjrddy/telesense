"""Validate the published portfolio samples without loading models."""
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'app/frontend/public'


class DemoAssetTests(unittest.TestCase):
    def test_manifest_and_results_are_complete(self):
        samples = json.loads((ROOT / 'demo/manifest.json').read_text())
        self.assertEqual(len(samples), 3)
        self.assertEqual(len({s['id'] for s in samples}), 3)
        for sample in samples:
            with self.subTest(sample=sample['id']):
                for key in ['media', 'audio', 'results']:
                    self.assertTrue((ROOT / sample[key].lstrip('/')).is_file())
                result = json.loads((ROOT / sample['results'].lstrip('/')).read_text())
                self.assertGreater(result['duration'], 0)
                self.assertTrue(result['transcript'])
                self.assertEqual(result['audio_url'], sample['audio'])
                self.assertEqual(result['sample_provenance']['kind'], 'precomputed_model_output')
                self.assertEqual(result['sample_provenance']['visual_emotion_model'], 'unavailable')
                self.assertEqual(result['engine']['name'], 'pretrained')
                for segment in result['transcript']:
                    self.assertGreaterEqual(segment['start_time'], 0)
                    self.assertGreaterEqual(segment['end_time'], segment['start_time'])
                    self.assertLessEqual(segment['end_time'], result['duration'] + .05)
                self.assertEqual(result['statistics']['total_flags'], len(result['combined_flags']))
