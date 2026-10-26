"""Check the deployable camera weights and their provenance without a webcam."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'app/frontend/public/models/face-expression'


class CameraAssetTests(unittest.TestCase):
    def test_only_detection_and_expression_weights_are_published(self):
        source = json.loads((MODEL_DIR / 'SOURCE.json').read_text())
        self.assertEqual(source['enabled'], ['tinyFaceDetector', 'faceExpressionNet'])
        self.assertEqual({p.name for p in MODEL_DIR.glob('*.bin')}, {
            'tiny_face_detector_model.bin', 'face_expression_model.bin',
        })
        package = json.loads((ROOT / 'app/frontend/package.json').read_text())
        self.assertEqual(package['dependencies'][source['package']], source['version'])
        for name, checksum in source['files'].items():
            self.assertEqual(hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest(), checksum)
            if name.endswith('manifest.json'):
                for group in json.loads((MODEL_DIR / name).read_text()):
                    for shard in group['paths']:
                        self.assertTrue((MODEL_DIR / shard).is_file())

    def test_runtime_and_upstream_notices_are_available(self):
        for name in ['LICENSE.txt', 'UPSTREAM-LICENSE.txt', 'TENSORFLOW-LICENSE.txt']:
            self.assertGreater((MODEL_DIR / name).stat().st_size, 500)
