"""Download only the five attributed public sample sources; verify source bytes.

Run before build_demo_samples.py when the ignored test_data sources are absent.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = json.loads((ROOT / 'scripts/demo_sample_sources.json').read_text())


def main():
    parquet_path = ROOT / 'test_data/external/librispeech/validation.parquet'
    rows = None
    for sample in SAMPLES:
        source = ROOT / 'test_data' / sample['source']
        source.parent.mkdir(parents=True, exist_ok=True)
        attribution = sample['attribution']
        if not source.exists():
            if attribution['dataset'] == 'LibriSpeech':
                if not parquet_path.exists():
                    urllib.request.urlretrieve(attribution['source_url'], parquet_path)
                if rows is None:
                    import pyarrow.parquet as pq
                    rows = {r['id']: r for r in pq.read_table(parquet_path).to_pylist()}
                source.write_bytes(rows[attribution['clip_id']]['audio']['bytes'])
            else:
                urllib.request.urlretrieve(attribution['source_url'], source)
        if hashlib.sha256(source.read_bytes()).hexdigest() != attribution['sha256']:
            raise RuntimeError(f'Source checksum mismatch: {sample["id"]}')
        print(f'Verified {sample["id"]}')


if __name__ == '__main__':
    main()
