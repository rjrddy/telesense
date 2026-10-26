"""Build published sample clips and real saved results from approved test_data.

Run from repo root: HF_HOME=app/storage/hf-cache .venv/bin/python scripts/build_demo_samples.py
All results come from the app's pretrained CPU pipeline; no fabricated predictions.
"""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import uuid
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['TELESENSE_DEVICE'] = 'cpu'
os.environ.setdefault('HF_HOME', str(ROOT / 'app/storage/hf-cache'))

SAMPLES = [
    dict(id='conversation', title='Dialogue clip', subtitle='Short video · audio + transcript',
         description='A brief dialogue clip for exploring synchronized playback and model signals.',
         source='76fdb89f-aa0b-451a-9784-8195744314c4/video.mp4', start=0, length=8.8, type='video'),
    dict(id='listening', title='Interview excerpt 01', subtitle='Audio excerpt · speech + language',
         description='A longer listening sample to explore how vocal and language signals change over time.',
         source='724ab7d6-948c-43b8-86f0-a9ccf61d9a89/video.wav', start=60, length=45, type='audio'),
    dict(id='reflection', title='Interview excerpt 02', subtitle='Audio excerpt · speech + language',
         description='A second excerpt from the same recording, with its own transcript and signal timeline.',
         source='724ab7d6-948c-43b8-86f0-a9ccf61d9a89/video.wav', start=120, length=30, type='audio'),
]

SAMPLES += json.loads((ROOT / "scripts/demo_sample_sources.json").read_text())


def run(*args):
    subprocess.run(args, check=True)


def main():
    from app.backend.api import routes
    from app.backend.api.schemas import AnalysisStatus, AnalysisResponse
    import torch
    torch.set_num_threads(2)
    output = ROOT / 'app/frontend/public/demo'
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in SAMPLES:
        sid = sample['id']
        ext = 'mp4' if sample['type'] == 'video' else 'mp3'
        media = output / f'{sid}.{ext}'
        source = ROOT / 'test_data' / sample['source']
        if sample.get('attribution') and hashlib.sha256(source.read_bytes()).hexdigest() != sample['attribution']['sha256']:
            raise RuntimeError(f'Source checksum mismatch: {sid}')
        args = ['ffmpeg', '-y', '-v', 'error']
        # Legacy FLV seeking, even to zero, can discard initial keyframes.
        if sample['start'] > 0:
            args += ['-ss', str(sample['start'])]
        args += ['-i', str(source), '-t', str(sample['length'])]
        if ext == 'mp4':
            args += ['-map', '0:v:0', '-map', '0:a:0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '25', '-preset', 'fast', '-c:a', 'aac', '-ac', '2', '-ar', '44100', '-movflags', '+faststart']
        else:
            args += ['-vn', '-ac', '1', '-ar', '16000', '-codec:a', 'libmp3lame', '-b:a', '64k']
        if not media.exists() or '--force' in sys.argv:
            run(*args, str(media))
        audio = output / f'{sid}.wav'
        if not audio.exists() or '--force' in sys.argv:
            run('ffmpeg', '-y', '-v', 'error', '-i', str(media), '-vn', '-ac', '1', '-ar', '16000', str(audio))
        if ext == 'mp4' and (not (output / f'{sid}.jpg').exists() or '--force' in sys.argv):
            run('ffmpeg', '-y', '-v', 'error', '-i', str(media), '-frames:v', '1', str(output / f'{sid}.jpg'))
        result_path = output / f'{sid}.json'
        if not result_path.exists() or '--force' in sys.argv or ('--refresh-reflection' in sys.argv and sid == 'reflection'):
            jid, vid = str(uuid.uuid4()), str(uuid.uuid4())
            now = datetime.now()
            routes.jobs[jid] = AnalysisStatus(job_id=jid, status='pending', progress=0, message='Preparing sample', created_at=now, updated_at=now)
            # Audio WAV input avoids compressed audio being copied under a WAV name.
            routes._process_video_analysis(jid, vid, str(media if ext == 'mp4' else audio), None, None)
            if routes.jobs[jid].status != 'completed':
                raise RuntimeError(routes.jobs[jid].message)
            result = AnalysisResponse(**routes.job_results[jid]).model_dump(mode='json')
            result['audio_url'] = f'/demo/{sid}.wav'
            result['sample_provenance'] = {'source': f'test_data/{sample["source"]}', 'offset_seconds': sample['start'], 'excerpt_seconds': result['duration'], 'kind': 'precomputed_model_output', 'visual_emotion_model': 'unavailable'}
            if sample.get('attribution'):
                result['sample_provenance']['attribution'] = sample['attribution']
            result_path.write_text(json.dumps(result, indent=2))
        result = json.loads(result_path.read_text())
        manifest.append({**{k: sample[k] for k in ['id', 'title', 'subtitle', 'description', 'type']},
                         'duration': result['duration'], 'media': f'/demo/{media.name}', 'audio': f'/demo/{sid}.wav',
                         'poster': f'/demo/{sid}.jpg' if ext == 'mp4' else None, 'results': f'/demo/{sid}.json',
                         **({'attribution': sample['attribution']} if sample.get('attribution') else {})})
        print(f'Published sample assets: {sid}', flush=True)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # This separable CREMA-D extract/catalog is distributed under the source database license.
    (output / 'crema-d-subset.json').write_text(json.dumps({
        'title': 'Telesense CREMA-D three-clip subset',
        'license': 'ODbL 1.0', 'license_url': 'https://opendatacommons.org/licenses/odbl/1-0/',
        'contents_license': 'DbCL 1.0',
        'notice': 'Contains information from CREMA-D, made available under the Open Database License (ODbL).',
        'source': 'https://github.com/CheyneyComputerScience/CREMA-D',
        'records': [s for s in manifest if s.get('attribution', {}).get('dataset') == 'CREMA-D'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
