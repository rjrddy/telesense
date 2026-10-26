# Telesense

Prototype multimodal (audio, video and text) behavioral analysis for telehealth sessions.

> Telesense is a research prototype. It does not diagnose, screen for, or assess risk of any condition, and its output (including "High Risk" flags, protective factors and missing-question prompts) must not be used to make clinical decisions. The models are untrained or uncalibrated for clinical use and produce false positives and false negatives. Anyone at risk needs evaluation by a qualified clinician; in an emergency, call local emergency services. Do not upload real patient recordings unless you have consent and the handling complies with applicable privacy law (e.g. HIPAA).

## What it is

Over video, clinicians lose many of the nonverbal cues they rely on in behavioral health appointments: facial expression, tone of voice, pauses, and how a patient's affect shifts during a conversation. Telesense explores whether a model can surface those cues as a second observer. It takes a recorded session and returns **timestamped flags** for moments that may need follow-up, along with an emotion timeline, protective factors the patient mentioned, and intake topics that were never covered.

The project has two parts:

- **`telesense_ml`** (`src/telesense_ml/`): a Python package with trainable multimodal models that fuse audio, video and transcript to localize behavioral concerns in time.
- **The Telesense app** (`app/`): a local web app (FastAPI + React) where you upload a session, run the analysis, and review flags alongside the synchronized video, waveform and transcript.

## Architecture

```
                    ┌─────────────────────────── Telesense app ────────────────────────────┐
 Browser (React)    │  FastAPI backend                                                     │
  upload ─────────► │  1. extract audio + frames (ffmpeg / OpenCV)                         │
  progress ◄─ WS ── │  2. transcribe (Whisper)                                             │
  results ◄──────── │  3. analysis engine ──┬─ MODEL_CHECKPOINT set → Telesense model      │
                    │                       └─ otherwise → pretrained emotion models       │
                    │  4. flags + severity  5. transcript-based panels  6. emotion timeline│
                    └──────────────────────────────────────────────────────────────────────┘

 Telesense model (src/telesense_ml)
   audio ─► Wav2Vec2 ─────────┐
   video ─► ResNet-50/DenseNet┼─► project to shared size ─► multi-scale temporal encoder ─► per-frame
   text  ─► DistilBERT ───────┘                            (TCN + Transformer)             concern scores
                                                                                            ─► segments
```

**Data flow.** A session is uploaded and stored locally. The backend extracts the audio, transcribes it with Whisper, and runs one of two analysis engines:

1. **The Telesense model.** It runs when `MODEL_CHECKPOINT` points to a trained checkpoint. Each modality goes through its own pretrained encoder. The features are projected to a shared size and passed through a multi-scale temporal encoder, then a localization head scores every frame for each concern type. High-scoring frames are merged into flagged segments.
2. **The pretrained pipeline.** This is the default until a model is trained. It combines off-the-shelf emotion models: wav2vec2 for speech emotion, RoBERTa (go_emotions) for the transcript and FER for faces. A keyword detector and temporal smoothing turn their scores into flags.

The dashboard shows which engine produced the results. In both cases, protective factors and missing intake questions come from transparent phrase-matching rules on the transcript.

### Why this design

| Choice | Reason |
|---|---|
| **Three modalities** | Each one catches cues the others miss: words in the transcript, prosody and pauses in the audio, affect in the face. Fusing them is the point of the project. |
| **Pretrained encoders** (Wav2Vec2, ResNet/DenseNet, DistilBERT) | Labeled behavioral-health video is scarce. Transfer learning reuses representations learned from large general datasets, so the task-specific layers need far less data. |
| **Temporal localization, not one label per session** | Clinicians need to know *when* something happened so they can review it. Frame-level scores also make the output easier to audit. |
| **Multi-scale temporal encoder** (TCN + Transformer) | Convolutions capture short and medium-range patterns such as a pause or a change in tone. Attention captures longer arcs across the whole session. |
| **Cross-modal attention fusion** | Lets one modality weigh another, for example a neutral face with distressed speech. It's inspired by the MEmoBERT option in the original proposal, which was chosen over the MESM and Bi-GRU options. |
| **Separate unsupervised track** (temporal VAE + contrastive learning) | Works without labels: it learns what typical sessions look like and flags moments with high reconstruction error. This is useful while labeled data is unavailable. |
| **Pretrained fallback pipeline** | Makes the prototype usable before any model is trained, and gives a baseline to compare the trained model against. |
| **Rule-based panels** | Phrase matching is predictable and easy to inspect. It sends no data to external services and requires no API key. |
| **Local FastAPI + React, not cloud services** | Keeps recordings on the machine during prototyping. The cloud pipeline from the original proposal (Kinesis, SageMaker, S3) is deferred. |
| **`src/` layout with `pyproject.toml`** | A standard installable Python package with absolute imports, so training scripts, the app and notebooks import the same code. |

## Requirements

| Tool | Version | Notes |
|---|---|---|
| macOS or Linux | | Developed on macOS (Apple Silicon). Windows is untested. |
| Conda (Miniforge recommended) | | Manages the Python environment |
| Python | 3.10 or newer | Developed and tested on 3.14 |
| Node.js and npm | Node 18 or newer | Developed on Node 23 |
| ffmpeg | any recent | Audio extraction |
| GPU | optional | Apple MPS or NVIDIA CUDA is used automatically; CPU works but is slow |

The first run downloads several GB of pretrained model weights from Hugging Face.

## Installation

**1. System tools** (macOS with Homebrew; use your package manager on Linux)

```bash
brew install ffmpeg node
```

**2. Python environment.** The launch scripts expect a conda environment named `fantasy`. Set `CONDA_ENV=<name>` to use another name.

```bash
conda create -n fantasy python=3.14
conda activate fantasy
```

**3. Python packages**, from the repo root:

```bash
pip install -r requirements-minimal.txt              # ML stack (torch, transformers, librosa, OpenCV, ...)
pip install -r app/backend/requirements_conda.txt    # backend (FastAPI, uvicorn, Whisper via transformers, ...)
pip install -e .                                     # makes the `telesense_ml` package importable
```

`requirements.txt` is the same list plus `dlib` and `face-recognition`, which add face cropping but often fail to build. If you want them, install dlib from conda-forge first: `conda install -c conda-forge dlib`.

**4. Optional extras**

```bash
pip install fer                  # facial emotion model for the video track
pip install pyannote.audio       # voice-activity detection (needs a Hugging Face token, see below)
```

**5. Frontend packages**

```bash
cd app/frontend && npm install && cd ../..
```

**6. Check the setup**

```bash
python scripts/verify_setup.py
```

## Running

Start the backend and frontend in two terminals from the repo root:

```bash
./app/start_backend_conda.sh             # API on http://localhost:8000 (interactive docs at /docs)
./app/start_frontend.sh                  # UI on http://localhost:3000
```

Without conda, `./app/start_backend.sh` creates a virtual environment in `app/backend/venv` and installs `app/backend/requirements.txt` on first run.

The backend listens on `127.0.0.1` only, so other machines on your network can't reach it. The API has no authentication, so don't change this to `0.0.0.0` or put it behind a public address.

Open http://localhost:3000, drop in an `.mp4`, `.mov`, `.avi`, `.mkv`, `.wav`, `.mp3`, `.m4a` or `.flac` file, and wait for the analysis to finish.

| Environment variable | Purpose |
|---|---|
| `MODEL_CHECKPOINT` | Path to a trained supervised checkpoint. When set, the trained Telesense model analyzes sessions. |
| `MODEL_CONFIG` | Optional YAML config that overrides the one stored in the checkpoint |
| `HUGGINGFACE_TOKEN` | Enables the gated pyannote voice-activity model (request access on Hugging Face first) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | Optional LLM severity scoring when the trained Telesense model is used. This sends transcript excerpts to that provider. |
| `USE_LLM_SEVERITY` | Set to `false` to turn LLM severity scoring off |
| `CONDA_ENV` | Conda environment used by `start_backend_conda.sh` (default `fantasy`) |
| `TELESENSE_STORAGE_DIR` | Where uploads, processed media and results are stored (default `app/storage/`) |
| `VITE_API_URL` | Backend address for the frontend's API requests (default `http://localhost:8000`; the audio and WebSocket URLs are still fixed to `localhost:8000`) |

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload a video or audio file; returns a `video_id` |
| `POST` | `/api/analyze/{video_id}` | Start an analysis job in the background; returns a `job_id` |
| `GET` | `/api/status/{job_id}` | Job status and progress |
| `GET` | `/api/results/{job_id}` | Flags, transcript, statistics, panels and the engine used |
| `WS` | `/api/ws/{job_id}` | Live progress updates |
| `GET` | `/api/audio/{video_id}/audio.wav` | Extracted audio for the waveform view |
| `GET` | `/health` | Health check |

IDs must be UUIDs. The model checkpoint is chosen by the server (`MODEL_CHECKPOINT`), never by the request. Interactive documentation is at http://localhost:8000/docs.

### Severity levels

Each flag gets one of three levels, shown in the UI as **High Risk** (`high`), **High Concern** (`medium`) and **Concern** (`low`).

| Source | High Risk | High Concern | Concern |
|---|---|---|---|
| Pretrained engine, fused flags | confidence ≥ 0.80 | ≥ 0.65 | below 0.65 |
| Pretrained engine, per-modality flags | ≥ 0.75 | ≥ 0.65 | below 0.65 |
| Trained model, rule-based severity | score ≥ 0.75 | ≥ 0.50 | below 0.50 |
| Trained model, LLM fallback rules | score ≥ 0.60 | ≥ 0.30 | below 0.30 |

Critical phrases such as suicidal ideation, self-harm or homicidal ideation are always **High Risk** (`psychiatric_keyword_detector.py`). These thresholds are uncalibrated defaults, not clinical cut-offs.

## Training the models

Create a `datasets/` folder (it is not committed) and add each dataset with a metadata CSV per split:

```
datasets/raw/
└── cmu_mosei/                 # also crema_d, meld, mer2024
    └── metadata/
        ├── train.csv          # columns: audio_path, video_path, text, label
        ├── val.csv
        └── test.csv
```

The supervised temporal model also needs **per-frame labels**: either segments `(start_frame, end_frame, concern_type)` or one concern label per frame. Loading these from your annotation format is not implemented yet (see `_segments_to_frames` in `src/telesense_ml/supervised/training/temporal_trainer.py`).

Then run:

```bash
python scripts/supervised/train_temporal.py --config configs/supervised/temporal_config.yaml
python scripts/supervised/inference_temporal.py --checkpoint checkpoints/best_model.pt --video session.mp4

python scripts/unsupervised/train.py --config configs/unsupervised/unsupervised_config.yaml
python scripts/unsupervised/inference.py --checkpoint checkpoints_unsupervised/vae_best_model.pt --video session.mp4
```

To use a trained supervised model in the app, start the backend with `MODEL_CHECKPOINT=checkpoints/best_model.pt`.

## Code organization

```
telesense/
├── README.md
├── pyproject.toml                      # Package definition (pip install -e .)
├── requirements.txt                    # Full ML dependencies (includes dlib / face-recognition)
├── requirements-minimal.txt            # Same without dlib (recommended)
│
├── src/telesense_ml/                   # ── ML package ──
│   ├── data/                           # Shared by both models
│   │   ├── datasets.py                 #   MultimodalDataset, get_dataset (MOSEI, CREMA-D, MELD, MER2024)
│   │   ├── preprocessing.py            #   Audio / video / text preprocessors
│   │   └── collators.py                #   Batch padding and tokenization
│   ├── encoders/                       # Shared pretrained feature extractors
│   │   ├── audio_encoder.py            #   Wav2Vec2 (+ optional temporal transformer)
│   │   ├── visual_encoder.py           #   ResNet-50 / DenseNet-121 per frame
│   │   └── text_encoder.py             #   DistilBERT
│   ├── supervised/
│   │   ├── models/
│   │   │   ├── fusion.py               #   Cross-modal attention
│   │   │   ├── multimodal_model.py     #   Clip-level classifier
│   │   │   ├── temporal_encoders.py    #   TCN + Transformer temporal encoders
│   │   │   ├── temporal_localization.py  # Per-frame head, attention, smoothing
│   │   │   └── temporal_behavioral_model.py  # Main model: per-frame concern scores
│   │   ├── training/                   #   trainer, temporal_trainer, losses, metrics
│   │   └── inference/                  #   predictor (clip), temporal_predictor (frames → segments)
│   └── unsupervised/
│       ├── models/                     #   temporal_vae, contrastive_learning, unsupervised_model
│       ├── training/                   #   vae_trainer, unsupervised_trainer
│       └── inference/                  #   unsupervised_predictor (anomaly segments)
│
├── configs/
│   ├── supervised/                     # model_config.yaml, temporal_config.yaml
│   └── unsupervised/                   # unsupervised_config.yaml
├── scripts/
│   ├── supervised/                     # train, train_temporal, evaluate, inference, inference_temporal
│   ├── unsupervised/                   # train, inference
│   └── verify_setup.py                 # Dependency and device check
│
├── app/                                # ── Telesense web app ──
│   ├── backend/
│   │   ├── main.py, run.py             #   FastAPI app, CORS, /health
│   │   ├── api/
│   │   │   ├── routes.py               #   upload, analyze, status, results, WebSocket, audio
│   │   │   └── schemas.py              #   Request / response models
│   │   ├── models/
│   │   │   ├── analyzer.py             #   trained-model engine (+ optional LLM severity)
│   │   │   ├── free_analyzer.py        #   pretrained-model engine
│   │   │   ├── audio_emotion_detector.py, text_emotion_detector.py,
│   │   │   │   video_emotion_detector*.py, vad_detector.py  # per-modality detectors
│   │   │   ├── psychiatric_keyword_detector.py, temporal_fusion.py
│   │   │   ├── clinical_evaluator.py   #   protective factors + missing intake questions
│   │   │   ├── emotion_summary.py      #   emotion timeline
│   │   │   ├── transcriber.py, video_transcriber.py  # Whisper
│   │   │   └── llm_severity_analyzer.py, video_processor.py
│   │   ├── utils/file_handler.py       #   upload storage, ID validation
│   │   └── configs/emotion_config.yaml
│   ├── frontend/                       #   React + Vite (VideoPlayer, flag monitors, SummaryDashboard, panels)
│   ├── storage/                        #   Runtime uploads, processed media, results (not committed)
│   └── start_backend.sh, start_backend_conda.sh, start_frontend.sh
│
├── datasets/                           # Your datasets (local only, not committed)
└── docs/                               # Local reference files (not committed)
```

## Current status and limitations

- **No trained model is included.** Until you train one, the app uses the pretrained pipeline.
- **Only the first minute reaches the Telesense model.** It analyzes the first `max_frames` of video (1800 frames, about 60 s by default). Longer sessions need windowed inference.
- **The pretrained pipeline isn't calibrated** and flags too readily. Treat its severity levels as relative, not clinical.
- **The panels are keyword rules.** They can miss paraphrases and match unrelated words.
- **Everything runs locally**, with no authentication. Uploads, extracted audio and results are stored under `app/storage/` (or `TELESENSE_STORAGE_DIR`) and job state is kept in memory. The server binds to `127.0.0.1` for this reason.

## Data

No datasets, recordings, analysis output or other data files are committed, and neither is the `docs/` folder. Datasets such as DAIC-WOZ, MELD, CREMA-D and CMU-MOSEI must be obtained under their own licenses and placed in `datasets/raw/<name>/`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `OMP: Error #15 ... libomp.dylib already initialized` (macOS) | `export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1` (the conda start script sets these) |
| `Library not loaded: @rpath/libgfortran.5.dylib` | `conda install -c conda-forge libgfortran5` |
| `dlib` fails to build | Use `requirements-minimal.txt`, or `conda install -c conda-forge dlib` first |
| `ModuleNotFoundError: telesense_ml` or `app` | `pip install -e .` fixes `telesense_ml`; `app` is only importable from the repo root. Use the start scripts, or run from the repo root with `PYTHONPATH="$(pwd):$(pwd)/src"` |
| Voice-activity detection disabled (403 from Hugging Face) | Accept the pyannote model terms on Hugging Face and set `HUGGINGFACE_TOKEN` |
| `torchcodec` / FFmpeg library warnings at startup | Harmless; audio is loaded through librosa/ffmpeg instead |
| Out of memory on MPS/CUDA | Lower `batch_size` or `max_frames` in the config, or set `device.type: cpu` |
| First run is slow | Pretrained weights (several GB) download once and are cached |

## Portfolio deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the Vercel frontend and single-instance Fly.io CPU backend setup, demo-key gate, persistent model cache, and smoke checks. Use synthetic or explicitly public demo recordings only.
