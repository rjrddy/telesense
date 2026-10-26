# Portfolio demo deployment

Research only. Not for clinical use. Use synthetic or explicitly public demo recordings; do not upload patient data. This is an upload-and-analyze app, not a live telehealth service.

## Backend: Fly.io

Install Docker for local image checks and `flyctl` for deployment, then authenticate with `flyctl auth login`. Run these commands from the repository root on `main`.

1. Choose a globally unique app name. The checked-in starting name is `telesense-demo-api`; availability is not guaranteed. Replace `app` in `fly.toml` if necessary.
2. Run `flyctl launch --no-deploy`. Reuse the existing configuration when prompted. Check the resulting `fly.toml`: keep the Dockerfile build, `ord` region, `/data` mount, 8080 internal port, forced HTTPS, `/health` check, two shared vCPUs and 4 GB RAM. Keep autostop off so background jobs survive idle periods.
3. Create the volume in the same region:

   ```sh
   flyctl volumes create telesense_data --region ord --size 10
   ```

4. Set a strong shared key without including it in shell history:

   ```sh
   read -s TELESENSE_DEMO_KEY
   export TELESENSE_DEMO_KEY
   flyctl secrets import <<EOF_SECRET
   TELESENSE_DEMO_KEY=$TELESENSE_DEMO_KEY
   EOF_SECRET
   unset TELESENSE_DEMO_KEY
   ```

   Do not leave the secret unset in a public deployment: unset means auth is disabled. Leave `MODEL_CHECKPOINT`, `MODEL_CONFIG`, `OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` unset for this pretrained-only demo.
5. Set `CORS_ORIGINS` in `fly.toml` to the exact frontend origin, e.g. `https://your-project.vercel.app` (no trailing slash). Multiple origins are comma-separated. Add preview origins explicitly if needed.
6. Deploy exactly one Machine:

   ```sh
   flyctl deploy --ha=false
   flyctl scale count 1
   flyctl status
   flyctl checks list
   flyctl logs
   ```

The container installs CPU PyTorch wheels and runs one Uvicorn worker. Its entrypoint initializes volume permissions before switching to the non-root app user. Hugging Face weights download on first analysis into `/data/hf-cache`; uploads, extracted audio and results live in `/data/storage`. First use can be slow. Start with short recordings; 4 GB is a starting allocation, not a guarantee for every recording. Inspect logs and increase RAM if inference runs out of memory.

## Frontend: Vercel

Import the repository and select branch `main` as the production branch. Keep Vercel's **Root Directory at the repository root (`.`)**: the root `vercel.json` explicitly installs and builds `app/frontend` and serves `app/frontend/dist`. Selecting `app/frontend` as the Vercel root would bypass this configuration.

Set `VITE_API_URL=https://YOUR-FLY-APP.fly.dev` for Production (and Preview if used), then deploy. Vite embeds this URL at build time, so URL changes require a new frontend deployment. Leave `VITE_TELESENSE_DEMO_KEY` unset: all `VITE_*` variables are public browser bundle contents. Visitors enter the shared key in the demo-key field; it is stored in their browser's localStorage. Save an empty value to clear it.

Once Vercel assigns the production domain, update Fly's `CORS_ORIGINS` to match and redeploy the backend. Share the key separately from the public portfolio link. Rotating it with Fly secrets requires visitors to save the new key. Audio and WebSocket URLs include the key because browser media/WS APIs cannot set the HTTP auth header; avoid sharing these URLs or logging query strings.

## Local verification

With a Python 3.11 environment and Node/npm installed:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r app/backend/requirements.txt
python -m pip install -e .
TELESENSE_DEMO_KEY=test CORS_ORIGINS=http://localhost:5173 python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Install system `ffmpeg` as well. In a second terminal:

```sh
cp app/frontend/.env.example app/frontend/.env.local
npm ci --prefix app/frontend
npm run dev --prefix app/frontend -- --port 5173
```

Open `http://localhost:5173`, enter `test`, and save. Upload a small synthetic clip and confirm progress, transcript, waveform, and extracted audio playback. Clear the key and confirm upload is rejected. The key field also appears locally for this check.

```sh
curl -i http://localhost:8000/health
# 200
curl -i http://localhost:8000/api/status/nonexistent
# 401
curl -i -H 'X-Telesense-Key: test' http://localhost:8000/api/status/nonexistent
# 404: auth passed; no such job
curl -i -X OPTIONS http://localhost:8000/api/upload \
  -H 'Origin: http://localhost:5173' \
  -H 'Access-Control-Request-Method: POST' \
  -H 'Access-Control-Request-Headers: X-Telesense-Key'
# 200 with matching Access-Control-Allow-Origin
npm run build --prefix app/frontend
```

Repeat the HTTP checks against the Fly URL with the production key and Vercel origin. `/health` is intentionally public and checks server liveness, not model download or inference readiness. Check an unauthenticated WebSocket is rejected and an authenticated one connects.

For a local container check:

```sh
docker build -t telesense-demo .
docker volume create telesense-demo-data
docker run --rm -p 8080:8080 -v telesense-demo-data:/data \
  -e TELESENSE_DEMO_KEY=test -e CORS_ORIGINS=http://localhost:5173 telesense-demo
```

## Automated middleware checks

These checks exercise the real FastAPI middleware with lightweight substitute routes; they do not load models or validate inference:

```sh
pip install fastapi httpx
python -m unittest discover -s tests -p 'test_demo_middleware.py' -v
```

## Operating limits

Only one Machine and one worker are supported: job state is in memory, and analyses run serially in a worker thread. Restarts/deployments lose job status and interrupt pending work, even though files remain on the volume. Wait for analyses to finish before deploying. Do not scale horizontally with this design.

`TELESENSE_MAX_UPLOAD_MB` defaults to 100 MiB for the entire multipart request, including overhead. Both declared-size and streamed bodies are capped before route processing. This is not a duration limit or storage quota. Uploaded media and results are retained until manually deleted; monitor volume usage and remove demo sessions when no longer needed. Do not remove the model cache unless you want weights downloaded again.

Configuration references: [Fly app configuration](https://docs.fly.io/reference/configuration), [Vercel build configuration](https://vercel.com/docs/builds/configure-a-build).

## Public sample explorer

Production URL: **https://telesense.vercel.app**. The Vercel project is `telesense` under `raj-reddys-projects`, linked locally through ignored `.vercel/` metadata. For CLI deploys use `npx --yes vercel@latest deploy --prod --yes`; the older global CLI may be rejected by Vercel.

The frontend includes eight samples under `app/frontend/public/demo/`: three approved local excerpts, three CREMA-D acted videos, and two LibriSpeech audio utterances, with saved outputs from the real pretrained pipeline. Visitors can play recordings, search and seek transcripts, inspect modality timelines, and export JSON without a backend or key. Results are explicitly labeled as saved model analysis. Video emotion inference is unavailable; video provides playback context.

Regenerate the sample assets from the local source collection with:

```sh
HF_HOME=app/storage/hf-cache python scripts/build_demo_samples.py --force
```

This downloads model weights and runs CPU inference locally. Only the selected short excerpts, their waveforms/audio, and saved results are published; the full `test_data` collection remains excluded. Each result records its source and excerpt offset under `sample_provenance`.

For a portfolio deployment, **leave `VITE_API_URL` unset** until a Fly backend is running. The sample explorer works independently; the upload screen explains that new analysis is not connected and disables file submission. Localhost still defaults to the local backend. When Fly is ready, set `VITE_API_URL` to its HTTPS origin, set matching CORS and the shared key on Fly, and redeploy the frontend to enable uploads.

The redesigned review workspace uses the same backend result format. Whisper now processes overlapping chunks with model-derived timestamps rather than estimating word positions from text length. Timestamps and transcripts can still be inaccurate.

## Provisioned Fly resources

App: `telesense-demo-api` in the personal Fly organization. Region: `ord` (Chicago). The `telesense_data` volume is 10 GB, mounted at `/data`. `fly.toml` configures two shared vCPUs, 4 GB RAM, CPU inference, and CORS for `https://telesense.vercel.app`.

The Fly CLI is installed at `~/.fly/bin/flyctl`. Use that full path if `flyctl` is not on your shell PATH. The shared upload key is saved in the ignored root `.env.local` and stored in Fly secrets; never put it in a `VITE_*` variable or commit it.

The live backend is deployed at **https://telesense-demo-api.fly.dev**. Vercel Production now sets `VITE_API_URL` to that origin, enabling the gated upload flow at **https://telesense.vercel.app**. Sample exploration still works without a key. For new uploads, copy the `TELESENSE_DEMO_KEY` value from the ignored root `.env.local` into the website's Shared demo key field and save it.

Hosted verification passed: `/health`, missing-key 401, Vercel-origin CORS preflight, real video upload and CPU inference, results retrieval, protected extracted audio, and a browser upload from the public Vercel frontend through transcript/waveform/signal rendering.

The black UI includes a model-output diagram with animated curved connectors. The library previews saved output from its first sample; session review uses that session’s actual results. Connector motion illustrates the data path rather than real-time inference, and respects reduced-motion preferences.

The sample library now contains eight recordings: the three original approved excerpts, three CREMA-D acted video clips, and two LibriSpeech audio utterances. `scripts/fetch_demo_sources.py` retrieves and checksum-verifies the five external source files in ignored `test_data/external/`. `scripts/build_demo_samples.py` transcodes new clips and generates actual pretrained CPU outputs, then publishes the complete manifest. Licenses and attribution are served with the assets; reference text and source labels are displayed separately from model predictions.
