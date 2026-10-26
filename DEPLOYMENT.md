# Portfolio demo deployment

Research only. Not for clinical use. Use synthetic or explicitly public demo recordings; do not upload patient data. This is an upload-and-analyze app, not a live telehealth service.

## Backend: Fly.io

Install Docker for local image checks and `flyctl` for deployment, then authenticate with `flyctl auth login`. Run these commands from the repository root on `telesense-ml-app`.

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

Import the repository and select branch `telesense-ml-app` as the production branch. Keep Vercel's **Root Directory at the repository root (`.`)**: the root `vercel.json` explicitly installs and builds `app/frontend` and serves `app/frontend/dist`. Selecting `app/frontend` as the Vercel root would bypass this configuration.

Set `VITE_API_URL=https://YOUR-FLY-APP.fly.dev` for Production (and Preview if used), then deploy. Vite embeds this URL at build time, so URL changes require a new frontend deployment. Leave `VITE_TELESENSE_DEMO_KEY` unset: all `VITE_*` variables are public browser bundle contents. Visitors enter the shared key in the demo-key field; it is stored in their browser's localStorage. Save an empty value to clear it.

Once Vercel assigns the production domain, update Fly's `CORS_ORIGINS` to match and redeploy the backend. Share the key separately from the public portfolio link. Rotating it with Fly secrets requires visitors to save the new key. Audio and WebSocket URLs include the key because browser media/WS APIs cannot set the HTTP auth header; avoid sharing these URLs or logging query strings.

## Local verification

With a Python 3.11 environment and Node/npm installed:

```sh
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'transformers>=4.35,<5' 'moviepy>=1.0.3,<2' -r requirements-minimal.txt -r app/backend/requirements_conda.txt
pip install -e .
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
