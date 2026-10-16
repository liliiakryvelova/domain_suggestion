# Deploy the API to Render (free tier)

## What this is

`render.yaml` is a Render Blueprint that deploys the FastAPI backend
(`scripts/api.py`) as a free web service. The GitHub Pages frontend
(`docs/index.html`) stays as-is until the last step.

## Steps

1. **Push these files to GitHub** (`render.yaml`, this file) on `main`.
   The Blueprint is read from the repo, so it must be committed first.
2. Go to https://render.com → **New → Blueprint**.
3. Connect the repo **`liliiakryvelova/domain_suggestion`** and select it.
   Render detects the `domain-suggester-api` service from `render.yaml`.
4. **Optional:** set `OPENAI_API_KEY` in the service Environment settings.
   Only needed for `/judge-domain` + moderation. Leave it unset and
   `/generate-domains` still works; `/judge-domain` returns an honest 503.
5. Click **Apply / Deploy**. The first build takes several minutes
   (torch + transformers are large). Watch the deploy logs for:
   `Model loaded successfully (Hugging Face Hub (fine-tuned))` and
   `Application startup complete.`
6. Copy the service URL, e.g. `https://domain-suggester-api.onrender.com`.
7. Verify:
   ```bash
   curl https://domain-suggester-api.onrender.com/health
   curl -X POST https://domain-suggester-api.onrender.com/generate-domains \
     -H 'Content-Type: application/json' \
     -d '{"business_description":"AI-powered fitness app","num_domains":3}'
   ```
8. **Separate follow-up (not done here):** change `API_BASE` in
   `docs/index.html` to the Render URL and push, so the GitHub Pages UI
   (`https://liliiakryvelova.github.io/domain_suggestion`) generates for
   visitors.

## Free-tier notes

- The free service **sleeps after ~15 minutes of inactivity**. The first
  request after a sleep takes ~1 minute: the instance wakes, then the app
  downloads/loads the model before it can answer.
- **RAM risk:** the free tier has 512 MB RAM. GPT-2 (124M params) in fp32
  plus torch/transformers can exceed that and OOM during model load or
  first generation. If the deploy logs show the process being killed
  (OOM / exit 137), the smallest fix is loading the model in float16
  (`torch_dtype=torch.float16` in `from_pretrained`, one-line change in
  `scripts/api.py`), or moving the service to a larger Render instance.
