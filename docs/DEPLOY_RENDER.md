# Deploy on Render (single service)

One Docker Web Service serves the React UI and the FastAPI backend from the same URL.

## Requirements

- Render **Standard** plan (2 GB RAM). Free/Starter (512 MB) runs out of memory when importing PyTorch.
- GitHub repo connected to Render.

## Deploy

1. Push `main` (includes `Dockerfile`, `render.yaml`, and `deploy/publish/`).
2. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Select the `attack2defense` repo and apply `render.yaml`.
4. Wait for the first build (often 10–20 minutes: Node build + CPU PyTorch).
5. Open `https://<service>.onrender.com`.

Or: **New** → **Web Service** → Docker → root `Dockerfile`, health check `/api/health`, plan Standard, env `A2D_SECRET_KEY` (generate).

## What ships in the image

| Path | Purpose |
| --- | --- |
| `deploy/publish/results/` | Reported ResNet-18 CSVs for the Dashboard |
| `deploy/publish/checkpoints/` | Small CNN weights for live Attack lab demos |

ResNet-18 `best.pt` files are not in the image (~85 MB each). Dashboard numbers still show; live ResNet attacks need those weights installed separately or train a Small CNN in the UI.

## After deploy

- First request may download CIFAR-10 into `/app/data` (ephemeral on free disk).
- Cold start on free instances can take a minute; Standard stays up.
- Create an account on `/login`, then try Attack lab with a Small CNN model.
