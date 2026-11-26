# Deploy on Render (single service)

> Prefer AWS EC2? Use `bash scripts/ec2_redeploy.sh` on the server so the SQLite
> database and JWT secret live in `~/a2d-persist` and survive image rebuilds
> (you will not have to register a new account every deploy).

One Docker Web Service serves the React UI and the FastAPI backend from the same URL.

## Requirements

- GitHub repo connected to Render.
- **Free** plan is the default in `render.yaml` (no payment). Free = **512 MB RAM**, sleeps after ~15 minutes idle, first open can take ~1 minute.
- If the service crashes with out-of-memory when loading PyTorch, you either stay on free and keep attacks tiny, or upgrade later to a 2 GB plan.

## Deploy

1. Push `main` (includes `Dockerfile`, `render.yaml`, and `deploy/publish/`).
2. [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Select the `attack2defense` repo and apply `render.yaml` (plan **free**).
4. Wait for the first build (often 10–20 minutes: Node build + CPU PyTorch).
5. Open `https://<service>.onrender.com`.

Or: **New** → **Web Service** → Docker → root `Dockerfile`, health check `/api/health`, plan **Free**, env `A2D_SECRET_KEY` (generate).

Do **not** add a credit card unless Render forces account verification for your region — Free web services do not require paid compute.

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
