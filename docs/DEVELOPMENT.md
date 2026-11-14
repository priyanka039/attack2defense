# Development

How to set up, run, test and extend Attack2Defense. For the big picture, read [ARCHITECTURE.md](ARCHITECTURE.md) first.

## Prerequisites

- Python 3.10 or newer
- Node 20 or newer (CI uses 22)
- About 1 GB of disk for CIFAR-10, PyTorch and the frontend dependencies
- Optional: an NVIDIA GPU with matching CUDA PyTorch wheels, for full-length training

## Setup

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[api,dev]"
cd frontend && npm install
```

Check what is installed:

```bash
python -m attack2defense doctor
```

It lists the dataset, the published results, which runs have weights, and the exact path for any missing `best.pt`. To install the reported ResNet-18 weights from a download:

```bash
python scripts/install_checkpoints.py path/to/downloads   # or a .zip
```

See [checkpoints.md](checkpoints.md) for details.

## Running

```bash
# terminal 1: API on :8000
python -m uvicorn backend.app.main:create_app --factory --reload --port 8000

# terminal 2: UI on :5173, proxies /api to :8000
cd frontend && npm run dev
```

Set `A2D_API_URL` before `npm run dev` to proxy to a different backend. The API's interactive docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

For anything beyond a laptop, set `A2D_SECRET_KEY` (in the environment or `backend/.env`). Every setting is listed in [API.md](API.md#configuration).

To make CPU demo models available without the UI:

```bash
python -m attack2defense train --model small_cnn --mode clean --epochs 3 --max-train-batches 100 --output-dir outputs/cpu_preview/small_cnn_clean
python -m attack2defense train --model small_cnn --mode fgsm  --epochs 3 --max-train-batches 100 --output-dir outputs/cpu_preview/small_cnn_fgsm
```

## Tests and checks

| Command | What it covers | Time |
| --- | --- | --- |
| `python -m pytest tests -q` | Attacks, projection, training loop, evaluation, smoothing, registry, every API route | 1 to 3 minutes on a CPU |
| `python -m ruff check .` | Lint (pycodestyle, pyflakes, isort, bugbear, pyupgrade, simplify) | Seconds |
| `python -m ruff format --check .` | Formatting, line length 110 | Seconds |
| `cd frontend && npm test` | Vitest and Testing Library: API client, formatting, login, guide redirect, InfoTip and radio-group accessibility | About a minute |
| `cd frontend && npm run build` | TypeScript type check and production build | Under a minute |

The API tests (`tests/api/`) build a real app with `create_app(Settings(...))` in a temporary folder. They use a random-weight Small CNN and fake CIFAR images, so they need no download. `inline_jobs=True` runs jobs in the request thread, so a test can start a training job and read its result straight away.

GitHub Actions (`.github/workflows/ci.yml`) runs all of the above on every push to `main` and on every pull request.

### Windows notes

If `npx` or `npm run` fails with "Cannot find module npm-prefix.js", the npm shim is broken. Call the tools through Node directly:

```powershell
node node_modules\typescript\bin\tsc --noEmit
node node_modules\vitest\vitest.mjs run
node node_modules\vite\bin\vite.js build
```

## Project conventions

- **Layering.** The ML package must not import from `backend`. Routers call services; services call the ML package. The UI only uses `src/api/endpoints.ts`.
- **Errors.** Raise `ApiError(status, code, message, field)`. Messages are shown to users, so write them as instructions ("Use at most 30 epochs here."), not as stack traces.
- **Units.** Epsilon and alpha are fractions of `[0, 1]` everywhere in code and in the API. The UI converts to and from "levels out of 255".
- **Colour meaning.** In the UI, `clean` (teal) means clean or defended, `adv` (pink) means attacked or fooled, and `warn` (amber) means target or warning. Do not use them decoratively.
- **Words.** New terms go in `frontend/src/lib/glossary.ts` and are shown with `<InfoTip term="...">`, so every screen explains them the same way.
- **Accessibility.** Use the primitives in `components/ui.tsx`: `Segmented` is a real radio group, `Slider` has a label and value, `InfoTip` works with keyboard and screen readers. Keep visible focus rings.

## Common changes

### Add an API endpoint

1. Add request and response models to `backend/app/schemas.py`.
2. Add the route to an existing router in `backend/app/routers/`, or create a router and add it to the tuple in `backend/app/main.py`.
3. Use `Depends(current_user)` if it spends compute, `Depends(optional_user)` if logged-in users see more.
4. Add a test in `tests/api/`.
5. Add the typed wrapper to `frontend/src/api/endpoints.ts` and `types.ts`.
6. Document it in [API.md](API.md).

### Add a background job type

Follow `routers/certify.py`: validate, `create_job(...)`, then `services.jobs.submit(job.id, lambda ctx: your_function(..., ctx))`. Inside the function, call `ctx.progress(fraction, message)` and check `ctx.cancelled()` regularly; raise `TrainingCancelled` to stop. Return a JSON-serializable dict; it becomes `job.result`.

### Add a defense (training mode)

1. Extend `_training_inputs` in `attack2defense/train.py` to build the training batch for the new mode.
2. Add the mode to `TrainConfig` validation, to `MODE_LABELS` in `registry.py`, and to the `mode` literal in `TrainingRequest`.
3. Add it to `DEFENSES` in `frontend/src/pages/Training.tsx` with a one-line explanation.
4. Add a test in `tests/test_pipeline.py`.

### Add a screen

Create `frontend/src/pages/Name.tsx`, lazy-load it in `App.tsx`, and add a nav item with a one-line hint in `components/Layout.tsx`. Start the page with `PageHeader`, and give every empty or error state an `EmptyState` or `Notice` that says what to do next.

## Reproducing the reported results

The ResNet-18 numbers need a GPU (a free Colab T4 takes about 2 to 3 hours for the three main runs):

1. `python scripts/package_for_colab.py`
2. Open `notebooks/run_experiments_colab.ipynb` on a GPU runtime, upload the zip, run every cell.
3. Download `results.zip` and the `best.pt` files.
4. `python scripts/build_report.py --results results.zip` regenerates `REPORT.md`, `docs/figures/`, `docs/results_tables.md` and the README results block.
5. `python scripts/install_checkpoints.py <downloads>` makes the models live in the app.
6. `python scripts/md_to_pdf.py` rebuilds the evaluation guide PDF (needs `pip install markdown xhtml2pdf`).

## Housekeeping

- `outputs/smoke/` and `outputs/smoke2/` are leftovers from smoke tests. Nothing scans them, so they are safe to delete.
- `backend/attack2defense.db` holds users and jobs. Deleting it resets all accounts; user runs in `outputs/api_runs/` then become orphaned and can be deleted too.
