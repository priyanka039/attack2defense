# My project notes: Attack2Defense

My own reference: what this project is, how it was built step by step, how every piece of logic works, and how to explain it. The polished docs for other people are the [README](../README.md), [ARCHITECTURE](ARCHITECTURE.md), [API](API.md), [USER_GUIDE](USER_GUIDE.md) and [DEVELOPMENT](DEVELOPMENT.md). The viva cheat sheet is [EVALUATION_GUIDE](EVALUATION_GUIDE.md).

---

## 1. The project in five sentences

1. A neural network that classifies CIFAR-10 images with 93.6% accuracy can be pushed down to 6.1% by changing every pixel by at most 2 brightness levels out of 255, which nobody can see.
2. That change is found with the model's own gradient: FGSM takes one step, PGD takes ten smaller steps and is much stronger.
3. The defense is **adversarial training**: during training, attack every batch on the fly and teach the model to classify the attacked images correctly.
4. With PGD adversarial training the same ResNet-18 keeps **78.9%** under the same PGD attack, at a cost of 3 points of clean accuracy. The target was 80%, so it is 1.1 points short.
5. All of this is packaged as a Python library, a FastAPI backend and a React web app, so anyone can train, attack, compare and certify models from the browser.

### Numbers to remember

| ResNet-18, 10,000 test images, budget 2/255 | Clean | FGSM | PGD-10 |
| --- | --- | --- | --- |
| No defense | 93.6% | 31.7% | 6.1% |
| FGSM adversarial training | 91.4% | 79.1% | 77.9% |
| PGD adversarial training | 90.7% | 79.9% | **78.9%** |

Stronger attacks than the model was trained for (PGD-10 accuracy):

| Budget | No defense | FGSM training | PGD training |
| --- | --- | --- | --- |
| 2/255 (trained for) | 6.1% | 77.9% | 78.9% |
| 4/255 | 0.2% | 56.9% | 60.5% |
| 8/255 | 0.0% | 21.2% | 24.7% |

The memorable line: **93.6, then 6.1, then 78.9**.

---

## 2. The process, from start to end

This is the order the project was actually built in.

### Step 1: Exploration (last week)

- Took ImageNet-pretrained ResNet-50 and ViT-B/16 and attacked them on CIFAR cat images only (`attack2defense/zeroshot.py`, the old notebook).
- Lesson: attacks clearly work, but this was one class and models never trained on CIFAR. That is not a fair benchmark of a defense. It survives as the "Earlier experiment" table on the Dashboard.

### Step 2: A proper ML package

- Restructured everything into `attack2defense/` with separate modules for data, models, attacks, training and evaluation.
- Key fix: **`PixelModel`** puts normalization *inside* the model, so the attacks edit real pixels (see section 4.1).
- Added a CLI (`python -m attack2defense train / evaluate / ...`) so every experiment is one reproducible command.

### Step 3: GPU training on Colab

- `scripts/package_for_colab.py` zips the code. `notebooks/run_experiments_colab.ipynb` trains three ResNet-18 models for 30 epochs each on a T4 (about 2 to 3 hours total): `resnet18_clean`, `resnet18_fgsm`, `resnet18_pgd`.
- Recipe: SGD, learning rate 0.1, OneCycle schedule, weight decay 5e-4, batch 128, mixed precision, epsilon 2/255, 5 PGD steps during training, seed 42.
- Each run is evaluated on the full 10,000-image test set with a sweep at 2/255, 4/255 and 8/255.
- Output: `results.zip` (CSVs, configs, figures) and a `best.pt` per run.

### Step 4: The report

- `python scripts/build_report.py --results results.zip` reads only the CSVs and fills `docs/report_template.md` to produce `REPORT.md`, `docs/figures/`, `docs/results_tables.md` and the README results block. Numbers are never typed by hand.

### Step 5: Extra features (PRD stretch goals)

- PGD trace (success after each step) and timing, `adv_ratio`, thresholds and flags, CSV/JSON export, a model registry.
- Randomized smoothing (certified defense), Imagenette support, CW and AutoAttack benchmarks.

### Step 6: The full-stack app

- **Backend** (`backend/app/`): login with JWT, model and dataset catalog, attack endpoints, background jobs for training, evaluation and certification, and one error format.
- **Frontend** (`frontend/`): React screens for each part.
- **Tests and CI**: pytest for the package and the API, Vitest for the UI, GitHub Actions on every push.

### Step 7: Redesign and documentation (this round)

- Dark "security lab" redesign where colour has meaning: teal is clean or defended, pink is attack or fooled, amber is the target.
- A **Start here** guide so a first-time visitor understands the project in a minute, and "?" definitions for every term.
- The Attack lab became three clear steps with presets, a plain-English explanation and a side-by-side compare mode.
- Presets with time estimates on Training, Evaluation and Certified.
- A **setup check** (`doctor` command, `/api/setup`, sidebar dot) because the ResNet-18 `best.pt` files were never copied into `checkpoints/`. The app now says exactly what is missing and `scripts/install_checkpoints.py` installs them.
- Professional docs plus this file.

---

## 3. How to run everything (my checklist)

### First time

```powershell
cd C:\Users\priya\OneDrive\Desktop\Attack2Defense
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[api,dev]"
cd frontend; npm install; cd ..
```

### Every time

```powershell
# terminal 1
python -m uvicorn backend.app.main:create_app --factory --reload --port 8000
# terminal 2
cd frontend; npm run dev
```

Open http://localhost:5173. On this laptop the API takes a few minutes to start because importing PyTorch is slow; wait for "Application startup complete".

### Check what is installed

```powershell
python -m attack2defense doctor
```

Right now it reports that the three ResNet-18 `best.pt` files are missing. The reported numbers still show; only live attacks on ResNet-18 need the weights.

### Install the ResNet-18 weights (when I find them on Google Drive)

1. Download `best.pt` for each run. Keep them in folders named after the run (`resnet18_pgd/best.pt`), or rename the files `resnet18_pgd.pt`.
2. `python scripts/install_checkpoints.py C:\Users\priya\Downloads\weights` (a `.zip` works too).
3. `python -m attack2defense doctor` should now say everything is ready.

### Demo for someone (5 minutes)

1. **Start here**: read the headline question and point at the three bars (93.6, 6.1, 78.9).
2. **Attack lab**: model "No defense", any image, PGD, 8/255. Show "Fooled", the invisible change and the "What just happened" card.
3. Tick **Compare with another model**, pick the FGSM- or PGD-trained model, and run the same attack. The defended model holds.
4. **Dashboard**: the stronger-attacks chart shows the defense is specific to the budget it was trained for.
5. Optionally **Training**: start a Quick demo and watch PGD accuracy rise per epoch.

### Before submitting or a viva

```powershell
python -m pytest tests -q
python -m ruff check .
python -m ruff format --check .
cd frontend
node node_modules\vitest\vitest.mjs run
node node_modules\vite\bin\vite.js build
```

(`npm test` and `npm run build` do the same if the npm shim works.)

---

## 4. The logic, piece by piece

Notation: \(x\) is an image with pixels in \([0, 1]\), \(y\) its true label, \(f\) the model, \(L\) the cross-entropy loss, \(\epsilon\) the budget.

### 4.1 Pixel space and `PixelModel`

**Code:** `attack2defense/models.py`, `PixelModel.forward`

\[ f(x) = \text{backbone}\left(\frac{x - \mu}{\sigma}\right) \]

Normalization is inside the model. So when an attack changes \(x\) by \(2/255\), that really is two brightness levels on every channel. If normalization happened in the data loader, the attack would work in normalized units, and the same \(\epsilon\) would mean different amounts for red, green and blue.

### 4.2 The threat model

- **L-infinity**: every channel of every pixel may change by at most \(\epsilon\): \(\|x' - x\|_\infty \le \epsilon\).
- **Untargeted**: any wrong answer counts as success.
- **White-box**: the attacker knows the model and uses its gradient.
- **True label**: the attack pushes away from the correct class \(y\), as in Madry et al.
- \(\epsilon = 2/255 \approx 0.00784\).

### 4.3 Projection

**Code:** `attacks.py`, `project`

\[ \Pi(x') = \operatorname{clip}_{[0,1]}\big(\operatorname{clip}_{[x-\epsilon,\, x+\epsilon]}(x')\big) \]

After every step, each pixel is pulled back into the budget box around the original image, then into the valid range. This is why the "Largest change" in the Attack lab never exceeds the budget.

### 4.4 FGSM (Goodfellow et al.)

**Code:** `attacks.py`, `fgsm_attack`

\[ x' = \Pi\big(x + \epsilon \cdot \operatorname{sign}(\nabla_x L(f(x), y))\big) \]

One step, the full budget, in the direction that increases the loss fastest for each pixel. In code it is `pgd_attack` with `steps=1`, `alpha=epsilon` and no random start. Fast, but it assumes the loss is linear, so it is a weak attack.

### 4.5 PGD (Madry et al.)

**Code:** `attacks.py`, `pgd_attack`, `run_attack`

\[ x^{(0)} = \Pi\big(x + U(-\epsilon, \epsilon)\big), \qquad x^{(t+1)} = \Pi\big(x^{(t)} + \alpha \cdot \operatorname{sign}(\nabla_x L(f(x^{(t)}), y))\big) \]

- Many small steps, each followed by projection. It follows the loss surface instead of assuming it is flat, so it finds much stronger examples.
- **Step size**: \(\alpha = 2.5\,\epsilon / \text{steps}\). With 10 steps that is \(\epsilon/4\). The factor 2.5 means the steps add up to more than the budget, so PGD can cross the whole box and still have room to adjust.
- **Random start** avoids starting exactly at \(x\), where the gradient can be misleading.
- **`reduction="sum"`** in the loss: only the sign of the gradient is used, so scaling does not matter, and a sum keeps gradients from underflowing in float16.
- **Trace**: `success_per_step` records the share of images fooled after each step; it is the "Was the model fooled after each step?" chart.

### 4.6 Adversarial training

**Code:** `train.py`, `_training_inputs` and `train_one_epoch`

Madry's min-max objective:

\[ \min_\theta \; \mathbb{E}_{(x,y)} \Big[ \max_{\|\delta\|_\infty \le \epsilon} L\big(f_\theta(x + \delta), y\big) \Big] \]

The inner max is approximated by an attack; the outer min is ordinary training on the result. For each batch:

1. Take `count = round(adv_ratio * batch_size)` images (the batch is shuffled, so taking the first ones is random).
2. Attack them with the current model:
   - mode `fgsm` uses **FGSM-RS** (Wong et al.): random start, then one step of \(1.25\,\epsilon\), projected.
   - mode `pgd` uses PGD with `pgd_steps` steps (5 in the reported run, 3 by default in the app).
3. Concatenate them with the clean rest of the batch and do a normal forward, loss, backward and optimizer step.

Other modes: `clean` uses the batch as is; `gaussian` adds \(N(0, \sigma^2)\) noise (for randomized smoothing).

Cost: FGSM training is about 2x a normal epoch (one extra forward and backward); PGD with \(k\) steps is about \((k+1)\)x.

**Catastrophic overfitting**: one-step training can suddenly learn to defeat FGSM only, so FGSM accuracy is high but PGD accuracy collapses. The training loop warns if validation FGSM accuracy is more than 20 points above PGD accuracy. Random start plus \(1.25\epsilon\) is what keeps FGSM-RS stable.

### 4.7 Choosing `best.pt`

**Code:** `train.py`, `run_training`

- Validation images come from the **training** split: 100 per class (1,000 total) held out with seed 42. The app's quick runs use 20 per class.
- After every epoch: clean, FGSM and PGD accuracy on validation (PGD with 3 steps, to keep it quick).
- Score = PGD accuracy for defended runs, clean accuracy for the clean run. `best.pt` is saved only when the score improves; `last.pt` is saved every epoch for resuming.
- The **test** set is never used for this choice, so the reported test numbers are honest.

### 4.8 Evaluation

**Code:** `evaluate.py`, `evaluate_loader`, `evaluate_sweep`

For every batch of test images: clean accuracy, then FGSM accuracy, then PGD-10 accuracy, **on the same images**. A sweep repeats this for several budgets. `flag_below_threshold` marks any robust accuracy below the target (80%). The app's Evaluation screen does the same on a class-balanced subset, for several models at once, so the comparison is fair.

Files per run: `summary.csv`, `per_class.csv`, `metrics.json`, `sweep.csv`, plots and the attack example grid.

### 4.9 Randomized smoothing (certified defense)

**Code:** `smoothing.py`

The smoothed classifier predicts the class the base model chooses most often under Gaussian noise:

\[ g(x) = \arg\max_c \; P\big(f(x + \eta) = c\big), \quad \eta \sim N(0, \sigma^2 I) \]

To certify one image (`certify_image`):

1. Draw `n0` noisy copies and take the most common class as the guess \(\hat c\).
2. Draw `n` fresh noisy copies and count the votes \(k\) for \(\hat c\).
3. Compute a lower bound \(\underline{p_A}\) on the true probability with the one-sided **Clopper-Pearson** interval at confidence \(1-\alpha\) (`clopper_pearson_lower`, a binary search on the binomial tail).
4. If \(\underline{p_A} > 0.5\), the **certified radius** is

\[ R = \sigma \, \Phi^{-1}(\underline{p_A}) \]

   otherwise the model abstains.

Meaning: no change with \(\|\delta\|_2 < R\) can change \(g\)'s answer, whatever attack is used. **Certified accuracy at radius r** is the share of images that are correct and have \(R \ge r\).

Important: this is an **L2** guarantee about the *smoothed* model. It is a different kind of number from the empirical L-infinity PGD accuracy, so they are never mixed in one table.

---

## 5. Where everything is in the code

| I want to... | Open |
| --- | --- |
| See the attack math | `attack2defense/attacks.py` (`project`, `pgd_attack`, `run_attack`) |
| See how a defended batch is built | `attack2defense/train.py`, `_training_inputs` |
| See how `best.pt` is chosen | `attack2defense/train.py`, `run_training` (search `best_score`) |
| See how accuracy is measured | `attack2defense/evaluate.py`, `evaluate_loader` |
| See the certified radius | `attack2defense/smoothing.py`, `certify_image`, `certified_radius` |
| See how models are found | `attack2defense/registry.py`, `discover_models` |
| See what "missing weights" checks | `attack2defense/setup_check.py`, `inspect_setup` |
| Change server limits or folders | `backend/app/settings.py` (or `A2D_*` env vars) |
| See an endpoint | `backend/app/routers/<area>.py` |
| See how a job runs in the background | `backend/app/services/jobs.py` and `services/tasks.py` |
| Change a screen | `frontend/src/pages/<Screen>.tsx` |
| Change a definition shown by "?" | `frontend/src/lib/glossary.ts` |
| Change colours or fonts | `frontend/src/index.css` (`@theme`) and `src/lib/tokens.ts` |
| Change buttons, sliders, cards | `frontend/src/components/ui.tsx` |

---

## 6. What happens when I click "Run attack"

1. `AttackLab.tsx` turns the settings into a request with `toRequest`: `model_id`, `sample_indices: [index]`, `epsilon = levels / 255`, `steps` (1 for FGSM), `alpha` (or `null` for automatic), `random_start`.
2. `api.attack` in `src/api/endpoints.ts` sends `POST /api/attacks/pgd` with the token from `localStorage`.
3. FastAPI validates the body against `AttackRequest` in `schemas.py`; for example an epsilon above 32/255 is rejected with a readable message.
4. `routers/attacks.py`, `_attack`: `ModelStore.get` finds the model, `ModelStore.load` loads `best.pt` (cached after the first time), and `DatasetService.items` fetches the test image.
5. The model predicts the clean image; then `run_attack` from `attack2defense/attacks.py` runs PGD.
6. The model predicts the attacked image. The router computes `fooled`, L-infinity and L2, and encodes the original, attacked, change and heatmap images as PNG data URLs (`services/imaging.py`).
7. The UI shows the verdict, images, explanation and measurements. With "Update automatically" on, moving a slider repeats this after a 300 ms pause.

Training, evaluation and certification are the same except step 4 creates a **job**: the API saves it as `queued`, returns `202` immediately, and a background thread runs it while the UI asks for its progress every 1.5 seconds.

---

## 7. Pitfalls I hit (and how they were solved)

| Problem | Cause | Fix |
| --- | --- | --- |
| Epsilon meant different things per channel | Attacking normalized tensors | `PixelModel` normalizes inside the model |
| FGSM training looked great, PGD collapsed | Catastrophic overfitting of one-step training | FGSM-RS (random start + 1.25 epsilon), warning in the training loop |
| Gradients vanished under mixed precision | Mean loss is tiny per pixel in float16 | `reduction="sum"`; only the sign is used |
| ResNet-18 too slow on the laptop | ~100 minutes per adversarial epoch on CPU | Train on Colab; `SmallCIFARCNN` for CPU demos |
| ResNet-18 can't be attacked live | `best.pt` files never copied from Drive | `doctor`, `/api/setup`, setup checklist, `install_checkpoints.py` |
| A restart left jobs "running" forever | The worker thread died with the server | `fail_interrupted_jobs` marks them failed on startup |
| Users could guess other users' job ids | Returning 403 confirms the id exists | Return 404 for other users' jobs |
| `npx` fails on this Windows machine | Broken npm shim (`npm-prefix.js`) | Run tools with `node node_modules\...` |
| Registration rejected `@attack2defense.local` | Email validation rejects reserved domains | Use a normal domain such as `example.com` |
| Frontend tests timed out | Lazy pages import Recharts slowly on a busy machine | Longer test and async timeouts in `vite.config.ts` and `src/test/setup.ts` |

---

## 8. How to explain the results honestly

**Say:**

- The undefended model is excellent on clean images and nearly useless under attack: 93.6% to 6.1%.
- PGD adversarial training recovers 78.9% (+72.8 points) for 3.0 points of clean accuracy.
- FGSM training gets close (77.9%) at much lower cost, because FGSM-RS avoided catastrophic overfitting.
- The 80% target is missed by 1.1 points. Validation PGD accuracy plateaued around 80% from epoch 27 (peak 80.2% at epoch 28), but validation uses a weaker 3-step PGD, so the 10-step test number is a little lower. Closing the gap probably needs a wider network (for example WideResNet) or extra data, not just more epochs.
- The defense is specific to its budget: at 8/255 (four times larger) PGD accuracy drops to 24.7%.
- FGSM accuracy is always above PGD accuracy because PGD is the stronger attack. That is why PGD is the headline number.

**Do not claim:**

- That the model is secure in general. It was tested against FGSM and PGD at fixed budgets, not adaptive attacks or other threat models.
- That the 80% target was met.
- That certified accuracy and PGD accuracy are comparable; they measure different things (L2 proof versus L-infinity attack).

### Likely questions

| Question | Answer |
| --- | --- |
| Why 2/255? | Small enough to be invisible, big enough to break an undefended model. Common in the CIFAR-10 literature. |
| Why 10 PGD steps for evaluation but 5 in training? | Training pays the attack cost on every batch; 5 steps is a good trade-off. Evaluation uses more steps so the reported number is a stronger, more honest attack. |
| Why does clean accuracy drop with the defense? | The model gives up some fine, fragile features that attacks exploit. This robustness-accuracy trade-off is well known. |
| Why validate on training data? | So the test set never influences which checkpoint is kept; the reported numbers stay unbiased. |
| What does `adv_ratio` do? | The share of each batch replaced by attacked images. Lower keeps more clean accuracy but gives less robustness. The reported runs use 1.0. |
| Is FGSM-RS the same as FGSM? | No. FGSM evaluation starts from the clean image with a step of epsilon; FGSM-RS training starts from a random point and steps 1.25 epsilon, then projects. |
| What does the certified radius mean? | A proof that no L2 change smaller than R can change the smoothed model's answer. |
| Why a web app at all? | So the result is explorable and reproducible without notebooks: anyone can train, attack and compare models with the same code that produced the report. |

---

## 9. Housekeeping notes

- `outputs/smoke/` and `outputs/smoke2/` are leftovers from smoke tests and can be deleted. `outputs/defense/` (a CPU FGSM run from 1 October) and `outputs/dry_pgd/` (a short PGD dry run) are real but small models; the app does not scan them. To show one in the app, move it into `outputs/cpu_preview/<name>/`.
- My demo account is `demo@example.com`. Deleting `backend/attack2defense.db` resets every account and job.
- After any new Colab run: `build_report.py`, then `install_checkpoints.py`, then `doctor`, then regenerate the PDF with `python scripts/md_to_pdf.py`.
