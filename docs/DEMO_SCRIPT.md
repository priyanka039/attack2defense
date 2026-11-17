# Demo script: walking through the Attack2Defense UI

A spoken walkthrough of the app, about 10 minutes, with the reasoning behind every screen. Lines in quotes are what to say; **Do** lines are what to click; **Why it is built this way** is the design reasoning to use when someone asks "why did you design it like that?".

A 2-minute version and likely questions are at the end.

---

## Before you start (5 minutes before)

1. Start the app (two terminals):

   ```powershell
   cd C:\Users\priya\OneDrive\Desktop\Attack2Defense
   python -m uvicorn backend.app.main:create_app --factory --port 8000
   ```

   ```powershell
   cd C:\Users\priya\OneDrive\Desktop\Attack2Defense\frontend
   npm run dev
   ```

   Wait for `Application startup complete` in the first terminal (up to a few minutes on this laptop).

2. Open http://localhost:5173 and log in (`demo@example.com`).
3. Warm up the models so the live demo is fast: open http://localhost:5173/attack?sample=3 once and let the first attack finish. The first attack on each model loads its weights (a few seconds); after that, attacks take about 2 seconds.
4. Pick a theme. Dark looks best on a projector in a dim room; switch to **Light** (bottom of the sidebar) in a bright room or for screenshots.
5. To show the Start here page again as a first-time visitor would see it, open http://localhost:5173/guide directly.

**Tested demo images** (ResNet-18, PGD, budget 2/255). PGD starts from a random point, so the *wrong* answer can change between runs (for example truck one time, ship the next); whether the model is fooled does not.

| Image | Undefended model | PGD-trained model | Use it for |
| --- | --- | --- | --- |
| #3 airplane | "truck" or "ship", about 100% sure | "airplane", 94% sure | The main demo |
| #1 ship | "automobile", 100% sure | "ship", 99% sure | A second clean example |
| #12 dog | "horse", fooled | "cat", also fooled | Honest limits: the defense is not perfect |

---

## 1. Start here (1.5 minutes)

**Do:** open http://localhost:5173/guide.

"This project asks one question, and it's the title of the page: *can a neural network be fooled by changes you cannot see?*"

**Do:** point at the three bars on the right.

"The answer is in these three bars. A ResNet-18 trained on CIFAR-10 gets 93.6% of ten thousand unseen test images right. Then I attack it, changing every pixel by at most 2 brightness levels out of 255, and it drops to 6.1%. That's worse than guessing among ten classes. With adversarial training, the same model holds 78.9% under the same attack. The amber line is my 80% target, and I miss it by 1.1 points."

**Do:** scroll to "How the project works".

"The project has four stages, and each card links to the screen where that stage happens: train a normal classifier, attack it, defend it, then compare fairly and prove robustness."

**Do:** scroll to "Key ideas in plain English", then to "Setup status".

"Every technical word in the app is defined here, and the same definitions appear wherever you see a small question mark. At the bottom, the setup checklist shows what this server has installed. Everything is green, so every model can be attacked live."

**Why it is built this way**

- **The question is the headline.** A first-time visitor doesn't know what FGSM or epsilon means, but anyone understands "can it be fooled by changes you can't see?". Leading with the question gives them a reason to keep reading.
- **The result comes first.** The three bars tell the whole story (works, breaks, recovers) before any explanation. Someone who leaves after ten seconds still takes away the main finding.
- **All three bars share one 0–100% axis**, so the size of the drop and of the recovery can be compared by eye. Separate charts would hide that.
- **The target line is always visible**, including when it isn't met. Showing the 1.1-point gap openly is more credible than hiding it.
- **First visitors land here automatically**; returning visitors go straight to the Dashboard. The guide is for orientation, not a hurdle.
- **The setup checklist turns a confusing failure into an instruction.** Without it, missing weights would just look like a broken Attack lab. The checklist names the exact file and the command that fixes it.

---

## 2. Dashboard (1.5 minutes)

**Do:** click **Dashboard** in the sidebar.

"The Dashboard holds the reported results from the full GPU training run. Every number is on the full ten thousand-image test set, which no model saw during training."

**Do:** point at the four number cards.

"Four numbers summarise it: clean accuracy, accuracy under attack without a defense, accuracy under attack with the defense, and what the defense costs. That's three points of clean accuracy."

**Do:** in "See the attack on real test images", switch between the **No defense (fooled)** and **PGD adversarial training (holds)** tabs.

"These are real test images. The rows are clean, FGSM-attacked and PGD-attacked versions, plus the change amplified so you can see it. Green titles are correct predictions and red are wrong. On the undefended model the attacked rows go almost entirely red. On the defended model they stay mostly green, even though the images look the same."

**Do:** scroll to the results table, then to "What happens with stronger attacks".

"The table compares every model on the same three measures and shows how far each one is from the target. The chart is the honest part: the defense was trained for 2/255. At 8/255, four times larger, PGD accuracy falls to about 25%. Robustness is specific to the budget it was trained for."

**Why it is built this way**

- **It's public, with no login**, because reading results costs the server nothing. A reviewer can check the claims without creating an account.
- **The tabs show the same images on both models.** Comparing the same pictures isolates the effect of the defense; different pictures would mix in luck.
- **Every image has a correct-or-wrong title**, so you don't have to judge by eye whether the model was fooled.
- **The sweep chart shows the defense's weakness on purpose.** A results page that shows only good numbers invites distrust; showing where the defense fades shows you understand it.
- **Colour has one meaning everywhere**: teal for clean or defended, pink for attacked or fooled, amber for the target. After one screen, people read colour without a legend.

---

## 3. Attack lab (3 minutes, the centrepiece)

**Do:** open http://localhost:5173/attack?sample=3 (or Models and data → the airplane → **Attack image #3**).

"This is where you can fool a model yourself. It's three steps, top to bottom: choose a model, choose an image, set the attack."

**Do:** step 1, choose **No defense (ResNet-18)**. Step 3, choose **PGD** and strength **2/255**.

"I'm attacking the undefended ResNet-18 with PGD, the strong attack, at 2/255, the budget the defense was trained for."

**Do:** point at the result.

"The banner says **Fooled**: the model was 99.5% sure this is an airplane, and now it says truck (or ship; PGD starts from a random point, so the wrong answer varies), with almost 100% confidence. Look at the original and the attacked image side by side. They look identical. The middle panel is the change, amplified so it's visible at all."

**Do:** point at "What just happened".

"This card explains the result in plain words. The attack used the model's own gradient to find which direction each pixel should move to increase its error. It took ten small steps and pulled every pixel back inside the budget after each one. The model's confidence in 'airplane' didn't just drop; airplane fell out of its top three answers entirely."

**Do:** point at Measurements and the step chart.

"The largest change to any pixel is exactly the budget, 2/255, never more. The chart shows how many steps the attack needed before the model gave in."

**Do:** tick **Compare with another model** and choose **PGD adversarial training (ResNet-18)**.

"Now the key comparison: the same image, the same attack, the same budget, but on the model trained with adversarial training. It **holds**. It still says airplane, 94% sure. That's the whole project in one screen."

**Do (optional):** click **Attack 16 random images** in Batch check.

"One image could be luck, so the batch check attacks sixteen random images. On the undefended model the attack wins almost every time; on the defended model, rarely."

**Do (optional, honest limits):** open http://localhost:5173/attack?sample=12 with compare on.

"And here's an image where both models are fooled. The defense makes attacks fail most of the time, not always. That matches the 78.9%: about one in five images still gets through."

**Why it is built this way**

- **Three numbered steps** follow the order you'd explain an attack in: what you attack, what you attack it with, how hard. The Attack lab used to be one wall of sliders; numbering the steps tells a newcomer where to start.
- **Strength presets are in brightness levels (2/255, 8/255, 16/255)**, not raw decimals like 0.00784. Levels out of 255 are something people can picture, and each preset says what it means ("trained budget", "paper standard", "very strong").
- **Advanced PGD settings are folded away.** Step size, number of steps and random start matter to experts but would stop a beginner. The defaults follow Madry et al., so hiding them loses nothing.
- **The verdict comes first, in a coloured banner.** The one thing everyone wants to know is "did it work?", so it's the first thing you see; the details follow below.
- **"What just happened" turns numbers into a story** and uses the actual values from this run (the true label, the confidence before and after), so it's never generic text.
- **The change panel is amplified and labelled as such.** At true scale it would be invisible, which is the point, so the UI shows both: the attacked image that looks the same, and the amplified change that proves something happened.
- **Compare mode exists because the claim is comparative.** "The defense works" only means something next to "the baseline fails" on the same input. Showing them side by side is the most direct evidence the app can give.
- **Updates are automatic**: change a setting and the attack re-runs after a 300 ms pause. Seeing accuracy react as you move the budget teaches more than pressing a button repeatedly.
- **Batch check guards against cherry-picking.** A single dramatic image is persuasive but not evidence; sixteen random images are a fairer sample.

---

## 4. Training (1.5 minutes)

**Do:** click **Training**.

"This is where the defense is made. You pick a defense from four options, and each card says in one line what it does and what it costs."

**Do:** point at the four defense cards and the **How long** presets.

"No defense is ordinary training. FGSM training attacks each batch with one step, which is about twice the cost. PGD training uses several steps: it's the slowest and the most robust. Noise training is for the certified defense. Under How long, Quick demo trains a small network in about a minute on this laptop; the estimate below uses this server's actual device."

**Do:** open an existing run under "Your runs" (or start a Quick demo).

"While a run trains, you see a progress bar and two charts per epoch: validation accuracy on clean, FGSM and PGD images, and the training loss. When it finishes, it's tested on held-out images and you can attack it straight away."

"The demo runs are deliberately short. They show the method working, with PGD accuracy rising as it trains. The reported numbers came from 30 full epochs of ResNet-18 on a GPU."

**Why it is built this way**

- **The defense is the first choice**, because it's the experiment's independent variable. Everything else is a detail of how long you train.
- **Every option states its cost.** Adversarial training is expensive, and saying "about 2x slower" up front sets expectations before someone waits ten minutes.
- **Presets with a time estimate** mean nobody has to guess epochs and batch counts. *Custom* still exposes everything for experts.
- **Long work runs as a background job.** Training takes minutes, so the page shows live progress instead of freezing, and you can leave and come back. Jobs run one at a time, because each uses the whole CPU or GPU.
- **Clean, FGSM and PGD accuracy are plotted together**, because the gap between them is the diagnosis. If FGSM accuracy rises but PGD doesn't, the model learned a shortcut (catastrophic overfitting); the backend also warns about this.
- **A finished run links straight to the Attack lab and Evaluation**, which closes the loop: train, then immediately test what you trained.

---

## 5. Evaluation (1 minute)

**Do:** click **Evaluation**.

"Evaluation is the fair comparison. Every selected model faces the same images and the same attacks at each budget."

**Do:** tick the three ResNet-18 models, choose **Quick check**, and point at the pass mark slider (or open an existing report).

"I choose the models, how thorough to be, the attack budgets and a pass mark. The report flags every result below the pass mark, shows a table and a chart, and exports to CSV or JSON for a report."

**Why it is built this way**

- **Same images, same attacks, same budgets for every model.** That's what makes a comparison fair; the Attack lab is for intuition, and this screen is for evidence.
- **A pass mark turns numbers into decisions.** "Below 80%" is easier to act on than a grid of decimals, and the amber summary says how many results failed.
- **Export exists because results end up in reports.** Copying numbers by hand from a screen is how mistakes get in.
- **Only models with weights are listed**, because evaluation attacks every model live. Results-only models are still on the Dashboard.

---

## 6. Certified defense (1 minute)

**Do:** click **Certified defense**.

"Everything so far is empirical: we attacked and the model held. But an attack can only prove a model *failed*. A cleverer attack might still exist. These two cards explain the difference: *tested* means PGD couldn't fool it, and *proven* means no change smaller than r can change the answer, whatever the attacker does."

"Randomized smoothing gives that proof. The model classifies hundreds of noisy copies of the image and takes a majority vote. If the vote is confident enough, there's a mathematical guarantee of a safe radius. The curve shows what share of images are certified at each radius."

**Why it is built this way**

- **Tested and proven are deliberately separate**, in different colours, and are never mixed into one table. They measure different things (an L2 proof versus an L-infinity attack), and blending them would overstate the result.
- **It opens with the reason it exists** ("attacks can only show a model failed") before any settings, because the motivation is less obvious than for an attack.
- **The sigma slider and the Quick and Careful presets** expose the one real trade-off: more noise gives larger guarantees but lower accuracy, and more samples give tighter proofs but take longer.

---

## 7. Models and data (30 seconds)

**Do:** click **Models and data**.

"This screen shows what the system knows about. Each model card shows its defense and its clean and PGD numbers, plus a status: **Live** means its weights are on this server and it can be attacked. Below are the dataset sizes and the actual test images. Click one, then **Attack image**, and it opens in the lab."

**Why it is built this way**

- **Live versus Results only** answers "why can't I attack this model?" before anyone has to ask.
- **The dataset card states the split** (49,000 training, 1,000 validation, 10,000 test) and that the test set is only used for the reported numbers. That's the claim that makes the results trustworthy, so it's on screen, not buried in code.

---

## 8. Close (30 seconds)

"To sum up: a standard image classifier can be pushed from 94% to 6% accuracy by changes nobody can see. Adversarial training brings it back to 79% under the same attack, at a cost of three points on clean images. The defense is specific to the budget it was trained for, and it's 1.1 points short of my target. The app makes every step of that visible and repeatable: you can train, attack, compare and certify from the browser, and it runs the same code that produced the reported numbers."

---

## Design principles behind the whole UI (if asked "how did you design it?")

1. **Lead with the outcome.** Every screen puts the answer first (the bars, the verdict banner, the flags summary) and the detail after.
2. **Colour has one meaning.** Teal is clean or defended, pink is attack or fooled, amber is the target or a warning. It's never decorative, so it can be read without a legend. In light mode the same colours are darker shades, keeping text readable on white.
3. **Plain words first, jargon on demand.** Labels say "Attack budget" and "Brightness levels out of 255"; the technical term and the formula are one click away in a "?".
4. **Sensible defaults, power hidden, not removed.** Presets cover the common case; *Custom* and *Advanced* expose everything.
5. **Every empty or error state says what to do next.** No model with weights? It links to Training and shows the install command. Not logged in? It explains why an account is needed and what's still available.
6. **Honesty is a feature.** The missed target, the budget-specific robustness and the failure cases are all shown in the UI, not hidden.
7. **Accessible by default.** Real radio groups and labelled sliders, keyboard-usable tooltips (Escape closes them), visible focus rings, a skip link, reduced-motion support, and a layout that works on a phone.
8. **The UI never computes results itself.** It calls the API, which calls the same Python package used to train and report. What you see in the browser is the actual research code running.

---

## The 2-minute version

1. **Start here:** "Can a network be fooled by changes you can't see? Yes: 93.6% drops to 6.1%. With adversarial training it holds at 78.9%."
2. **Attack lab, image #3:** undefended ResNet-18, PGD, 2/255. "A confident airplane becomes a confident truck or ship, and the images look identical."
3. **Compare** with the PGD-trained model: "Same image, same attack, and it holds: airplane, 94%."
4. **Dashboard sweep chart:** "The protection is specific to the trained budget, and I'm 1.1 points short of 80%. The app shows both."

---

## Likely questions

| Question | Answer |
| --- | --- |
| Why does it need a login? | Attacks and training use server compute and save runs per person. Reading results is free and public. |
| Why not show everything on one page? | Each screen matches one stage of the method (learn, attack, defend, compare, prove). One page would mix goals and overwhelm a newcomer. |
| Why presets instead of free inputs? | Most people don't know good values for epochs or PGD steps. Presets give a correct result fast; Custom and Advanced keep full control. |
| Why does the attacked image look identical? | That's the point: the change is at most 2 of 255 brightness levels per pixel. The amplified panel shows it exists. |
| Is the defended model safe? | Safer against this attack at this budget: about 79% of images hold. It isn't proven safe against every attack; that's what the Certified screen addresses, and it's a different, L2 guarantee. |
| Why dark mode by default? | It suits the "security lab" theme and is easy on the eyes for long sessions. Light mode is one click away, and the app follows your system setting on a first visit. |
| What if the server has no weights? | The sidebar dot turns amber, the Start here checklist names the missing file, and the Dashboard still shows the reported numbers. |

## If something goes wrong during the demo

| Symptom | Fix |
| --- | --- |
| Red dot, "API offline" | The API isn't running or is still starting. Wait for `Application startup complete`, then refresh |
| The first attack is slow | That model's weights are loading. The next attacks are fast; warm up before the demo |
| Logged out | Tokens last 12 hours. Log in again |
| A job says "Queued" | Another job is running; jobs run one at a time |
