import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

import { api } from "../api/endpoints";
import { useAuth } from "../auth/AuthContext";
import { AccuracyTrack, type TrackRow } from "../components/AccuracyTrack";
import { SetupStatus } from "../components/SetupStatus";
import { buttonClass, Card } from "../components/ui";
import { epsilonLabel, percent } from "../lib/format";
import { glossary, type GlossaryKey } from "../lib/glossary";
import { useResource } from "../lib/hooks";
import { architectureName, headline } from "../lib/results";
import { guideFlag } from "../lib/setup";

const STEPS: { title: string; body: string; screen: string; to: string; action: string }[] = [
  {
    title: "Train an ordinary classifier",
    body: "A ResNet-18 learns to sort 32x32 CIFAR-10 photos into ten classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck. On clean test images it is right more than 9 times out of 10.",
    screen: "Dashboard",
    to: "/",
    action: "See the baseline numbers",
  },
  {
    title: "Attack it with invisible noise",
    body: "FGSM and PGD use the model's own gradients to find the worst tiny change for each image. Every pixel moves by at most 2 brightness levels out of 255, yet the model's accuracy collapses.",
    screen: "Attack lab",
    to: "/attack",
    action: "Fool a model yourself",
  },
  {
    title: "Defend it with adversarial training",
    body: "Retrain the model on attacked images, made fresh every batch against its current weights. It learns features the attack cannot easily push around, at a small cost in clean accuracy.",
    screen: "Training",
    to: "/training",
    action: "Train a defended model",
  },
  {
    title: "Compare, then prove",
    body: "Evaluate every model on the same held-out images at several attack strengths and flag anything under the 80% target. Randomized smoothing goes further and proves a safe radius for each image.",
    screen: "Evaluation and Certified defense",
    to: "/evaluation",
    action: "Compare models",
  },
];

const KEY_TERMS: GlossaryKey[] = [
  "epsilon",
  "fgsm",
  "pgd",
  "adversarialTraining",
  "cleanAccuracy",
  "robustAccuracy",
  "advRatio",
  "certifiedRadius",
];

export function Guide() {
  const { user } = useAuth();
  const results = useResource(api.publicResults);
  const { baseline, defended, epsilon, target } = headline(results.data?.models ?? []);

  const location = useLocation();

  useEffect(() => guideFlag.markSeen(), []);
  useEffect(() => {
    if (location.hash === "#setup") document.getElementById("setup")?.scrollIntoView({ block: "start" });
  }, [location.hash, location.key]);

  const rows: TrackRow[] =
    baseline?.summary && defended?.summary
      ? [
          { label: "No defense,", detail: "ordinary test images", value: baseline.summary.clean_accuracy, tone: "clean" },
          { label: "No defense,", detail: "after a PGD attack", value: baseline.summary.pgd_accuracy, tone: "adv" },
          { label: `${defended.label},`, detail: "after the same attack", value: defended.summary.pgd_accuracy, tone: "clean" },
        ]
      : [];

  const levels = epsilon ? Math.round(epsilon * 255) : 2;

  return (
    <div className="space-y-14">
      <section className="relative overflow-hidden rounded-2xl border border-line bg-surface">
        <div className="pixel-grid pointer-events-none absolute inset-0 opacity-40 [mask-image:linear-gradient(to_bottom_right,black,transparent_70%)]" aria-hidden="true" />
        <div className="relative grid gap-10 p-6 sm:p-10 lg:grid-cols-[1.05fr_1fr] lg:items-center">
          <div>
            <h1 className="text-4xl font-semibold leading-[1.1] sm:text-5xl">
              Can a neural network be fooled by changes you cannot see?
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
              Yes. Shift every pixel of a photo by {levels} brightness levels out of 255 in exactly the wrong
              direction, and a model that was{" "}
              {baseline?.summary ? percent(baseline.summary.clean_accuracy, 0) : "over 90%"} accurate starts guessing. Attack2Defense measures that failure on
              CIFAR-10, then trains a model that holds up under the same attack.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link to={user ? "/attack" : "/login"} state={{ from: "/attack" }} className={buttonClass("attack")}>
                {user ? "Open the attack lab" : "Log in and try an attack"}
              </Link>
              <Link to="/" className={buttonClass("secondary")}>
                See all results
              </Link>
            </div>
          </div>

          <div className="rounded-xl border border-line bg-paper/70 p-5 backdrop-blur-sm">
            {rows.length ? (
              <>
                <p className="mb-4 text-sm text-muted">
                  {architectureName(baseline?.architecture)} on {baseline?.summary?.images.toLocaleString()} unseen
                  test images, attack budget {epsilon ? epsilonLabel(epsilon) : "2/255"}
                </p>
                <AccuracyTrack rows={rows} target={target} />
              </>
            ) : (
              <p className="text-sm text-muted">
                {results.loading
                  ? "Loading the reported results..."
                  : "No reported results yet. Unzip results.zip from the Colab run into results/ to see the headline numbers here."}
              </p>
            )}
          </div>
        </div>
      </section>

      <section aria-labelledby="how-heading">
        <h2 id="how-heading" className="text-2xl font-semibold">
          How the project works
        </h2>
        <p className="mt-2 max-w-2xl text-muted">
          Four stages, each with its own screen. Follow them in order the first time.
        </p>
        <ol className="mt-6 grid gap-4 md:grid-cols-2">
          {STEPS.map((step, index) => (
            <li key={step.title} className="panel flex flex-col p-5">
              <div className="flex items-center gap-3">
                <span
                  aria-hidden="true"
                  className={`flex h-8 w-8 items-center justify-center rounded-full font-mono text-sm font-semibold ${
                    index === 1 ? "bg-adv-soft text-adv" : "bg-clean-soft text-clean"
                  }`}
                >
                  {index + 1}
                </span>
                <h3 className="text-lg font-semibold">{step.title}</h3>
              </div>
              <p className="mt-3 flex-1 text-sm leading-relaxed text-muted">{step.body}</p>
              <div className="mt-4 flex items-center justify-between gap-3 border-t border-line pt-3">
                <span className="text-xs text-muted">Screen: {step.screen}</span>
                <Link to={step.to} className="link text-sm">
                  {step.action}
                </Link>
              </div>
            </li>
          ))}
        </ol>
      </section>

      {baseline?.summary && defended?.summary && (
        <section aria-labelledby="takeaway-heading" className="grid gap-4 md:grid-cols-3">
          <h2 id="takeaway-heading" className="sr-only">
            What the numbers mean
          </h2>
          <div className="panel p-5">
            <p className="text-sm font-semibold text-adv">The attack works</p>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Without a defense, PGD drops accuracy by{" "}
              {((baseline.summary.clean_accuracy - baseline.summary.pgd_accuracy) * 100).toFixed(1)} points. The
              changed images look identical to a person.
            </p>
          </div>
          <div className="panel p-5">
            <p className="text-sm font-semibold text-clean">The defense works</p>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Adversarial training recovers {percent(defended.summary.pgd_accuracy)} under the same attack, for{" "}
              {((baseline.summary.clean_accuracy - defended.summary.clean_accuracy) * 100).toFixed(1)} points of
              clean accuracy.
            </p>
          </div>
          <div className="panel p-5">
            <p className="text-sm font-semibold text-warn">The limits are real</p>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              The {percent(target, 0)} target is{" "}
              {defended.summary.pgd_accuracy >= target
                ? "met"
                : `${((target - defended.summary.pgd_accuracy) * 100).toFixed(1)} points away`}
              , and protection fades for attacks bigger than the {epsilon ? epsilonLabel(epsilon) : "trained"} budget.
            </p>
          </div>
        </section>
      )}

      <section aria-labelledby="terms-heading">
        <h2 id="terms-heading" className="text-2xl font-semibold">
          Key ideas in plain English
        </h2>
        <p className="mt-2 max-w-2xl text-muted">
          These words appear throughout the app. Look for the small question-mark buttons next to controls for the
          same explanations in context.
        </p>
        <dl className="mt-6 grid gap-x-8 gap-y-5 md:grid-cols-2">
          {KEY_TERMS.map((key) => {
            const entry = glossary[key];
            return (
              <div key={key} className="border-l-2 border-line-strong pl-4">
                <dt className="font-semibold">{entry.term}</dt>
                <dd className="mt-1 text-sm leading-relaxed text-muted">
                  {entry.short}
                  {"long" in entry && entry.long && <span className="mt-1 block text-ink/70">{entry.long}</span>}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <div id="setup" className="scroll-mt-20">
        <Card
          title="Setup status"
          description="What this server has installed. Live attacks need model weights; the reported numbers need only the results folder."
        >
          <SetupStatus />
        </Card>
      </div>
    </div>
  );
}
