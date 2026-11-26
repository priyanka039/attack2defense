import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";

import { api } from "../api/endpoints";
import { useAuth } from "../auth/AuthContext";
import { AccuracyTrack, type TrackRow } from "../components/AccuracyTrack";
import { SetupStatus } from "../components/SetupStatus";
import { buttonClass, Card } from "../components/ui";
import { epsilonLabel, percent } from "../lib/format";
import { glossary, type GlossaryEntry, type GlossaryKey } from "../lib/glossary";
import { useResource } from "../lib/hooks";
import { architectureName, headline } from "../lib/results";
import { guideFlag } from "../lib/setup";

const STEPS: { title: string; body: string; to: string; action: string }[] = [
  {
    title: "Train a baseline classifier",
    body: "A ResNet-18 is trained on CIFAR-10: 32×32 colour images in ten classes, from airplane to truck. Without any defense it classifies over 90% of unseen test images correctly.",
    to: "/",
    action: "View baseline results",
  },
  {
    title: "Attack it",
    body: "FGSM and PGD use the model's own gradients to find, for each image, the small perturbation that most increases its error. No pixel changes by more than 2 intensity levels out of 255.",
    to: "/attack",
    action: "Open the attack lab",
  },
  {
    title: "Defend it with adversarial training",
    body: "The model is retrained on attacked images, regenerated every batch against its current weights. It learns features the attack cannot easily exploit, at a small cost in clean accuracy.",
    to: "/training",
    action: "Train a defended model",
  },
  {
    title: "Evaluate and certify",
    body: "Every model is tested on the same held-out images at several attack strengths and checked against the 80% target. Randomized smoothing adds a provable robustness radius for each image.",
    to: "/evaluation",
    action: "Compare models",
  },
];

const KEY_TERMS: GlossaryKey[] = [
  "cleanAccuracy",
  "robustAccuracy",
  "epsilon",
  "fgsm",
  "pgd",
  "adversarialTraining",
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
          { label: "Undefended model", detail: "Clean test images", value: baseline.summary.clean_accuracy, tone: "clean" },
          { label: "Undefended model", detail: "Under PGD attack", value: baseline.summary.pgd_accuracy, tone: "adv" },
          { label: defended.label, detail: "Under the same attack", value: defended.summary.pgd_accuracy, tone: "clean" },
        ]
      : [];

  const budget = epsilon ? epsilonLabel(epsilon) : "2/255";
  const specs: [string, string][] = baseline?.summary
    ? [
        ["Model", architectureName(baseline.architecture)],
        ["Test set", `${baseline.summary.images.toLocaleString()} CIFAR-10 images`],
        ["Attack", `PGD, ${baseline.summary.pgd_steps} steps`],
        ["Budget (ε)", budget],
      ]
    : [];

  return (
    <div className="space-y-14">
      <section className="relative overflow-hidden rounded-2xl border border-line bg-surface">
        <div className="pixel-grid pointer-events-none absolute inset-0 opacity-40 [mask-image:linear-gradient(to_bottom_right,black,transparent_70%)]" aria-hidden="true" />
        <div className="relative grid gap-10 p-6 sm:p-10 lg:grid-cols-[1.05fr_1fr] lg:items-center">
          <div>
            <h1 className="text-4xl font-semibold leading-[1.1] sm:text-5xl">
              Adversarial robustness for image classifiers
            </h1>
            <p className="mt-5 max-w-xl text-lg leading-relaxed text-muted">
              A perturbation of at most {budget} per pixel, invisible to the human eye, can take an image classifier
              from {baseline?.summary ? percent(baseline.summary.clean_accuracy) : "over 90%"} accuracy to
              {baseline?.summary ? ` ${percent(baseline.summary.pgd_accuracy)}` : " near zero"}. Attack2Defense
              reproduces that failure on CIFAR-10 and measures how much of it adversarial training recovers.
            </p>
            <div className="mt-7 flex flex-wrap gap-3">
              <Link to={user ? "/attack" : "/login"} state={{ from: "/attack" }} className={buttonClass("primary")}>
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
                <h2 className="text-sm font-semibold">Reported result</h2>
                <dl className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 border-b border-line pb-4 text-xs">
                  {specs.map(([name, value]) => (
                    <div key={name}>
                      <dt className="text-muted">{name}</dt>
                      <dd className="mt-0.5 font-medium text-ink">{value}</dd>
                    </div>
                  ))}
                </dl>
                <div className="mt-4">
                  <AccuracyTrack rows={rows} target={target} />
                </div>
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
          The workflow has four stages, each with its own screen. Working through them in order is the quickest way
          to understand the results.
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
              <div className="mt-4 border-t border-line pt-3">
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
          <div className="panel border-t-2 border-t-adv p-5">
            <h3 className="text-sm font-semibold">Attack impact</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              PGD lowers the undefended model's accuracy by{" "}
              {((baseline.summary.clean_accuracy - baseline.summary.pgd_accuracy) * 100).toFixed(1)} percentage
              points, although the attacked images are visually indistinguishable from the originals.
            </p>
          </div>
          <div className="panel border-t-2 border-t-clean p-5">
            <h3 className="text-sm font-semibold">Defense recovery</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              Adversarial training restores accuracy under the same attack to{" "}
              {percent(defended.summary.pgd_accuracy)}, at a cost of{" "}
              {((baseline.summary.clean_accuracy - defended.summary.clean_accuracy) * 100).toFixed(1)} points on
              clean images.
            </p>
          </div>
          <div className="panel border-t-2 border-t-warn p-5">
            <h3 className="text-sm font-semibold">Limitations</h3>
            <p className="mt-2 text-sm leading-relaxed text-muted">
              The {percent(target, 0)} target is{" "}
              {defended.summary.pgd_accuracy >= target
                ? "met"
                : `missed by ${((target - defended.summary.pgd_accuracy) * 100).toFixed(1)} points`}
              , and protection weakens against attacks larger than the {budget} budget used in training.
            </p>
          </div>
        </section>
      )}

      <section aria-labelledby="terms-heading">
        <h2 id="terms-heading" className="text-2xl font-semibold">
          Key terms
        </h2>
        <p className="mt-2 max-w-2xl text-muted">
          Definitions used throughout the app. The ? button beside a control shows the same definition in context.
        </p>
        <dl className="panel mt-6 divide-y divide-line">
          {KEY_TERMS.map((key) => {
            const entry: GlossaryEntry = glossary[key];
            return (
              <div key={key} className="grid gap-2 p-5 md:grid-cols-[minmax(12rem,18rem)_1fr] md:gap-8">
                <dt className="text-[15px] font-semibold leading-snug">
                  {entry.term}
                  {entry.abbr && <span className="ml-1.5 font-normal text-muted">({entry.abbr})</span>}
                </dt>
                <dd className="max-w-2xl text-sm leading-relaxed text-muted">
                  <p className="text-ink">{entry.short}</p>
                  {entry.formula && (
                    <code className="mt-3 block w-fit max-w-full overflow-x-auto rounded-md border border-line bg-raised px-3 py-2 font-mono text-[13px] text-ink">
                      {entry.formula}
                    </code>
                  )}
                  {entry.long && <p className="mt-2">{entry.long}</p>}
                </dd>
              </div>
            );
          })}
        </dl>
      </section>

      <div id="setup" className="scroll-mt-20">
        <Card
          title="Setup status"
          description="For visitors: if it says Demo ready, you can explore. Developer install details stay collapsed underneath."
        >
          <SetupStatus />
        </Card>
      </div>
    </div>
  );
}
