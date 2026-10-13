import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/endpoints";
import type { ModelEntry } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { AccuracyVsEpsilon } from "../components/charts";
import { EmptyState, Notice, Spinner } from "../components/ui";
import { epsilonLabel, percent } from "../lib/format";
import { useResource } from "../lib/hooks";

const modeOrder = ["clean", "fgsm", "pgd", "gaussian"];

function sortModels(models: ModelEntry[]): ModelEntry[] {
  return [...models].sort(
    (a, b) => a.architecture.localeCompare(b.architecture) || modeOrder.indexOf(a.mode) - modeOrder.indexOf(b.mode),
  );
}

function AccuracyBar({ value, tone }: { value: number; tone: "clean" | "adv" }) {
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 text-right tabular-nums">{percent(value)}</span>
      <span className="hidden h-1.5 w-16 overflow-hidden rounded-full bg-ink/10 lg:block" aria-hidden="true">
        <span className={`block h-full ${tone === "clean" ? "bg-clean" : "bg-adv"}`} style={{ width: `${value * 100}%` }} />
      </span>
    </div>
  );
}

export function Overview() {
  const { user } = useAuth();
  const results = useResource(api.publicResults);
  const models = useMemo(() => sortModels(results.data?.models ?? []), [results.data]);

  const featured = models.filter((model) => model.architecture === models[0]?.architecture);
  const baseline = featured.find((model) => model.mode === "clean");
  const defended = featured
    .filter((model) => model.mode !== "clean" && model.summary)
    .sort((a, b) => (b.summary?.pgd_accuracy ?? 0) - (a.summary?.pgd_accuracy ?? 0))[0];
  const [shown, setShown] = useState<"baseline" | "defended">("baseline");
  const exampleModel = shown === "baseline" ? baseline : defended;

  if (results.loading) return <Spinner label="Loading published results" />;
  if (results.error) return <Notice tone="error">{results.error.message}</Notice>;
  if (!models.length) {
    return (
      <EmptyState title="No published results yet">
        Run the Colab notebook, unzip results.zip into the results folder, and refresh this page.
      </EmptyState>
    );
  }

  const epsilon = baseline?.summary?.epsilon ?? defended?.summary?.epsilon;

  return (
    <div className="space-y-12">
      <section className="grid items-start gap-8 xl:grid-cols-[minmax(0,26rem)_1fr]">
        <div>
          <h1 className="text-3xl font-semibold leading-tight sm:text-4xl">
            What a {epsilon ? epsilonLabel(epsilon) : "tiny"} nudge does to a CIFAR-10 classifier
          </h1>
          <p className="mt-4 text-muted">
            Every pixel moves by at most {epsilon ? Math.round(epsilon * 255) : "a few"} intensity levels out of 255, too
            little to see.
            {baseline?.summary && defended?.summary && (
              <>
                {" "}
                Without a defense, accuracy falls from {percent(baseline.summary.clean_accuracy)} to{" "}
                <strong className="text-adv">{percent(baseline.summary.pgd_accuracy)}</strong> under PGD. With{" "}
                {defended.label} it stays at{" "}
                <strong className="text-clean">{percent(defended.summary.pgd_accuracy)}</strong>.
              </>
            )}
          </p>
          <p className="mt-4 text-sm text-muted">
            Measured on the {baseline?.summary?.images.toLocaleString() ?? "held-out"} CIFAR-10 test images, which no
            model saw during training.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Link
              to={user ? "/attack" : "/login"}
              state={{ from: "/attack" }}
              className="rounded-md bg-ink px-4 py-2 text-sm font-medium text-white hover:bg-ink/90"
            >
              {user ? "Attack a model yourself" : "Log in to attack a model"}
            </Link>
            <Link to="/models" className="rounded-md border border-line bg-surface px-4 py-2 text-sm font-medium">
              Browse models and data
            </Link>
          </div>
        </div>

        {baseline && defended && (
          <figure className="panel p-4">
            <div role="tablist" aria-label="Model shown in the example grid" className="mb-3 flex gap-1">
              {(["baseline", "defended"] as const).map((key) => (
                <button
                  key={key}
                  role="tab"
                  type="button"
                  aria-selected={shown === key}
                  onClick={() => setShown(key)}
                  className={`rounded-md px-3 py-1.5 text-sm ${
                    shown === key ? (key === "baseline" ? "bg-adv-soft text-adv" : "bg-clean-soft text-clean") : "text-muted hover:bg-ink/5"
                  }`}
                >
                  {key === "baseline" ? baseline.label : defended.label}
                </button>
              ))}
            </div>
            {exampleModel && (
              <img
                src={api.examplesUrl(exampleModel.id)}
                alt={`The first test image of each class before and after FGSM and PGD, classified by ${exampleModel.label}. Green titles are correct, red are wrong.`}
                className="w-full rounded"
              />
            )}
            <figcaption className="mt-2 text-xs text-muted">
              Rows: clean image, FGSM, PGD, and the PGD change amplified so it is visible. Green titles are correct
              predictions, red are wrong.
            </figcaption>
          </figure>
        )}
      </section>

      <section aria-labelledby="results-heading">
        <h2 id="results-heading" className="text-xl font-semibold">
          Accuracy before and after attack
        </h2>
        <div className="panel mt-4 overflow-x-auto">
          <table className="w-full min-w-[48rem] text-sm">
            <thead className="border-b border-line text-left text-muted">
              <tr>
                <th scope="col" className="px-4 py-3 font-medium">Model</th>
                <th scope="col" className="px-4 py-3 font-medium">Clean</th>
                <th scope="col" className="px-4 py-3 font-medium">FGSM</th>
                <th scope="col" className="px-4 py-3 font-medium">PGD</th>
                <th scope="col" className="px-4 py-3 font-medium">Trained on</th>
              </tr>
            </thead>
            <tbody>
              {models
                .filter((model) => model.summary)
                .map((model) => (
                  <tr key={model.id} className="border-b border-line last:border-0">
                    <th scope="row" className="min-w-[14rem] px-4 py-3 text-left font-medium">
                      {model.label}
                      <span className="block text-xs font-normal text-muted">
                        {model.architecture}, {model.epochs} epochs
                      </span>
                    </th>
                    <td className="px-4 py-3">
                      <AccuracyBar value={model.summary!.clean_accuracy} tone="clean" />
                    </td>
                    <td className="px-4 py-3 tabular-nums">{percent(model.summary!.fgsm_accuracy)}</td>
                    <td className="px-4 py-3">
                      <AccuracyBar value={model.summary!.pgd_accuracy} tone="adv" />
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-xs text-muted">{model.tag}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>

      {models.some((model) => model.sweep.length > 1) && (
        <section aria-labelledby="sweep-heading">
          <h2 id="sweep-heading" className="text-xl font-semibold">
            Stronger attacks
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            The defenses were trained for {epsilon ? epsilonLabel(epsilon) : "one budget"}. Larger budgets show how
            quickly that protection runs out.
          </p>
          <div className="panel mt-4 p-4">
            <AccuracyVsEpsilon
              series={models
                .filter((model) => model.sweep.length > 1)
                .map((model) => ({ id: model.id, label: `${model.label} (${model.architecture})`, rows: model.sweep }))}
              threshold={baseline?.summary?.target ?? 0.8}
            />
          </div>
        </section>
      )}

      {results.data!.zeroshot.length > 0 && (
        <section aria-labelledby="zeroshot-heading">
          <h2 id="zeroshot-heading" className="text-xl font-semibold">
            Last week: ImageNet models on CIFAR cats
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-muted">
            Pretrained ResNet-50 and ViT-B/16, never trained on CIFAR-10. A prediction counts as correct when the
            ImageNet label is a kind of cat.
          </p>
          <div className="panel mt-4 overflow-x-auto">
            <table className="w-full min-w-[28rem] text-sm">
              <thead className="border-b border-line text-left text-muted">
                <tr>
                  <th scope="col" className="px-4 py-3 font-medium">Model</th>
                  <th scope="col" className="px-4 py-3 font-medium">Cat images</th>
                  <th scope="col" className="px-4 py-3 font-medium">Clean</th>
                  <th scope="col" className="px-4 py-3 font-medium">FGSM</th>
                  <th scope="col" className="px-4 py-3 font-medium">PGD</th>
                </tr>
              </thead>
              <tbody>
                {results.data!.zeroshot.map((row) => (
                  <tr key={String(row.model)} className="border-b border-line last:border-0">
                    <th scope="row" className="px-4 py-3 text-left font-medium">{String(row.model)}</th>
                    <td className="px-4 py-3 tabular-nums">{row.images}</td>
                    <td className="px-4 py-3 tabular-nums">{percent(Number(row.clean_accuracy))}</td>
                    <td className="px-4 py-3 tabular-nums">{percent(Number(row.fgsm_accuracy))}</td>
                    <td className="px-4 py-3 tabular-nums">{percent(Number(row.pgd_accuracy))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
