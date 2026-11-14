import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/endpoints";
import { useAuth } from "../auth/AuthContext";
import { AccuracyVsEpsilon } from "../components/charts";
import { Badge, buttonClass, Card, CopyCommand, EmptyState, KpiCard, Notice, PageHeader, Spinner, Term } from "../components/ui";
import { epsilonLabel, percent } from "../lib/format";
import { useResource } from "../lib/hooks";
import { architectureName, headline, sortModels } from "../lib/results";

function Bar({ value, tone }: { value: number; tone: "clean" | "adv" | "muted" }) {
  const color = { clean: "bg-clean", adv: "bg-adv", muted: "bg-muted" }[tone];
  return (
    <div className="flex items-center gap-3">
      <span className="w-14 text-right font-mono tabular-nums">{percent(value)}</span>
      <span className="hidden h-1.5 w-20 overflow-hidden rounded-full bg-raised md:block" aria-hidden="true">
        <span className={`block h-full rounded-full ${color}`} style={{ width: `${value * 100}%` }} />
      </span>
    </div>
  );
}

export function Overview() {
  const { user } = useAuth();
  const results = useResource(api.publicResults);
  const models = useMemo(() => sortModels(results.data?.models ?? []), [results.data]);
  const { baseline, defended, epsilon, target } = headline(models);
  const [shown, setShown] = useState<"baseline" | "defended">("baseline");
  const exampleModel = shown === "baseline" ? baseline : defended;

  if (results.loading) return <Spinner label="Loading the reported results" />;
  if (results.error) return <Notice tone="error">{results.error.message}</Notice>;
  if (!models.length) {
    return (
      <EmptyState
        title="No reported results yet"
        action={<CopyCommand command="python scripts/build_report.py --results results.zip" />}
      >
        Run the Colab notebook, put results.zip in the project folder, and unzip it into results/. Then refresh
        this page.
      </EmptyState>
    );
  }

  const b = baseline?.summary;
  const d = defended?.summary;
  const featuredArchitecture = baseline?.architecture ?? defended?.architecture;
  const swept = models.filter((model) => model.sweep.length > 1 && model.architecture === featuredArchitecture);

  return (
    <div className="space-y-10">
      <PageHeader
        title="Dashboard"
        actions={
          <>
            <Link to={user ? "/attack" : "/login"} state={{ from: "/attack" }} className={buttonClass("attack")}>
              {user ? "Attack a model" : "Log in to attack"}
            </Link>
            <Link to="/guide" className={buttonClass("secondary")}>
              How this works
            </Link>
          </>
        }
      >
        The reported results from the full GPU training run. Every number is measured on the{" "}
        {b?.images.toLocaleString() ?? "held-out"} CIFAR-10 test images, which no model saw during training, with
        an attack budget of <Term id="epsilon">{epsilon ? epsilonLabel(epsilon) : "2/255"}</Term> per pixel.
      </PageHeader>

      {b && d && (
        <section aria-label="Headline numbers" className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <KpiCard label="No defense, clean images" value={percent(b.clean_accuracy)} tone="clean" hint="How good the model is normally." />
          <KpiCard label="No defense, under PGD" value={percent(b.pgd_accuracy)} tone="adv" hint="What the invisible attack does to it." />
          <KpiCard
            label={`${defended!.label}, under PGD`}
            value={percent(d.pgd_accuracy)}
            tone="clean"
            hint={`+${((d.pgd_accuracy - b.pgd_accuracy) * 100).toFixed(1)} points recovered by the defense.`}
          />
          <KpiCard
            label="Cost of the defense"
            value={`${((b.clean_accuracy - d.clean_accuracy) * 100).toFixed(1)} pts`}
            hint="Clean accuracy given up in exchange for robustness."
          />
        </section>
      )}

      {baseline && defended && (
        <Card
          title="See the attack on real test images"
          description="One test image per class. Rows show the clean image, the FGSM attack, the PGD attack, and the PGD change amplified so you can see it. Green titles are correct predictions, red titles are wrong."
        >
          <div role="tablist" aria-label="Model shown in the example grid" className="mb-4 inline-flex gap-1 rounded-lg border border-line bg-paper p-1">
            {(["baseline", "defended"] as const).map((key) => (
              <button
                key={key}
                role="tab"
                type="button"
                aria-selected={shown === key}
                onClick={() => setShown(key)}
                className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                  shown === key ? (key === "baseline" ? "bg-adv-soft text-adv" : "bg-clean-soft text-clean") : "text-muted hover:text-ink"
                }`}
              >
                {key === "baseline" ? `${baseline.label} (fooled)` : `${defended.label} (holds)`}
              </button>
            ))}
          </div>
          {exampleModel && (
            <div className="overflow-hidden rounded-lg bg-white p-2">
              <img
                src={api.examplesUrl(exampleModel.id)}
                alt={`The first test image of each class before and after FGSM and PGD, classified by ${exampleModel.label}.`}
                className="w-full"
              />
            </div>
          )}
        </Card>
      )}

      <Card
        title="Accuracy before and after attack"
        description={
          <>
            <Term id="cleanAccuracy">Clean</Term> is normal accuracy. <Term id="fgsm">FGSM</Term> and{" "}
            <Term id="pgd">PGD</Term> show <Term id="robustAccuracy">robust accuracy</Term> after each attack. The
            last column compares PGD accuracy to the {percent(target, 0)} target.
          </>
        }
      >
        <div className="-mx-5 overflow-x-auto sm:-mx-6">
          <table className="w-full min-w-[52rem] text-sm">
            <thead className="border-y border-line text-left text-muted">
              <tr>
                <th scope="col" className="px-5 py-3 font-medium sm:px-6">Model</th>
                <th scope="col" className="px-4 py-3 font-medium">Clean</th>
                <th scope="col" className="px-4 py-3 font-medium">FGSM</th>
                <th scope="col" className="px-4 py-3 font-medium">PGD</th>
                <th scope="col" className="px-4 py-3 font-medium">Target</th>
              </tr>
            </thead>
            <tbody>
              {models
                .filter((model) => model.summary)
                .map((model) => {
                  const summary = model.summary!;
                  const gap = target - summary.pgd_accuracy;
                  return (
                    <tr key={model.id} className="border-b border-line last:border-0">
                      <th scope="row" className="min-w-[15rem] px-5 py-3 text-left font-medium sm:px-6">
                        <span className="flex items-center gap-2">
                          {model.label}
                          {model.has_weights && <Badge tone="success">Live</Badge>}
                        </span>
                        <span className="block text-xs font-normal text-muted">
                          {architectureName(model.architecture)}, {model.epochs} epochs
                        </span>
                      </th>
                      <td className="px-4 py-3"><Bar value={summary.clean_accuracy} tone="clean" /></td>
                      <td className="px-4 py-3"><Bar value={summary.fgsm_accuracy} tone="muted" /></td>
                      <td className="px-4 py-3"><Bar value={summary.pgd_accuracy} tone={summary.pgd_accuracy >= 0.5 ? "clean" : "adv"} /></td>
                      <td className="px-4 py-3">
                        {gap <= 0 ? <Badge tone="success">Met</Badge> : <Badge tone={gap < 0.05 ? "warn" : "adv"}>{(gap * 100).toFixed(1)} pts short</Badge>}
                      </td>
                    </tr>
                  );
                })}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs text-muted">
          "Live" models have weights on this server and can be attacked in the lab. The others show reported
          numbers only; see <Link to="/guide#setup" className="link">setup status</Link>.
        </p>
      </Card>

      {swept.length > 0 && (
        <Card
          title="What happens with stronger attacks"
          description={`${architectureName(swept[0].architecture)} models. The defenses were trained for ${epsilon ? epsilonLabel(epsilon) : "one budget"}. Bigger budgets show how quickly that protection runs out: robustness is specific to the budget it was trained for.`}
        >
          <AccuracyVsEpsilon
            series={swept.map((model) => ({ id: model.id, label: model.label, rows: model.sweep }))}
            threshold={target}
          />
        </Card>
      )}

      {results.data!.zeroshot.length > 0 && (
        <Card
          title="Earlier experiment: ImageNet models on CIFAR cats"
          description="Before training on CIFAR-10, pretrained ResNet-50 and ViT-B/16 were tested on cat images only. A prediction counts as correct when the ImageNet label is any kind of cat. Both are just as fragile."
        >
          <div className="-mx-5 overflow-x-auto sm:-mx-6">
            <table className="w-full min-w-[32rem] text-sm">
              <thead className="border-y border-line text-left text-muted">
                <tr>
                  <th scope="col" className="px-5 py-3 font-medium sm:px-6">Model</th>
                  <th scope="col" className="px-4 py-3 font-medium">Cat images</th>
                  <th scope="col" className="px-4 py-3 font-medium">Clean</th>
                  <th scope="col" className="px-4 py-3 font-medium">FGSM</th>
                  <th scope="col" className="px-4 py-3 font-medium">PGD</th>
                </tr>
              </thead>
              <tbody>
                {results.data!.zeroshot.map((row) => (
                  <tr key={String(row.model)} className="border-b border-line last:border-0">
                    <th scope="row" className="px-5 py-3 text-left font-medium sm:px-6">{architectureName(String(row.model))}</th>
                    <td className="px-4 py-3 font-mono tabular-nums">{row.images}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-clean">{percent(Number(row.clean_accuracy))}</td>
                    <td className="px-4 py-3 font-mono tabular-nums">{percent(Number(row.fgsm_accuracy))}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-adv">{percent(Number(row.pgd_accuracy))}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
