import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/endpoints";
import type { ModelEntry } from "../api/types";
import { TrainingCurves } from "../components/charts";
import { Badge, Button, buttonClass, Card, EmptyState, Notice, PageHeader, Spinner } from "../components/ui";
import { epsilonLabel, percent, sentenceCase } from "../lib/format";
import { useResource } from "../lib/hooks";
import { architectureName, sortModels } from "../lib/results";

const PAGE = 24;

const CONFIG_LABELS: Record<string, string> = {
  model: "Architecture",
  mode: "Defense",
  epsilon: "Attack budget",
  pgd_steps: "PGD steps",
  adv_ratio: "Adversarial share",
  epochs: "Epochs",
  batch_size: "Batch size",
  lr: "Learning rate",
  optimizer_name: "Optimizer",
  scheduler_name: "Scheduler",
  seed: "Random seed",
};

function SampleBrowser({ classes }: { classes: string[] }) {
  const [label, setLabel] = useState<string | undefined>();
  const [limit, setLimit] = useState(PAGE);
  const [selected, setSelected] = useState<number | null>(null);
  const samples = useResource(() => api.samples("cifar10", { label, limit }), [label, limit]);

  return (
    <Card
      title="Test images"
      description="The held-out test split. No model trained on these. Select one to attack it in the lab."
      actions={
        selected !== null ? (
          <Link to={`/attack?sample=${selected}`} className={buttonClass("attack", "sm")}>
            Attack image #{selected}
          </Link>
        ) : undefined
      }
    >
      <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by class">
        {[undefined, ...classes].map((name) => (
          <button
            key={name ?? "all"}
            type="button"
            aria-pressed={label === name}
            onClick={() => {
              setLabel(name);
              setLimit(PAGE);
              setSelected(null);
            }}
            className={`rounded-full border px-3 py-1 text-sm transition-colors ${
              label === name ? "border-clean bg-clean-soft text-ink" : "border-line text-muted hover:border-line-strong hover:text-ink"
            }`}
          >
            {name ? sentenceCase(name) : "All classes"}
          </button>
        ))}
      </div>

      {samples.error && (
        <div className="mt-4">
          <Notice tone="error">{samples.error.message}</Notice>
        </div>
      )}
      {samples.data && (
        <>
          <ul className="mt-4 grid grid-cols-4 gap-2 sm:grid-cols-6 lg:grid-cols-8">
            {samples.data.items.map((sample) => (
              <li key={sample.index}>
                <button
                  type="button"
                  onClick={() => setSelected(sample.index)}
                  aria-pressed={selected === sample.index}
                  className={`block w-full overflow-hidden rounded-md border-2 ${
                    selected === sample.index ? "border-adv" : "border-transparent hover:border-line-strong"
                  }`}
                >
                  <img src={sample.image} alt={`Test image ${sample.index}, a ${sample.label}`} className="pixelated aspect-square w-full" />
                </button>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm">
            <span className="text-muted">
              Showing {samples.data.items.length} of {samples.data.total.toLocaleString()}
            </span>
            {samples.data.items.length < samples.data.total && (
              <Button variant="secondary" size="sm" busy={samples.loading} onClick={() => setLimit(limit + PAGE)}>
                Show more
              </Button>
            )}
          </div>
        </>
      )}
      {samples.loading && !samples.data && (
        <div className="mt-4">
          <Spinner label="Loading images" />
        </div>
      )}
    </Card>
  );
}

function ModelDetail({ id }: { id: string }) {
  const detail = useResource(() => api.model(id), [id]);
  if (detail.loading) return <Spinner label="Loading model" />;
  if (detail.error) return <Notice tone="error">{detail.error.message}</Notice>;
  const model = detail.data!;
  const config = Object.entries(model.config).filter(([, value]) => value !== null && value !== undefined);

  return (
    <div className="space-y-6">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3 lg:grid-cols-4">
        {config.map(([key, value]) => (
          <div key={key}>
            <dt className="text-xs text-muted">{CONFIG_LABELS[key] ?? key.replace(/_/g, " ")}</dt>
            <dd className="font-mono tabular-nums">
              {key === "epsilon" && typeof value === "number"
                ? epsilonLabel(value)
                : key === "model"
                  ? architectureName(String(value))
                  : String(value)}
            </dd>
          </div>
        ))}
      </dl>
      {model.history.length > 0 ? (
        <TrainingCurves history={model.history} />
      ) : (
        <p className="text-sm text-muted">No training history was saved for this model.</p>
      )}
      {model.has_examples && (
        <figure className="overflow-hidden rounded-lg bg-white p-2">
          <img src={api.examplesUrl(model.id)} alt={`Before and after attack examples for ${model.label}`} className="w-full" />
        </figure>
      )}
    </div>
  );
}

function ModelCard({ model, open, onToggle }: { model: ModelEntry; open: boolean; onToggle: () => void }) {
  const tone = model.mode === "clean" ? "adv" : "success";
  return (
    <li className={`panel flex flex-col p-4 ${open ? "border-clean" : ""}`}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-semibold">{model.label}</p>
          <p className="text-xs text-muted">
            {architectureName(model.architecture)}
            {model.epochs ? `, ${model.epochs} epochs` : ""}
            {model.owner === "you" ? ", your run" : ""}
          </p>
        </div>
        {model.has_weights ? <Badge tone="success">Live</Badge> : <Badge>Results only</Badge>}
      </div>
      {model.summary ? (
        <dl className="mt-4 grid grid-cols-2 gap-3">
          <div>
            <dt className="text-xs text-muted">Clean</dt>
            <dd className="font-mono text-lg font-semibold tabular-nums text-clean">{percent(model.summary.clean_accuracy)}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Under PGD</dt>
            <dd className={`font-mono text-lg font-semibold tabular-nums ${model.summary.pgd_accuracy >= 0.5 ? "text-clean" : "text-adv"}`}>
              {percent(model.summary.pgd_accuracy)}
            </dd>
          </div>
        </dl>
      ) : (
        <p className="mt-4 text-sm text-muted">Not tested yet. Test numbers appear when training finishes.</p>
      )}
      <div className="mt-4 flex items-center justify-between gap-2 border-t border-line pt-3">
        <Badge tone={tone}>{model.mode === "clean" ? "Undefended" : `Defense: ${model.mode.toUpperCase()}`}</Badge>
        <Button variant="ghost" size="sm" onClick={onToggle} aria-expanded={open}>
          {open ? "Hide details" : "Details"}
        </Button>
      </div>
      {!model.has_weights && (
        <p className="mt-2 text-xs text-muted">
          Add <span className="font-mono">checkpoints/{model.id}/best.pt</span> to attack it live.
        </p>
      )}
    </li>
  );
}

function ModelGrid({ models }: { models: ModelEntry[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const opened = models.find((model) => model.id === open);

  return (
    <section aria-labelledby="models-heading" className="space-y-4">
      <div>
        <h2 id="models-heading" className="text-xl font-semibold">
          Trained models
        </h2>
        <p className="mt-1 text-sm text-muted">
          Every model the server found. "Live" models have weights here and can be attacked; the rest show their reported
          results only.
        </p>
      </div>
      <ul className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {models.map((model) => (
          <ModelCard key={model.id} model={model} open={open === model.id} onToggle={() => setOpen(open === model.id ? null : model.id)} />
        ))}
      </ul>
      {opened && (
        <Card title={`${opened.label}, ${architectureName(opened.architecture)}`} description={<span className="font-mono">{opened.id}</span>}>
          <ModelDetail id={opened.id} />
        </Card>
      )}
    </section>
  );
}

export function ModelsData() {
  const datasets = useResource(api.datasets);
  const models = useResource(api.models);
  const cifar = datasets.data?.[0];

  return (
    <div className="space-y-10">
      <PageHeader title="Models and data">
        The dataset the models learned from, the test images you can attack, and every trained model with its numbers.
      </PageHeader>

      {models.loading && <Spinner label="Loading models" />}
      {models.error && <Notice tone="error">{models.error.message}</Notice>}
      {models.data &&
        (models.data.length ? (
          <ModelGrid models={sortModels(models.data)} />
        ) : (
          <EmptyState
            title="No models found"
            action={
              <Link to="/training" className={buttonClass("primary")}>
                Train a model
              </Link>
            }
          >
            Train one on the Training page, or install the Colab checkpoints into checkpoints/.
          </EmptyState>
        ))}

      {datasets.error && <Notice tone="error">{datasets.error.message}</Notice>}
      {cifar && (
        <Card title={cifar.title} description={cifar.description}>
          <dl className="grid grid-cols-3 gap-4 text-sm sm:max-w-md">
            {Object.entries(cifar.splits).map(([split, count]) => (
              <div key={split}>
                <dt className="capitalize text-muted">{split}</dt>
                <dd className="font-mono text-xl font-semibold tabular-nums">{count.toLocaleString()}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-muted">
            Validation is 100 training images per class, fixed by seed 42 and used only to choose the best checkpoint. The test
            split is used only for the reported numbers.
          </p>
        </Card>
      )}
      {cifar && <SampleBrowser classes={cifar.classes} />}
    </div>
  );
}
