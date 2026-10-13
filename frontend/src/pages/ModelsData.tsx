import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api/endpoints";
import type { ModelEntry } from "../api/types";
import { TrainingCurves } from "../components/charts";
import { Button, EmptyState, Notice, PageHeader, Spinner } from "../components/ui";
import { epsilonLabel, percent, sentenceCase } from "../lib/format";
import { useResource } from "../lib/hooks";

const PAGE = 24;

function SampleBrowser({ classes }: { classes: string[] }) {
  const [label, setLabel] = useState<string | undefined>();
  const [limit, setLimit] = useState(PAGE);
  const [selected, setSelected] = useState<number | null>(null);
  const samples = useResource(() => api.samples("cifar10", { label, limit }), [label, limit]);

  return (
    <section aria-labelledby="samples-heading" className="panel p-5">
      <h2 id="samples-heading" className="text-lg font-semibold">
        Test images
      </h2>
      <p className="mt-1 text-sm text-muted">Pick an image to attack. These come from the held-out test split.</p>

      <div className="mt-4 flex flex-wrap gap-1.5" role="group" aria-label="Filter by class">
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
            className={`rounded-md border px-2.5 py-1 text-sm ${
              label === name ? "border-ink bg-ink text-white" : "border-line bg-surface hover:border-ink/40"
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
                  className={`block w-full overflow-hidden rounded border-2 ${
                    selected === sample.index ? "border-clean" : "border-transparent hover:border-line"
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
            <div className="flex gap-2">
              {selected !== null && (
                <Link to={`/attack?sample=${selected}`} className="rounded-md bg-ink px-3 py-1.5 font-medium text-white">
                  Attack image {selected}
                </Link>
              )}
              {samples.data.items.length < samples.data.total && (
                <Button variant="secondary" busy={samples.loading} onClick={() => setLimit(limit + PAGE)}>
                  Show more
                </Button>
              )}
            </div>
          </div>
        </>
      )}
      {samples.loading && !samples.data && (
        <div className="mt-4">
          <Spinner label="Loading images" />
        </div>
      )}
    </section>
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
      <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3 lg:grid-cols-4">
        {config.map(([key, value]) => (
          <div key={key}>
            <dt className="text-muted">{key.replace(/_/g, " ")}</dt>
            <dd className="font-medium tabular-nums">
              {key === "epsilon" && typeof value === "number" ? epsilonLabel(value) : String(value)}
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
        <figure>
          <img
            src={api.examplesUrl(model.id)}
            alt={`Before and after attack examples for ${model.label}`}
            className="w-full rounded border border-line"
          />
        </figure>
      )}
    </div>
  );
}

function ModelTable({ models }: { models: ModelEntry[] }) {
  const [open, setOpen] = useState<string | null>(null);

  return (
    <section aria-labelledby="models-heading">
      <h2 id="models-heading" className="text-lg font-semibold">
        Model registry
      </h2>
      <p className="mt-1 text-sm text-muted">
        Every checkpoint the server found, with the attack it was trained against. Models without weights show their
        reported results only.
      </p>
      <div className="panel mt-4 overflow-x-auto">
        <table className="w-full min-w-[46rem] text-sm">
          <thead className="border-b border-line text-left text-muted">
            <tr>
              <th scope="col" className="px-4 py-3 font-medium">Model</th>
              <th scope="col" className="px-4 py-3 font-medium">Trained on</th>
              <th scope="col" className="px-4 py-3 font-medium">Clean</th>
              <th scope="col" className="px-4 py-3 font-medium">PGD</th>
              <th scope="col" className="px-4 py-3 font-medium">Weights</th>
              <th scope="col" className="px-4 py-3"><span className="sr-only">Details</span></th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <ModelRow key={model.id} model={model} open={open === model.id} onToggle={() => setOpen(open === model.id ? null : model.id)} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ModelRow({ model, open, onToggle }: { model: ModelEntry; open: boolean; onToggle: () => void }) {
  return (
    <>
      <tr className="border-b border-line">
        <th scope="row" className="px-4 py-3 text-left font-medium">
          {model.label}
          <span className="block text-xs font-normal text-muted">
            {model.architecture}
            {model.owner === "you" ? ", your run" : ""}
          </span>
        </th>
        <td className="px-4 py-3 font-mono text-xs">{model.tag}</td>
        <td className="px-4 py-3 tabular-nums">{percent(model.summary?.clean_accuracy)}</td>
        <td className="px-4 py-3 tabular-nums">{percent(model.summary?.pgd_accuracy)}</td>
        <td className="px-4 py-3">
          {model.has_weights ? (
            <span className="text-clean">Available</span>
          ) : (
            <span className="text-muted" title={`Download best.pt into checkpoints/${model.id}/`}>
              Results only
            </span>
          )}
        </td>
        <td className="px-4 py-3 text-right">
          <Button variant="ghost" onClick={onToggle} aria-expanded={open} className="px-2 py-1">
            {open ? "Hide details" : "Details"}
          </Button>
        </td>
      </tr>
      {open && (
        <tr className="border-b border-line bg-paper/60">
          <td colSpan={6} className="px-4 py-5">
            <ModelDetail id={model.id} />
          </td>
        </tr>
      )}
    </>
  );
}

export function ModelsData() {
  const datasets = useResource(api.datasets);
  const models = useResource(api.models);
  const cifar = datasets.data?.[0];

  return (
    <div className="space-y-8">
      <PageHeader title="Models and data">
        The dataset the models were trained on, the held-out images you can attack, and every trained checkpoint.
      </PageHeader>

      {datasets.error && <Notice tone="error">{datasets.error.message}</Notice>}
      {cifar && (
        <section aria-labelledby="dataset-heading" className="panel p-5">
          <h2 id="dataset-heading" className="text-lg font-semibold">
            {cifar.title}
          </h2>
          <p className="mt-1 text-sm text-muted">{cifar.description}</p>
          <dl className="mt-4 grid grid-cols-3 gap-4 text-sm sm:max-w-md">
            {Object.entries(cifar.splits).map(([split, count]) => (
              <div key={split}>
                <dt className="capitalize text-muted">{split}</dt>
                <dd className="text-lg font-semibold tabular-nums">{count.toLocaleString()}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-muted">
            The validation split is 100 training images per class, fixed by seed 42 and used only to pick the best
            checkpoint.
          </p>
        </section>
      )}
      {cifar && <SampleBrowser classes={cifar.classes} />}

      {models.loading && <Spinner label="Loading models" />}
      {models.error && <Notice tone="error">{models.error.message}</Notice>}
      {models.data &&
        (models.data.length ? (
          <ModelTable models={models.data} />
        ) : (
          <EmptyState title="No models found">Train one on the Training page, or add Colab checkpoints to checkpoints/.</EmptyState>
        ))}
    </div>
  );
}
