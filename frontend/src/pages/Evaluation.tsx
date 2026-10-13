import { useEffect, useMemo, useState, type FormEvent } from "react";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type EvaluationJob, type EvaluationParams, type SweepRow } from "../api/types";
import { useToast } from "../components/Toast";
import { AccuracyVsEpsilon } from "../components/charts";
import { Button, EmptyState, Notice, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { dateTime, epsilonLabel, percent } from "../lib/format";
import { downloadBlob, usePolling, useResource } from "../lib/hooks";

const EPSILON_CHOICES = [1, 2, 4, 8, 16];

function EvaluationForm({ onStarted }: { onStarted: (job: EvaluationJob) => void }) {
  const { notify } = useToast();
  const models = useResource(api.models);
  const usable = useMemo(() => (models.data ?? []).filter((model) => model.has_weights), [models.data]);
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [levels, setLevels] = useState<number[]>([2, 4, 8]);
  const [threshold, setThreshold] = useState(0.8);
  const [perClass, setPerClass] = useState(20);
  const [steps, setSteps] = useState(10);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!modelIds.length && usable.length) setModelIds(usable.slice(0, 2).map((model) => model.id));
  }, [usable, modelIds.length]);

  const toggle = <T,>(list: T[], value: T) => (list.includes(value) ? list.filter((item) => item !== value) : [...list, value]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    const params: EvaluationParams = {
      model_ids: modelIds,
      epsilons: levels.map((level) => level / 255),
      threshold,
      images_per_class: perClass,
      pgd_steps: steps,
    };
    try {
      onStarted(await api.startEvaluation(params));
      notify("Evaluation started.", "success");
    } catch (caught) {
      setProblem(toProblem(caught));
    } finally {
      setBusy(false);
    }
  }

  if (models.loading) return <Spinner label="Loading models" />;
  if (models.error) return <Notice tone="error">{models.error.message}</Notice>;
  if (!usable.length) {
    return <EmptyState title="No model weights to evaluate">Add Colab checkpoints to checkpoints/ or train a model first.</EmptyState>;
  }

  return (
    <form onSubmit={submit} className="panel space-y-5 p-5" aria-label="New evaluation">
      <fieldset>
        <legend className="field-label">Models to compare</legend>
        <p className="field-hint">Every model is attacked on the same class-balanced test images.</p>
        <div className="mt-2 grid gap-2 sm:grid-cols-2">
          {usable.map((model) => (
            <label key={model.id} className="flex items-start gap-2 rounded-md border border-line p-2.5 text-sm">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={modelIds.includes(model.id)}
                onChange={() => setModelIds(toggle(modelIds, model.id))}
              />
              <span>
                <span className="font-medium">{model.label}</span>
                <span className="block text-xs text-muted">
                  {model.architecture}
                  {model.owner === "you" ? ", your run" : ""} <span className="font-mono">{model.tag}</span>
                </span>
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend className="field-label">Attack budgets, out of 255</legend>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EPSILON_CHOICES.map((level) => (
            <label
              key={level}
              className={`cursor-pointer rounded-md border px-3 py-1.5 text-sm tabular-nums has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-clean ${
                levels.includes(level) ? "border-ink bg-ink text-white" : "border-line bg-surface"
              }`}
            >
              <input type="checkbox" className="sr-only" checked={levels.includes(level)} onChange={() => setLevels(toggle(levels, level).sort((a, b) => a - b))} />
              {level}/255
            </label>
          ))}
        </div>
        <p className={`mt-1 text-xs ${problem?.field?.startsWith("epsilons") ? "text-flag" : "text-muted"}`}>
          {problem?.field?.startsWith("epsilons") ? problem.message : "A standard sweep uses three or more budgets."}
        </p>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-3">
        <div>
          <div className="flex justify-between">
            <label htmlFor="threshold" className="field-label">
              Minimum acceptable accuracy
            </label>
            <span className="text-sm tabular-nums">{percent(threshold, 0)}</span>
          </div>
          <input
            id="threshold"
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
            className="mt-2 w-full"
          />
          <p className="field-hint">Results under this are flagged.</p>
        </div>
        <div>
          <label htmlFor="per-class" className="field-label">
            Test images per class
          </label>
          <input
            id="per-class"
            type="number"
            min={1}
            max={200}
            value={perClass}
            onChange={(event) => setPerClass(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
          <p className={`mt-1 text-xs ${problem?.field === "images_per_class" ? "text-flag" : "text-muted"}`}>
            {problem?.field === "images_per_class" ? problem.message : `${(perClass * 10).toLocaleString()} images in total.`}
          </p>
        </div>
        <div>
          <label htmlFor="eval-steps" className="field-label">
            PGD steps
          </label>
          <input
            id="eval-steps"
            type="number"
            min={1}
            max={50}
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
          <p className="field-hint">10 matches the reported results.</p>
        </div>
      </div>

      {problem && !problem.field?.startsWith("epsilons") && problem.field !== "images_per_class" && (
        <Notice tone="error">{problem.message}</Notice>
      )}
      <div className="flex justify-end border-t border-line pt-4">
        <Button type="submit" busy={busy} disabled={!modelIds.length || !levels.length}>
          Run evaluation
        </Button>
      </div>
    </form>
  );
}

function Cell({ row, attack }: { row: SweepRow; attack: "clean" | "fgsm" | "pgd" }) {
  const value = row[`${attack}_accuracy`];
  const flagged =
    (attack === "fgsm" && row.fgsm_below_threshold) || (attack === "pgd" && row.pgd_below_threshold) || false;
  return (
    <td className={`px-4 py-2.5 tabular-nums ${flagged ? "bg-flag-soft text-flag" : ""}`}>
      {percent(value)}
      {flagged && <span className="ml-1.5 text-xs font-medium">below</span>}
    </td>
  );
}

function Report({ job }: { job: EvaluationJob }) {
  const { notify } = useToast();
  const result = job.result!;
  const attackedCount = result.models.reduce((total, model) => total + model.rows.length * 2, 0);

  async function exportAs(format: "csv" | "json") {
    try {
      downloadBlob(await api.exportEvaluation(job.id, format), `evaluation_${job.id.slice(0, 8)}.${format}`);
    } catch (caught) {
      notify(toProblem(caught).message, "error");
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="text-sm text-muted">
          {result.images.toLocaleString()} test images, PGD with {result.pgd_steps} steps, threshold{" "}
          {percent(result.threshold, 0)}.
        </p>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => exportAs("csv")}>
            Export CSV
          </Button>
          <Button variant="secondary" onClick={() => exportAs("json")}>
            Export JSON
          </Button>
        </div>
      </div>

      {result.flags.length > 0 ? (
        <Notice tone="error">
          {result.flags.length} of {attackedCount} attack results fell below {percent(result.threshold, 0)}. They are
          marked "below" in the table.
        </Notice>
      ) : (
        <Notice>Every model stayed at or above {percent(result.threshold, 0)} at every budget.</Notice>
      )}

      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[36rem] text-sm">
          <thead className="border-b border-line bg-paper text-left text-muted">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Model</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Epsilon</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Clean</th>
              <th scope="col" className="px-4 py-2.5 font-medium">FGSM</th>
              <th scope="col" className="px-4 py-2.5 font-medium">PGD</th>
            </tr>
          </thead>
          <tbody>
            {result.models.flatMap((model) =>
              model.rows.map((row, index) => (
                <tr key={`${model.model_id}-${row.epsilon}`} className="border-b border-line last:border-0">
                  {index === 0 && (
                    <th scope="rowgroup" rowSpan={model.rows.length} className="border-r border-line px-4 py-2.5 text-left align-top font-medium">
                      {model.label}
                      <span className="block font-mono text-xs font-normal text-muted">{model.model_id}</span>
                    </th>
                  )}
                  <td className="px-4 py-2.5 tabular-nums">{epsilonLabel(row.epsilon)}</td>
                  <Cell row={row} attack="clean" />
                  <Cell row={row} attack="fgsm" />
                  <Cell row={row} attack="pgd" />
                </tr>
              )),
            )}
          </tbody>
        </table>
      </div>

      <AccuracyVsEpsilon
        series={result.models.map((model) => ({ id: model.model_id, label: model.label, rows: model.rows }))}
        threshold={result.threshold}
      />
    </div>
  );
}

export function Evaluation() {
  const { notify } = useToast();
  const jobs = useResource(api.evaluations);
  const [selected, setSelected] = useState<string | null>(null);
  const list = jobs.data ?? [];
  const current = list.find((job) => job.id === selected) ?? list[0];

  usePolling(() => void jobs.reload(), list.some((job) => isActive(job.status)));

  async function cancel(id: string) {
    try {
      await api.cancelEvaluation(id);
      await jobs.reload();
    } catch (caught) {
      notify(toProblem(caught).message, "error");
    }
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Evaluation">
        Compare models on the same held-out images: clean accuracy, and accuracy after FGSM and PGD at each attack
        budget. Anything under your threshold is flagged.
      </PageHeader>

      <EvaluationForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />

      <section aria-labelledby="reports-heading">
        <h2 id="reports-heading" className="text-lg font-semibold">
          Reports
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading reports" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No evaluations yet">Pick models above and run one. 20 images per class takes about a minute on a CPU.</EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[16rem_1fr]">
            <ul className="space-y-1" aria-label="Evaluations">
              {list.map((job) => (
                <li key={job.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(job.id)}
                    aria-current={job.id === current.id}
                    className={`w-full rounded-md border px-3 py-2 text-left text-sm ${
                      job.id === current.id ? "border-ink bg-surface" : "border-transparent hover:bg-surface"
                    }`}
                  >
                    <span className="flex items-center justify-between gap-2">
                      <span className="font-medium">
                        {job.params.model_ids.length} model{job.params.model_ids.length > 1 ? "s" : ""}
                      </span>
                      <StatusBadge status={job.status} />
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">{dateTime(job.created_at)}</span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="panel p-5">
              {isActive(current.status) && (
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span>{current.message}</span>
                    <Button variant="danger" onClick={() => cancel(current.id)} disabled={current.status === "cancelling"}>
                      Stop evaluation
                    </Button>
                  </div>
                  <ProgressBar value={current.progress} label="Evaluation progress" />
                </div>
              )}
              {current.error && <Notice tone="error">{current.error}</Notice>}
              {current.status === "cancelled" && <Notice>This evaluation was stopped before it finished.</Notice>}
              {current.status === "succeeded" && current.result && <Report job={current} />}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
