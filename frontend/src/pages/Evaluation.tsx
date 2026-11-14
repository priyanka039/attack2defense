import { useEffect, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type EvaluationJob, type EvaluationParams, type SweepRow } from "../api/types";
import { useToast } from "../components/Toast";
import { AccuracyVsEpsilon } from "../components/charts";
import {
  Badge,
  Button,
  buttonClass,
  Card,
  EmptyState,
  InfoTip,
  Notice,
  PageHeader,
  ProgressBar,
  Segmented,
  Slider,
  Spinner,
  StatusBadge,
} from "../components/ui";
import { dateTime, epsilonLabel, percent } from "../lib/format";
import { downloadBlob, usePolling, useResource } from "../lib/hooks";
import { architectureName } from "../lib/results";

const EPSILON_CHOICES = [1, 2, 4, 8, 16];

type PresetId = "quick" | "thorough" | "custom";
const PRESETS: Record<Exclude<PresetId, "custom">, { levels: number[]; perClass: number; steps: number }> = {
  quick: { levels: [2, 4, 8], perClass: 10, steps: 10 },
  thorough: { levels: [1, 2, 4, 8, 16], perClass: 50, steps: 20 },
};

function EvaluationForm({ onStarted }: { onStarted: (job: EvaluationJob) => void }) {
  const { notify } = useToast();
  const models = useResource(api.models);
  const usable = useMemo(() => (models.data ?? []).filter((model) => model.has_weights), [models.data]);
  const [modelIds, setModelIds] = useState<string[]>([]);
  const [preset, setPreset] = useState<PresetId>("quick");
  const [levels, setLevels] = useState<number[]>(PRESETS.quick.levels);
  const [threshold, setThreshold] = useState(0.8);
  const [perClass, setPerClass] = useState(PRESETS.quick.perClass);
  const [steps, setSteps] = useState(PRESETS.quick.steps);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!modelIds.length && usable.length) setModelIds(usable.slice(0, 2).map((model) => model.id));
  }, [usable, modelIds.length]);

  const toggle = <T,>(list: T[], value: T) => (list.includes(value) ? list.filter((item) => item !== value) : [...list, value]);

  function choosePreset(id: PresetId) {
    setPreset(id);
    if (id === "custom") return;
    setLevels(PRESETS[id].levels);
    setPerClass(PRESETS[id].perClass);
    setSteps(PRESETS[id].steps);
  }

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
    return (
      <EmptyState
        title="No model weights to evaluate"
        action={
          <>
            <Link to="/training" className={buttonClass("primary")}>
              Train a model
            </Link>
            <Link to="/guide#setup" className={buttonClass("secondary")}>
              Check setup status
            </Link>
          </>
        }
      >
        Evaluations attack every model live, so they need weights. The reported numbers are on the Dashboard.
      </EmptyState>
    );
  }

  return (
    <form onSubmit={submit} className="panel space-y-6 p-5 sm:p-6" aria-label="New evaluation">
      <fieldset>
        <legend className="field-label">Models to compare</legend>
        <p className="field-hint">Every model is attacked on exactly the same class-balanced test images.</p>
        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {usable.map((model) => {
            const checked = modelIds.includes(model.id);
            return (
              <label
                key={model.id}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-clean ${
                  checked ? "border-clean bg-clean-soft" : "border-line bg-paper hover:border-line-strong"
                }`}
              >
                <input type="checkbox" className="mt-0.5" checked={checked} onChange={() => setModelIds(toggle(modelIds, model.id))} />
                <span className="min-w-0">
                  <span className="block font-medium">{model.label}</span>
                  <span className="block text-xs text-muted">
                    {architectureName(model.architecture)}
                    {model.owner === "you" ? ", your run" : ""}
                    {model.summary ? `, reported PGD ${percent(model.summary.pgd_accuracy, 0)}` : ""}
                  </span>
                </span>
              </label>
            );
          })}
        </div>
      </fieldset>

      <Segmented
        legend="How thorough"
        name="eval-preset"
        value={preset}
        onChange={choosePreset}
        options={[
          { value: "quick", label: "Quick check", hint: "100 images, 3 budgets" },
          { value: "thorough", label: "Thorough", hint: "500 images, 5 budgets, PGD-20" },
          { value: "custom", label: "Custom", hint: "Choose below" },
        ]}
      />

      <fieldset>
        <legend className="flex items-center gap-1.5">
          <span className="field-label">Attack budgets to test</span>
          <InfoTip term="epsilon" />
        </legend>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EPSILON_CHOICES.map((level) => (
            <label
              key={level}
              className={`cursor-pointer rounded-lg border px-3 py-1.5 font-mono text-sm tabular-nums has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-clean ${
                levels.includes(level) ? "border-adv bg-adv-soft text-ink" : "border-line bg-paper text-muted hover:text-ink"
              }`}
            >
              <input
                type="checkbox"
                className="sr-only"
                checked={levels.includes(level)}
                onChange={() => {
                  setLevels(toggle(levels, level).sort((a, b) => a - b));
                  setPreset("custom");
                }}
              />
              {level}/255
            </label>
          ))}
        </div>
        <p className={`mt-1 text-xs ${problem?.field?.startsWith("epsilons") ? "text-flag" : "text-muted"}`}>
          {problem?.field?.startsWith("epsilons") ? problem.message : "Three or more budgets show how robustness fades as attacks grow."}
        </p>
      </fieldset>

      <div className="grid gap-5 sm:grid-cols-3">
        <Slider
          id="threshold"
          label="Pass mark"
          term="threshold"
          min={0}
          max={1}
          step={0.05}
          value={threshold}
          onChange={setThreshold}
          display={percent(threshold, 0)}
          hint="Robust accuracy below this is flagged."
        />
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
            onChange={(event) => {
              setPerClass(Number(event.target.value));
              setPreset("custom");
            }}
            className="input mt-1 font-mono tabular-nums"
          />
          <p className={`mt-1 text-xs ${problem?.field === "images_per_class" ? "text-flag" : "text-muted"}`}>
            {problem?.field === "images_per_class" ? problem.message : `${(perClass * 10).toLocaleString()} images in total.`}
          </p>
        </div>
        <div>
          <span className="flex items-center gap-1.5">
            <label htmlFor="eval-steps" className="field-label">
              PGD steps
            </label>
            <InfoTip term="pgd" />
          </span>
          <input
            id="eval-steps"
            type="number"
            min={1}
            max={50}
            value={steps}
            onChange={(event) => {
              setSteps(Number(event.target.value));
              setPreset("custom");
            }}
            className="input mt-1 font-mono tabular-nums"
          />
          <p className="field-hint mt-1">10 matches the reported results.</p>
        </div>
      </div>

      {problem && !problem.field?.startsWith("epsilons") && problem.field !== "images_per_class" && (
        <Notice tone="error">{problem.message}</Notice>
      )}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className="text-sm text-muted">
          {modelIds.length} model{modelIds.length === 1 ? "" : "s"} x {levels.length} budget{levels.length === 1 ? "" : "s"} x{" "}
          {(perClass * 10).toLocaleString()} images.
        </p>
        <Button type="submit" busy={busy} disabled={!modelIds.length || !levels.length}>
          Run evaluation
        </Button>
      </div>
    </form>
  );
}

function Cell({ row, attack }: { row: SweepRow; attack: "clean" | "fgsm" | "pgd" }) {
  const value = row[`${attack}_accuracy`];
  const flagged = (attack === "fgsm" && row.fgsm_below_threshold) || (attack === "pgd" && row.pgd_below_threshold) || false;
  const passed = attack !== "clean" && !flagged;
  return (
    <td className="px-4 py-2.5">
      <span className="flex items-center gap-2">
        <span className={`font-mono tabular-nums ${flagged ? "text-flag" : attack === "clean" ? "text-ink" : "text-clean"}`}>{percent(value)}</span>
        {flagged && <Badge tone="error">below</Badge>}
        {passed && <span className="sr-only">passes</span>}
      </span>
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
          {result.images.toLocaleString()} test images, PGD with {result.pgd_steps} steps, pass mark {percent(result.threshold, 0)}.
        </p>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => exportAs("csv")}>
            Download CSV
          </Button>
          <Button variant="secondary" size="sm" onClick={() => exportAs("json")}>
            Download JSON
          </Button>
        </div>
      </div>

      {result.flags.length > 0 ? (
        <Notice tone="warn" title={`${result.flags.length} of ${attackedCount} attack results are below ${percent(result.threshold, 0)}`}>
          They are marked "below" in the table. This is expected for undefended models and for budgets bigger than the one a model was trained on.
        </Notice>
      ) : (
        <Notice tone="success">Every model stayed at or above {percent(result.threshold, 0)} at every budget.</Notice>
      )}

      <div className="overflow-x-auto rounded-lg border border-line">
        <table className="w-full min-w-[38rem] text-sm">
          <thead className="border-b border-line bg-paper text-left text-muted">
            <tr>
              <th scope="col" className="px-4 py-2.5 font-medium">Model</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Budget</th>
              <th scope="col" className="px-4 py-2.5 font-medium">Clean</th>
              <th scope="col" className="px-4 py-2.5 font-medium">After FGSM</th>
              <th scope="col" className="px-4 py-2.5 font-medium">After PGD</th>
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
                  <td className="px-4 py-2.5 font-mono tabular-nums text-muted">{epsilonLabel(row.epsilon)}</td>
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
        A fair comparison: every selected model faces the same images and the same attacks at each budget. You get
        clean accuracy and accuracy after FGSM and PGD, with anything under your pass mark flagged.
      </PageHeader>

      <EvaluationForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />

      <section aria-labelledby="reports-heading">
        <h2 id="reports-heading" className="text-xl font-semibold">
          Your reports
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading reports" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No evaluations yet">Pick models above and press Run evaluation. A quick check takes about a minute on a CPU.</EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[17rem_1fr]">
            <ul className="space-y-1.5" aria-label="Evaluations">
              {list.map((job) => (
                <li key={job.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(job.id)}
                    aria-current={job.id === current.id}
                    className={`w-full rounded-lg border px-3 py-2.5 text-left text-sm transition-colors ${
                      job.id === current.id ? "border-line-strong bg-raised" : "border-line bg-surface hover:border-line-strong"
                    }`}
                  >
                    <span className="flex items-center justify-between gap-2">
                      <span className="font-medium">
                        {job.params.model_ids.length} model{job.params.model_ids.length > 1 ? "s" : ""}, {job.params.epsilons.length} budgets
                      </span>
                      <StatusBadge status={job.status} />
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">{dateTime(job.created_at)}</span>
                  </button>
                </li>
              ))}
            </ul>
            <Card>
              {isActive(current.status) && (
                <div className="space-y-3">
                  <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                    <span className="text-muted">{current.message}</span>
                    <Button variant="danger" size="sm" onClick={() => cancel(current.id)} disabled={current.status === "cancelling"}>
                      Stop evaluation
                    </Button>
                  </div>
                  <ProgressBar value={current.progress} label="Evaluation progress" />
                </div>
              )}
              {current.error && <Notice tone="error" title="Evaluation failed">{current.error}</Notice>}
              {current.status === "cancelled" && <Notice>This evaluation was stopped before it finished.</Notice>}
              {current.status === "succeeded" && current.result && <Report job={current} />}
            </Card>
          </div>
        )}
      </section>
    </div>
  );
}
