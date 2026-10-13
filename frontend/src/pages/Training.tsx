import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type TrainingJob, type TrainingParams } from "../api/types";
import { useToast } from "../components/Toast";
import { TrainingCurves } from "../components/charts";
import { Button, EmptyState, Notice, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { dateTime, epsilonLabel, percent } from "../lib/format";
import { usePolling, useResource } from "../lib/hooks";

const DEFENSES: { mode: TrainingParams["mode"]; title: string; text: string }[] = [
  { mode: "clean", title: "No defense", text: "Ordinary training on clean images. The baseline to beat." },
  { mode: "fgsm", title: "FGSM adversarial training", text: "One random-start FGSM step per batch. About 2x the cost." },
  { mode: "pgd", title: "PGD adversarial training", text: "Several PGD steps per batch. Slower, and the most robust." },
  { mode: "gaussian", title: "Gaussian noise training", text: "Train with N(0, sigma) noise for the certified-defense track." },
];

const DEFAULTS: TrainingParams = {
  model: "small_cnn",
  mode: "pgd",
  epsilon: 2 / 255,
  pgd_steps: 3,
  adv_ratio: 1,
  sigma: 0.25,
  epochs: 2,
  batch_size: 128,
  lr: 0.001,
  optimizer: "adamw",
  max_train_batches: 40,
  val_per_class: 20,
};

function NumberField({
  id,
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step = 1,
  error,
}: {
  id: string;
  label: string;
  hint?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  error?: string | null;
}) {
  return (
    <div>
      <label htmlFor={id} className="field-label">
        {label}
      </label>
      <input
        id={id}
        type="number"
        className="input mt-1 tabular-nums"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-invalid={Boolean(error)}
        aria-describedby={`${id}-hint`}
      />
      <p id={`${id}-hint`} className={`mt-1 text-xs ${error ? "text-flag" : "text-muted"}`}>
        {error ?? hint}
      </p>
    </div>
  );
}

function TrainingForm({ onStarted }: { onStarted: (job: TrainingJob) => void }) {
  const { notify } = useToast();
  const [params, setParams] = useState<TrainingParams>(DEFAULTS);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (patch: Partial<TrainingParams>) => setParams((current) => ({ ...current, ...patch }));
  const fieldError = (field: string) => (problem?.field === field ? problem.message : null);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      const job = await api.startTraining(params);
      notify("Training started. Progress updates below.", "success");
      onStarted(job);
    } catch (caught) {
      setProblem(toProblem(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="panel space-y-5 p-5" aria-label="New training run">
      <fieldset>
        <legend className="field-label">Defense</legend>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          {DEFENSES.map((defense) => (
            <label
              key={defense.mode}
              className={`cursor-pointer rounded-md border p-3 has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-clean ${
                params.mode === defense.mode ? "border-ink bg-ink/[0.03]" : "border-line hover:border-ink/40"
              }`}
            >
              <input
                type="radio"
                name="mode"
                className="sr-only"
                checked={params.mode === defense.mode}
                onChange={() => set({ mode: defense.mode })}
              />
              <span className="block text-sm font-semibold">{defense.title}</span>
              <span className="mt-1 block text-xs text-muted">{defense.text}</span>
            </label>
          ))}
        </div>
      </fieldset>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label htmlFor="arch" className="field-label">
            Architecture
          </label>
          <select
            id="arch"
            className="input mt-1"
            value={params.model}
            onChange={(event) => set({ model: event.target.value as TrainingParams["model"] })}
          >
            <option value="small_cnn">Small CNN (fast on a CPU)</option>
            <option value="resnet18">ResNet-18 (wants a GPU)</option>
          </select>
        </div>
        <NumberField
          id="epsilon-levels"
          label="Epsilon, out of 255"
          value={Math.round(params.epsilon * 255 * 100) / 100}
          min={0}
          max={32}
          step={0.5}
          onChange={(value) => set({ epsilon: value / 255 })}
          hint={`Budget ${epsilonLabel(params.epsilon)} for the training attack.`}
          error={fieldError("epsilon")}
        />
        <NumberField
          id="epochs"
          label="Epochs"
          value={params.epochs}
          min={1}
          max={30}
          onChange={(epochs) => set({ epochs })}
          hint="Full passes over the training subset."
          error={fieldError("epochs")}
        />
        <NumberField
          id="batches"
          label="Batches per epoch"
          value={params.max_train_batches}
          min={1}
          max={400}
          onChange={(max_train_batches) => set({ max_train_batches })}
          hint={`${(params.max_train_batches * params.batch_size).toLocaleString()} images per epoch.`}
          error={fieldError("max_train_batches")}
        />
      </div>

      {params.mode !== "clean" && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <div className="flex justify-between">
              <label htmlFor="ratio" className="field-label">
                Adversarial share of each batch
              </label>
              <span className="text-sm tabular-nums">{percent(params.adv_ratio, 0)}</span>
            </div>
            <input
              id="ratio"
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={params.adv_ratio}
              onChange={(event) => set({ adv_ratio: Number(event.target.value) })}
              className="mt-2 w-full"
              aria-describedby="ratio-hint"
            />
            <p id="ratio-hint" className="field-hint">
              The rest of each batch stays clean. Lower shares trade robustness for clean accuracy.
            </p>
          </div>
          {params.mode === "pgd" && (
            <NumberField
              id="pgd-steps"
              label="PGD steps per batch"
              value={params.pgd_steps}
              min={1}
              max={20}
              onChange={(pgd_steps) => set({ pgd_steps })}
              hint="More steps make a stronger training attack and a slower epoch."
              error={fieldError("pgd_steps")}
            />
          )}
        </div>
      )}

      <details className="text-sm">
        <summary className="cursor-pointer font-medium">Optimizer settings</summary>
        <div className="mt-3 grid gap-4 sm:grid-cols-3">
          <div>
            <label htmlFor="optimizer" className="field-label">
              Optimizer
            </label>
            <select
              id="optimizer"
              className="input mt-1"
              value={params.optimizer}
              onChange={(event) => set({ optimizer: event.target.value as TrainingParams["optimizer"] })}
            >
              <option value="adamw">AdamW</option>
              <option value="sgd">SGD with momentum</option>
            </select>
          </div>
          <NumberField id="lr" label="Learning rate" value={params.lr} step={0.0005} min={0} onChange={(lr) => set({ lr })} error={fieldError("lr")} />
          <NumberField
            id="batch-size"
            label="Batch size"
            value={params.batch_size}
            min={8}
            max={512}
            onChange={(batch_size) => set({ batch_size })}
            error={fieldError("batch_size")}
          />
        </div>
      </details>

      {problem && !problem.field && <Notice tone="error">{problem.message}</Notice>}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className="text-xs text-muted">
          Runs here are short demos on the server. The reported ResNet-18 models were trained for 30 epochs on a GPU.
        </p>
        <Button type="submit" busy={busy}>
          Start training
        </Button>
      </div>
    </form>
  );
}

function JobDetail({ job, onCancel }: { job: TrainingJob; onCancel: () => void }) {
  const summary = job.result?.summary;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold">
            {DEFENSES.find((defense) => defense.mode === job.params.mode)?.title} on {job.params.model}
          </h3>
          <p className="text-sm text-muted">
            Started {dateTime(job.created_at)}. {job.params.epochs} epochs, epsilon {epsilonLabel(job.params.epsilon)}
            {job.params.mode !== "clean" && `, ${percent(job.params.adv_ratio, 0)} adversarial`}.
          </p>
        </div>
        {isActive(job.status) && (
          <Button variant="danger" onClick={onCancel} disabled={job.status === "cancelling"}>
            Stop training
          </Button>
        )}
      </div>

      <div>
        <div className="mb-1 flex justify-between text-sm">
          <span>{job.message}</span>
          <span className="tabular-nums text-muted">{percent(job.progress, 0)}</span>
        </div>
        <ProgressBar value={job.progress} label="Training progress" />
      </div>

      {job.error && <Notice tone="error">{job.error}</Notice>}

      {job.history.length > 0 ? (
        <TrainingCurves history={job.history} height={220} />
      ) : (
        isActive(job.status) && <p className="text-sm text-muted">Curves appear after the first epoch.</p>
      )}

      {summary && job.result && (
        <div className="rounded-md border border-line p-4 text-sm">
          <p>
            Held-out test ({summary.images} images, PGD at {epsilonLabel(summary.epsilon)}): clean{" "}
            <strong>{percent(summary.clean_accuracy)}</strong>, FGSM <strong>{percent(summary.fgsm_accuracy)}</strong>, PGD{" "}
            <strong>{percent(summary.pgd_accuracy)}</strong>.
          </p>
          <p className="mt-2">
            <Link to="/attack" className="font-medium text-clean underline">
              Attack it in the lab
            </Link>{" "}
            or add it to an{" "}
            <Link to="/evaluation" className="font-medium text-clean underline">
              evaluation
            </Link>
            . It is listed as <span className="font-mono text-xs">{job.result.model_id}</span>.
          </p>
        </div>
      )}
    </div>
  );
}

export function Training() {
  const { notify } = useToast();
  const jobs = useResource(api.trainingJobs);
  const [selected, setSelected] = useState<string | null>(null);
  const list = jobs.data ?? [];
  const current = list.find((job) => job.id === selected) ?? list[0];

  useEffect(() => {
    if (!selected && list.length) setSelected(list[0].id);
  }, [list, selected]);

  usePolling(() => void jobs.reload(), list.some((job) => isActive(job.status)));

  async function cancel(id: string) {
    try {
      await api.cancelTraining(id);
      notify("Stopping after the current batch.");
      await jobs.reload();
    } catch (caught) {
      notify(toProblem(caught).message, "error");
    }
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Training">
        Train a baseline or a defended model. Adversarial training replaces part of every batch with FGSM or PGD
        examples made on the fly against the current weights.
      </PageHeader>

      <TrainingForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />

      <section aria-labelledby="runs-heading">
        <h2 id="runs-heading" className="text-lg font-semibold">
          Your runs
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading runs" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No training runs yet">Start one with the form above. The defaults finish in about a minute.</EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[16rem_1fr]">
            <ul className="space-y-1" aria-label="Training runs">
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
                      <span className="font-medium">{job.params.mode === "clean" ? "No defense" : `${job.params.mode.toUpperCase()} training`}</span>
                      <StatusBadge status={job.status} />
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">{dateTime(job.created_at)}</span>
                  </button>
                </li>
              ))}
            </ul>
            <div className="panel p-5">
              <JobDetail job={current} onCancel={() => cancel(current.id)} />
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
