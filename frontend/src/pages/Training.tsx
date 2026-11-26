import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type TrainingJob, type TrainingParams } from "../api/types";
import { useToast } from "../components/Toast";
import { TrainingCurves } from "../components/charts";
import {
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
import type { GlossaryKey } from "../lib/glossary";
import { usePolling, useResource } from "../lib/hooks";
import { architectureName } from "../lib/results";
import { useSetup } from "../lib/setup";

const DEFENSES: { mode: TrainingParams["mode"]; title: string; text: string }[] = [
  { mode: "clean", title: "No defense", text: "Ordinary training. The baseline that attacks break." },
  { mode: "fgsm", title: "FGSM training", text: "Train on one-step attacks. About 2x slower." },
  { mode: "pgd", title: "PGD training", text: "Train on multi-step attacks. Slowest, most robust." },
  { mode: "gaussian", title: "Noise training", text: "Train on random noise, for the certified defense." },
];

type PresetId = "quick" | "stronger" | "custom";

const BASE: TrainingParams = {
  model: "small_cnn",
  mode: "pgd",
  epsilon: 2 / 255,
  pgd_steps: 3,
  adv_ratio: 1,
  sigma: 0.25,
  epochs: 2,
  batch_size: 64,
  lr: 0.001,
  optimizer: "adamw",
  max_train_batches: 20,
  val_per_class: 10,
};

// Kept within Render Free server caps (see render.yaml A2D_MAX_*).
const PRESETS: Record<Exclude<PresetId, "custom">, Partial<TrainingParams>> = {
  quick: { model: "small_cnn", epochs: 2, max_train_batches: 20, pgd_steps: 3, batch_size: 64 },
  stronger: { model: "small_cnn", epochs: 3, max_train_batches: 40, pgd_steps: 3, batch_size: 64 },
};

/** Very rough images-per-second for one forward+backward pass, used only for the time hint. */
function estimateSeconds(params: TrainingParams, gpu: boolean): number {
  const throughput = (params.model === "resnet18" ? 60 : 700) * (gpu ? 12 : 1);
  const cost = params.mode === "pgd" ? params.pgd_steps + 1 : params.mode === "fgsm" ? 2 : 1;
  const adversarial = params.mode === "clean" || params.mode === "gaussian" ? 1 : 1 + (cost - 1) * params.adv_ratio;
  const images = params.epochs * params.max_train_batches * params.batch_size;
  return (images * adversarial) / throughput + params.epochs * 5;
}

function duration(secondsTotal: number): string {
  if (secondsTotal < 90) return "about 1 minute";
  if (secondsTotal < 3600) return `about ${Math.round(secondsTotal / 60)} minutes`;
  return `about ${(secondsTotal / 3600).toFixed(1)} hours`;
}

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
  term,
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
  term?: GlossaryKey;
}) {
  return (
    <div>
      <span className="flex items-center gap-1.5">
        <label htmlFor={id} className="field-label">
          {label}
        </label>
        {term && <InfoTip term={term} />}
      </span>
      <input
        id={id}
        type="number"
        className="input mt-1 font-mono tabular-nums"
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
  const setup = useSetup();
  const gpu = setup.data?.device.startsWith("cuda") ?? false;
  const [params, setParams] = useState<TrainingParams>(BASE);
  const [preset, setPreset] = useState<PresetId>("quick");
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);
  const set = (patch: Partial<TrainingParams>) => {
    setParams((current) => ({ ...current, ...patch }));
    setPreset("custom");
  };
  const fieldError = (field: string) => (problem?.field === field ? problem.message : null);

  function choosePreset(id: PresetId) {
    setPreset(id);
    if (id !== "custom") setParams((current) => ({ ...current, ...PRESETS[id] }));
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      const job = await api.startTraining(params);
      notify("Training started. Watch progress below.", "success");
      onStarted(job);
    } catch (caught) {
      setProblem(toProblem(caught));
    } finally {
      setBusy(false);
    }
  }

  const adversarial = params.mode === "fgsm" || params.mode === "pgd";

  return (
    <form onSubmit={submit} className="panel space-y-6 p-5 sm:p-6" aria-label="New training run">
      <Segmented
        legend="Defense"
        name="mode"
        columns={2}
        value={params.mode}
        onChange={(mode) => set({ mode })}
        options={DEFENSES.map((defense) => ({ value: defense.mode, label: defense.title, hint: defense.text }))}
      />

      <Segmented
        legend="How long"
        name="preset"
        value={preset}
        onChange={choosePreset}
        options={[
          { value: "quick", label: "Quick demo", hint: "Small CNN, 2 short epochs" },
          { value: "stronger", label: "Stronger demo", hint: "Small CNN, 3 fuller epochs" },
          { value: "custom", label: "Custom", hint: "Set everything below" },
        ]}
      />

      {adversarial && (
        <div className="grid gap-5 sm:grid-cols-2">
          <Slider
            id="epsilon-levels"
            label="Training attack budget"
            term="epsilon"
            min={0.5}
            max={16}
            step={0.5}
            value={Math.round(params.epsilon * 255 * 100) / 100}
            onChange={(value) => set({ epsilon: value / 255 })}
            display={epsilonLabel(params.epsilon)}
            hint={fieldError("epsilon") ?? "The model learns to resist attacks up to this size. Reported runs use 2/255."}
          />
          <Slider
            id="ratio"
            label="Adversarial share of each batch"
            term="advRatio"
            min={0}
            max={1}
            step={0.05}
            value={params.adv_ratio}
            onChange={(adv_ratio) => set({ adv_ratio })}
            display={percent(params.adv_ratio, 0)}
            hint="Lower shares keep more clean accuracy but give less robustness."
          />
        </div>
      )}

      <details className="group rounded-lg border border-line bg-paper px-4 py-3 text-sm" open={preset === "custom"}>
        <summary className="cursor-pointer font-medium text-muted hover:text-ink">
          <span className="mr-1 inline-block transition-transform group-open:rotate-90" aria-hidden="true">
            &#9656;
          </span>
          Advanced settings
        </summary>
        <div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
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
              <option value="resnet18">ResNet-18 (needs a GPU)</option>
            </select>
            <p className="mt-1 text-xs text-muted">ResNet-18 is what the reported results use.</p>
          </div>
          <NumberField id="epochs" label="Epochs" value={params.epochs} min={1} max={30} onChange={(epochs) => set({ epochs })} hint="Passes over the training subset." error={fieldError("epochs")} />
          <NumberField
            id="batches"
            label="Batches per epoch"
            value={params.max_train_batches}
            min={1}
            max={400}
            onChange={(max_train_batches) => set({ max_train_batches })}
            hint={`${(params.max_train_batches * params.batch_size).toLocaleString()} of 49,000 images per epoch.`}
            error={fieldError("max_train_batches")}
          />
          {params.mode === "pgd" && (
            <NumberField
              id="pgd-steps"
              label="PGD steps per batch"
              term="pgd"
              value={params.pgd_steps}
              min={1}
              max={20}
              onChange={(pgd_steps) => set({ pgd_steps })}
              hint="More steps: stronger training attack, slower epoch."
              error={fieldError("pgd_steps")}
            />
          )}
          {params.mode === "gaussian" && (
            <NumberField
              id="sigma"
              label="Noise sigma"
              term="sigma"
              value={params.sigma ?? 0.25}
              min={0.05}
              max={1}
              step={0.05}
              onChange={(sigma) => set({ sigma })}
              hint="Use the same sigma when certifying."
              error={fieldError("sigma")}
            />
          )}
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
          <NumberField id="lr" label="Learning rate" value={params.lr} step={0.0005} min={0} onChange={(lr) => set({ lr })} hint="0.001 for AdamW, 0.1 for SGD." error={fieldError("lr")} />
          <NumberField id="batch-size" label="Batch size" value={params.batch_size} min={8} max={512} onChange={(batch_size) => set({ batch_size })} error={fieldError("batch_size")} />
        </div>
      </details>

      {problem && !problem.field && <Notice tone="error">{problem.message}</Notice>}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className="text-sm text-muted">
          Estimated time on this server ({gpu ? "GPU" : "CPU"}):{" "}
          <span className="font-medium text-ink">{duration(estimateSeconds(params, gpu))}</span>. Demo runs are short; the
          reported models trained for 30 full epochs on a GPU.
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
            {DEFENSES.find((defense) => defense.mode === job.params.mode)?.title} on {architectureName(job.params.model)}
          </h3>
          <p className="text-sm text-muted">
            Started {dateTime(job.created_at)}. {job.params.epochs} epochs
            {job.params.mode === "fgsm" || job.params.mode === "pgd"
              ? `, budget ${epsilonLabel(job.params.epsilon)}, ${percent(job.params.adv_ratio, 0)} of each batch attacked`
              : ""}
            .
          </p>
        </div>
        {isActive(job.status) && (
          <Button variant="danger" onClick={onCancel} disabled={job.status === "cancelling"}>
            Stop training
          </Button>
        )}
      </div>

      <div>
        <div className="mb-1.5 flex justify-between text-sm">
          <span className="text-muted">{job.message}</span>
          <span className="font-mono tabular-nums">{percent(job.progress, 0)}</span>
        </div>
        <ProgressBar value={job.progress} label="Training progress" />
      </div>

      {job.error && <Notice tone="error" title="Training failed">{job.error}</Notice>}

      {job.history.length > 0 ? (
        <TrainingCurves history={job.history} height={220} />
      ) : (
        isActive(job.status) && <p className="text-sm text-muted">Charts appear after the first epoch finishes.</p>
      )}

      {summary && job.result && (
        <Notice tone="success" title="Finished. Tested on held-out images">
          <p>
            {summary.images} images at {epsilonLabel(summary.epsilon)}: clean <strong>{percent(summary.clean_accuracy)}</strong>, FGSM{" "}
            <strong>{percent(summary.fgsm_accuracy)}</strong>, PGD <strong>{percent(summary.pgd_accuracy)}</strong>.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Link to="/attack" className={buttonClass("attack", "sm")}>
              Attack it in the lab
            </Link>
            <Link to="/evaluation" className={buttonClass("secondary", "sm")}>
              Compare it with other models
            </Link>
          </div>
        </Notice>
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
        Train a model and choose its defense. <span className="text-ink">Adversarial training</span> attacks each
        batch on the fly and teaches the model to classify the attacked images correctly. Finished models appear in
        the Attack lab automatically.
      </PageHeader>

      <TrainingForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />

      <section aria-labelledby="runs-heading">
        <h2 id="runs-heading" className="text-xl font-semibold">
          Your runs
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading runs" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No training runs yet">Pick a defense above and press Start training. The quick demo takes about a minute.</EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[17rem_1fr]">
            <ul className="space-y-1.5" aria-label="Training runs">
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
                      <span className="font-medium">{DEFENSES.find((defense) => defense.mode === job.params.mode)?.title}</span>
                      <StatusBadge status={job.status} />
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">
                      {architectureName(job.params.model)}, {dateTime(job.created_at)}
                    </span>
                    {isActive(job.status) && (
                      <span className="mt-2 block">
                        <ProgressBar value={job.progress} label={`Progress of run ${job.id.slice(0, 6)}`} />
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
            <Card>
              <JobDetail job={current} onCancel={() => cancel(current.id)} />
            </Card>
          </div>
        )}
      </section>
    </div>
  );
}
