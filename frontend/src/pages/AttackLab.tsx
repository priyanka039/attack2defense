import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { useSearchParams } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import type { AttackKind, AttackRequest, AttackResponse, Prediction } from "../api/types";
import { SuccessPerStep } from "../components/charts";
import { Button, EmptyState, Notice, PageHeader, Spinner, Stat } from "../components/ui";
import { epsilonLabel, percent, seconds, sentenceCase } from "../lib/format";
import { useDebounced, useResource } from "../lib/hooks";

const CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
const EPSILON_TICKS = [0, 2, 4, 8, 16];
const BATCH = 16;

type Source = { type: "sample"; index: number } | { type: "upload"; image: string; name: string; label: string | null };

interface Settings {
  modelId: string;
  kind: AttackKind;
  levels: number;
  steps: number;
  alphaAuto: boolean;
  alphaLevels: number;
  randomStart: boolean;
}

function toRequest(settings: Settings, source: Source): AttackRequest {
  const base = {
    model_id: settings.modelId,
    epsilon: settings.levels / 255,
    steps: settings.steps,
    alpha: settings.kind === "pgd" && !settings.alphaAuto ? settings.alphaLevels / 255 : null,
    random_start: settings.randomStart,
  };
  return source.type === "sample"
    ? { ...base, sample_indices: [source.index] }
    : { ...base, image: source.image, label: source.label };
}

async function randomIndex(label?: string): Promise<number> {
  const first = await api.samples("cifar10", { label, limit: 1 });
  const offset = Math.floor(Math.random() * first.total);
  const page = await api.samples("cifar10", { label, offset, limit: 1 });
  return page.items[0].index;
}

function Predictions({ title, items, tone }: { title: string; items: Prediction[]; tone: "clean" | "adv" }) {
  return (
    <div>
      <p className="text-xs text-muted">{title}</p>
      <ul className="mt-1 space-y-1">
        {items.map((item, position) => (
          <li key={item.label} className="text-sm">
            <div className="flex justify-between gap-2">
              <span className={`capitalize ${position === 0 ? "font-semibold" : ""}`}>{item.label}</span>
              <span className="tabular-nums text-muted">{percent(item.confidence)}</span>
            </div>
            <span className="mt-0.5 block h-1 overflow-hidden rounded-full bg-ink/10" aria-hidden="true">
              <span
                className={`block h-full ${tone === "clean" ? "bg-clean" : "bg-adv"}`}
                style={{ width: `${item.confidence * 100}%` }}
              />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function EpsilonControl({ levels, onChange }: { levels: number; onChange: (value: number) => void }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <label htmlFor="epsilon" className="field-label">
          Attack budget, epsilon
        </label>
        <span className="text-xs text-muted tabular-nums">{(levels / 255).toFixed(4)}</span>
      </div>
      <output htmlFor="epsilon" className="mt-1 block font-display text-4xl font-semibold tabular-nums">
        {epsilonLabel(levels / 255)}
      </output>
      <input
        id="epsilon"
        type="range"
        min={0}
        max={16}
        step={0.5}
        value={levels}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full"
        aria-describedby="epsilon-hint"
        list="epsilon-ticks"
      />
      <datalist id="epsilon-ticks">
        {EPSILON_TICKS.map((tick) => (
          <option key={tick} value={tick} />
        ))}
      </datalist>
      <div className="flex justify-between text-xs text-muted" aria-hidden="true">
        {EPSILON_TICKS.map((tick) => (
          <span key={tick}>{tick}</span>
        ))}
      </div>
      <p id="epsilon-hint" className="field-hint mt-1">
        Largest change allowed to any pixel, in intensity levels out of 255. The defenses were trained at 2/255.
      </p>
    </div>
  );
}

function BatchCheck({ settings }: { settings: Settings }) {
  const [result, setResult] = useState<AttackResponse | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  async function run() {
    setBusy(true);
    setProblem(null);
    try {
      const indices = new Set<number>();
      while (indices.size < BATCH) indices.add(Math.floor(Math.random() * 10000));
      const request = { ...toRequest(settings, { type: "sample", index: 0 }), sample_indices: [...indices] };
      setResult(await api.attack(settings.kind, request));
    } catch (caught) {
      setProblem(toProblem(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section aria-labelledby="batch-heading" className="panel p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 id="batch-heading" className="text-lg font-semibold">
            Batch check
          </h2>
          <p className="mt-1 max-w-xl text-sm text-muted">
            One image says little. Attack {BATCH} random test images with the same settings to see how often the
            attack wins and how quickly.
          </p>
        </div>
        <Button variant="secondary" busy={busy} onClick={run} disabled={!settings.modelId}>
          Attack {BATCH} random images
        </Button>
      </div>
      {problem && (
        <div className="mt-4">
          <Notice tone="error">{problem.message}</Notice>
        </div>
      )}
      {result && (
        <div className="mt-5 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div>
            <p className="text-sm">
              <span className="font-display text-3xl font-semibold text-adv tabular-nums">
                {percent(result.success_rate, 0)}
              </span>{" "}
              of images misclassified after the attack, in {seconds(result.seconds)}.
            </p>
            <ul className="mt-4 grid grid-cols-8 gap-1.5">
              {result.items.map((item) => (
                <li key={item.sample_index} className="text-center">
                  <img
                    src={item.adversarial_image}
                    alt={`Attacked ${item.true_label}, now predicted ${item.adversarial_prediction.label}`}
                    className={`pixelated aspect-square w-full rounded border-2 ${item.fooled ? "border-adv" : "border-clean"}`}
                  />
                  <span className={`text-[11px] ${item.fooled ? "text-adv" : "text-clean"}`}>
                    {item.fooled ? "fooled" : "held"}
                  </span>
                </li>
              ))}
            </ul>
          </div>
          <figure>
            <figcaption className="mb-1 text-sm font-medium">Share of the batch fooled after each step</figcaption>
            <SuccessPerStep values={result.success_per_step} />
          </figure>
        </div>
      )}
    </section>
  );
}

export function AttackLab() {
  const [params] = useSearchParams();
  const models = useResource(api.models);
  const attackable = useMemo(() => (models.data ?? []).filter((model) => model.has_weights), [models.data]);

  const [settings, setSettings] = useState<Settings>({
    modelId: "",
    kind: "pgd",
    levels: 8,
    steps: 10,
    alphaAuto: true,
    alphaLevels: 2,
    randomStart: true,
  });
  const [source, setSource] = useState<Source>({ type: "sample", index: Number(params.get("sample") ?? 0) });
  const [pickClass, setPickClass] = useState<string>("");
  const [live, setLive] = useState(true);
  const [view, setView] = useState<"heatmap" | "signed">("heatmap");
  const [result, setResult] = useState<AttackResponse | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);
  const requestId = useRef(0);

  const update = (patch: Partial<Settings>) => setSettings((current) => ({ ...current, ...patch }));

  useEffect(() => {
    if (!settings.modelId && attackable.length) {
      const preferred = attackable.find((model) => model.mode === "clean") ?? attackable[0];
      update({ modelId: preferred.id });
    }
  }, [attackable, settings.modelId]);

  const request = useMemo(() => (settings.modelId ? toRequest(settings, source) : null), [settings, source]);
  const debounced = useDebounced(request, 300);

  async function run(next: AttackRequest | null = request) {
    if (!next) return;
    const id = ++requestId.current;
    setBusy(true);
    try {
      const response = await api.attack(settings.kind, next);
      if (id === requestId.current) {
        setResult(response);
        setProblem(null);
      }
    } catch (caught) {
      if (id === requestId.current) setProblem(toProblem(caught));
    } finally {
      if (id === requestId.current) setBusy(false);
    }
  }

  useEffect(() => {
    if (live && debounced) void run(debounced);
  }, [debounced, live]);

  async function pickRandom() {
    try {
      setSource({ type: "sample", index: await randomIndex(pickClass || undefined) });
    } catch (caught) {
      setProblem(toProblem(caught));
    }
  }

  function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setSource({ type: "upload", image: String(reader.result), name: file.name, label: null });
    reader.readAsDataURL(file);
  }

  const item = result?.items[0];
  const autoAlpha = (2.5 * settings.levels) / Math.max(settings.steps, 1);

  if (models.loading) return <Spinner label="Loading models" />;
  if (models.error) return <Notice tone="error">{models.error.message}</Notice>;
  if (!attackable.length) {
    return (
      <EmptyState title="No model weights to attack yet">
        Download best.pt files from the Colab run into checkpoints/&lt;run name&gt;/, or train a small model on the
        Training page.
      </EmptyState>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Attack lab">
        Change one pixel budget and watch the prediction. FGSM takes a single signed-gradient step; PGD takes many
        small ones and stays inside the budget after each.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[19rem_1fr]">
        <form className="panel space-y-5 p-5" onSubmit={(event) => event.preventDefault()} aria-label="Attack settings">
          <div>
            <label htmlFor="model" className="field-label">
              Model
            </label>
            <select
              id="model"
              className="input mt-1"
              value={settings.modelId}
              onChange={(event) => update({ modelId: event.target.value })}
            >
              {attackable.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label} ({model.architecture})
                </option>
              ))}
            </select>
          </div>

          <fieldset>
            <legend className="field-label">Attack</legend>
            <div className="mt-1 grid grid-cols-2 gap-1 rounded-md bg-ink/5 p-1">
              {(["fgsm", "pgd"] as const).map((kind) => (
                <label
                  key={kind}
                  className={`cursor-pointer rounded px-3 py-1.5 text-center text-sm font-medium has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-clean ${
                    settings.kind === kind ? "bg-surface shadow-sm" : "text-muted"
                  }`}
                >
                  <input
                    type="radio"
                    name="kind"
                    value={kind}
                    checked={settings.kind === kind}
                    onChange={() => update({ kind })}
                    className="sr-only"
                  />
                  {kind.toUpperCase()}
                </label>
              ))}
            </div>
          </fieldset>

          <EpsilonControl levels={settings.levels} onChange={(levels) => update({ levels })} />

          {settings.kind === "pgd" && (
            <>
              <div>
                <div className="flex justify-between">
                  <label htmlFor="steps" className="field-label">
                    Steps
                  </label>
                  <span className="text-sm tabular-nums">{settings.steps}</span>
                </div>
                <input
                  id="steps"
                  type="range"
                  min={1}
                  max={50}
                  value={settings.steps}
                  onChange={(event) => update({ steps: Number(event.target.value) })}
                  className="mt-1 w-full"
                />
              </div>
              <div>
                <div className="flex items-center justify-between">
                  <label htmlFor="alpha" className="field-label">
                    Step size, alpha
                  </label>
                  <label className="flex items-center gap-1.5 text-xs text-muted">
                    <input
                      type="checkbox"
                      checked={settings.alphaAuto}
                      onChange={(event) => update({ alphaAuto: event.target.checked })}
                    />
                    Automatic
                  </label>
                </div>
                <input
                  id="alpha"
                  type="range"
                  min={0.25}
                  max={8}
                  step={0.25}
                  disabled={settings.alphaAuto}
                  value={settings.alphaAuto ? Math.min(autoAlpha, 8) : settings.alphaLevels}
                  onChange={(event) => update({ alphaLevels: Number(event.target.value) })}
                  className="mt-1 w-full disabled:opacity-40"
                  aria-describedby="alpha-hint"
                />
                <p id="alpha-hint" className="field-hint">
                  {settings.alphaAuto
                    ? `2.5 x epsilon / steps = ${epsilonLabel(autoAlpha / 255)}, enough to cross the whole budget.`
                    : `${epsilonLabel(settings.alphaLevels / 255)} per step.`}
                </p>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={settings.randomStart}
                  onChange={(event) => update({ randomStart: event.target.checked })}
                />
                Start from a random point inside the budget
              </label>
            </>
          )}

          <fieldset className="space-y-2 border-t border-line pt-4">
            <legend className="field-label">Image</legend>
            <div className="flex gap-2">
              <label htmlFor="pick-class" className="sr-only">
                Class to pick from
              </label>
              <select
                id="pick-class"
                className="input"
                value={pickClass}
                onChange={(event) => setPickClass(event.target.value)}
              >
                <option value="">Any class</option>
                {CLASSES.map((name) => (
                  <option key={name} value={name}>
                    {sentenceCase(name)}
                  </option>
                ))}
              </select>
              <Button variant="secondary" onClick={pickRandom} className="shrink-0">
                Random image
              </Button>
            </div>
            <label htmlFor="upload" className="block text-sm text-muted">
              Or upload your own (resized to 32x32)
            </label>
            <input id="upload" type="file" accept="image/png,image/jpeg" onChange={onUpload} className="block w-full text-sm" />
            {source.type === "upload" && (
              <div>
                <label htmlFor="upload-label" className="field-hint">
                  True class of {source.name}, if you know it
                </label>
                <select
                  id="upload-label"
                  className="input mt-1"
                  value={source.label ?? ""}
                  onChange={(event) => setSource({ ...source, label: event.target.value || null })}
                >
                  <option value="">Unknown: attack the current prediction</option>
                  {CLASSES.map((name) => (
                    <option key={name} value={name}>
                      {sentenceCase(name)}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </fieldset>

          <div className="flex items-center justify-between border-t border-line pt-4">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={live} onChange={(event) => setLive(event.target.checked)} />
              Re-attack as I change settings
            </label>
            {!live && (
              <Button onClick={() => run()} busy={busy}>
                Run attack
              </Button>
            )}
          </div>
        </form>

        <section aria-labelledby="result-heading" aria-busy={busy} className="space-y-4">
          <h2 id="result-heading" className="sr-only">
            Attack result
          </h2>
          {problem && <Notice tone="error">{problem.message}</Notice>}
          {!item && !problem && <Spinner label="Running the first attack" />}
          {item && result && (
            <>
              <div
                className={`rounded-md border px-4 py-3 ${
                  item.fooled ? "border-adv/30 bg-adv-soft text-adv" : "border-clean/30 bg-clean-soft text-clean"
                }`}
                role="status"
              >
                <p className="font-semibold">
                  {item.fooled
                    ? `Fooled. The model now says "${item.adversarial_prediction.label}" (${percent(item.adversarial_prediction.confidence, 0)}).`
                    : `Held. The model still says "${item.adversarial_prediction.label}" (${percent(item.adversarial_prediction.confidence, 0)}).`}
                </p>
                {item.true_label && item.clean_prediction.label !== item.true_label && (
                  <p className="mt-1 text-sm">
                    The model already got this {item.true_label} wrong before the attack.
                  </p>
                )}
              </div>

              <div className={`grid gap-4 sm:grid-cols-3 transition-opacity ${busy ? "opacity-60" : ""}`}>
                <figure className="panel p-3">
                  <figcaption className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">Clean</span>
                    <span className="text-muted capitalize">
                      {item.true_label ? `true: ${item.true_label}` : source.type === "upload" ? source.name : ""}
                    </span>
                  </figcaption>
                  <img src={item.clean_image} alt={`Original image${item.true_label ? ` of a ${item.true_label}` : ""}`} className="pixelated aspect-square w-full rounded" />
                  <div className="mt-3">
                    <Predictions title="Prediction before" items={item.clean_top} tone="clean" />
                  </div>
                </figure>

                <figure className="panel p-3">
                  <figcaption className="mb-2 flex items-center justify-between text-sm">
                    <span className="font-medium">Perturbation</span>
                    <span className="flex gap-1 text-xs" role="group" aria-label="Perturbation view">
                      {(["heatmap", "signed"] as const).map((option) => (
                        <button
                          key={option}
                          type="button"
                          aria-pressed={view === option}
                          onClick={() => setView(option)}
                          className={`rounded px-1.5 py-0.5 ${view === option ? "bg-ink text-white" : "text-muted hover:bg-ink/5"}`}
                        >
                          {option === "heatmap" ? "Size" : "Signed"}
                        </button>
                      ))}
                    </span>
                  </figcaption>
                  <img
                    src={view === "heatmap" ? item.heatmap_image : item.perturbation_image}
                    alt={view === "heatmap" ? "Heatmap of how much each pixel changed" : "Signed change per pixel, gray is no change"}
                    className="pixelated aspect-square w-full rounded"
                  />
                  <p className="mt-3 text-xs text-muted">
                    {view === "heatmap"
                      ? "Brighter pixels changed more. Yellow is the full budget."
                      : "Gray is unchanged; colour shows the direction of the change, scaled up to be visible."}
                  </p>
                </figure>

                <figure className="panel p-3">
                  <figcaption className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">Adversarial</span>
                    <span className="text-muted">{result.attack.toUpperCase()}</span>
                  </figcaption>
                  <img src={item.adversarial_image} alt={`Attacked image, predicted as ${item.adversarial_prediction.label}`} className="pixelated aspect-square w-full rounded" />
                  <div className="mt-3">
                    <Predictions title="Prediction after" items={item.adversarial_top} tone="adv" />
                  </div>
                </figure>
              </div>

              <div className="panel grid gap-6 p-4 md:grid-cols-[1fr_1.3fr]">
                <dl className="grid grid-cols-3 gap-4">
                  <Stat label="Largest pixel change" value={epsilonLabel(item.linf)} hint="L-infinity" />
                  <Stat label="Total change" value={item.l2.toFixed(3)} hint="L2 norm" />
                  <Stat label="Run time" value={seconds(result.seconds)} />
                  <Stat label="Steps" value={result.steps} />
                  <Stat label="Step size" value={epsilonLabel(result.alpha)} />
                  <Stat label="Random start" value={result.random_start ? "Yes" : "No"} />
                </dl>
                {result.success_per_step.length > 1 ? (
                  <figure>
                    <figcaption className="mb-1 text-sm font-medium">Fooled after each step</figcaption>
                    <SuccessPerStep values={result.success_per_step} height={150} />
                  </figure>
                ) : (
                  <p className="self-center text-sm text-muted">
                    FGSM is a single step, so there is no per-step curve. Switch to PGD to see the attack converge.
                  </p>
                )}
              </div>
            </>
          )}
        </section>
      </div>

      <BatchCheck settings={settings} />
    </div>
  );
}
