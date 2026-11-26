import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import type { AttackItem, AttackKind, AttackRequest, AttackResponse, ModelEntry, Prediction } from "../api/types";
import { BootPlayground } from "../components/BootPlayground";
import { SuccessPerStep } from "../components/charts";
import {
  Button,
  buttonClass,
  Card,
  CopyCommand,
  EmptyState,
  InfoTip,
  Notice,
  PageHeader,
  Segmented,
  Slider,
  Spinner,
  Stat,
  StepHeader,
} from "../components/ui";
import { epsilonLabel, percent, seconds, sentenceCase } from "../lib/format";
import { useDebounced, useResource } from "../lib/hooks";
import { architectureName } from "../lib/results";

const CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
const BATCH = 16;

type Preset = "trained" | "standard" | "strong" | "custom";
const PRESETS: { value: Preset; label: string; hint: string; levels?: number }[] = [
  { value: "trained", label: "2/255", hint: "Trained budget", levels: 2 },
  { value: "standard", label: "8/255", hint: "Paper standard", levels: 8 },
  { value: "strong", label: "16/255", hint: "Very strong", levels: 16 },
  { value: "custom", label: "Custom", hint: "Use the slider" },
];

type Source = { type: "sample"; index: number } | { type: "upload"; image: string; name: string; label: string | null };

interface Settings {
  modelId: string;
  compareId: string;
  compare: boolean;
  kind: AttackKind;
  levels: number;
  steps: number;
  alphaAuto: boolean;
  alphaLevels: number;
  randomStart: boolean;
}

function toRequest(settings: Settings, source: Source, modelId = settings.modelId): AttackRequest {
  const base = {
    model_id: modelId,
    epsilon: settings.levels / 255,
    steps: settings.kind === "pgd" ? settings.steps : 1,
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

function modelName(model: ModelEntry | undefined): string {
  if (!model) return "";
  return `${model.label} (${architectureName(model.architecture)}${model.owner === "you" ? ", your run" : ""})`;
}

function Predictions({ title, items, tone }: { title: string; items: Prediction[]; tone: "clean" | "adv" }) {
  return (
    <div>
      <p className="text-xs text-muted">{title}</p>
      <ul className="mt-1.5 space-y-1.5">
        {items.map((item, position) => (
          <li key={item.label} className="text-sm">
            <div className="flex justify-between gap-2">
              <span className={`capitalize ${position === 0 ? "font-semibold text-ink" : "text-muted"}`}>{item.label}</span>
              <span className="font-mono text-xs tabular-nums text-muted">{percent(item.confidence)}</span>
            </div>
            <span className="mt-0.5 block h-1 overflow-hidden rounded-full bg-raised" aria-hidden="true">
              <span
                className={`block h-full ${tone === "clean" ? "bg-clean" : "bg-adv"} ${position === 0 ? "" : "opacity-50"}`}
                style={{ width: `${item.confidence * 100}%` }}
              />
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Verdict({ item }: { item: AttackItem }) {
  const after = item.adversarial_prediction;
  const before = item.clean_prediction;
  return (
    <div
      className={`rounded-xl border px-5 py-4 ${item.fooled ? "border-adv/40 bg-adv-soft" : "border-clean/40 bg-clean-soft"}`}
      role="status"
    >
      <p className={`font-display text-2xl font-semibold ${item.fooled ? "text-adv" : "text-clean"}`}>
        {item.fooled ? "Fooled" : "Held"}
      </p>
      <p className="mt-1 text-sm text-ink">
        {item.fooled
          ? `The model changed its answer from "${before.label}" to "${after.label}" (${percent(after.confidence, 0)} sure).`
          : `The model still says "${after.label}" (${percent(after.confidence, 0)} sure). The attack could not flip it within this budget.`}
      </p>
      {item.true_label && before.label !== item.true_label && (
        <p className="mt-1 text-sm text-warn">
          Note: the model already got this {item.true_label} wrong before the attack.
        </p>
      )}
    </div>
  );
}

function confidenceChange(item: AttackItem): string {
  const label = item.clean_prediction.label;
  const before = percent(item.clean_prediction.confidence, 0);
  const after = item.adversarial_top.find((prediction) => prediction.label === label);
  if (after) return `The model's confidence in "${label}" went from ${before} to ${percent(after.confidence, 0)}.`;
  const lowest = item.adversarial_top[item.adversarial_top.length - 1];
  return `The model's confidence in "${label}" went from ${before} to below ${percent(lowest?.confidence ?? 0, 1)}: it is no longer even in the top ${item.adversarial_top.length}.`;
}

function Explainer({ result, item }: { result: AttackResponse; item: AttackItem }) {
  const pgd = result.attack === "pgd";
  return (
    <Card title="What just happened">
      <ol className="list-decimal space-y-2 pl-5 text-sm leading-relaxed text-muted marker:text-muted">
        <li>
          The server asked the model which direction of change would most increase its error on this image (its{" "}
          <span className="text-ink">gradient</span>
          {item.true_label ? `, using the true label "${item.true_label}"` : ", using its own prediction as the label"}).
        </li>
        <li>
          {pgd ? (
            <>
              <span className="text-ink">PGD</span> took {result.steps} small steps of {epsilonLabel(result.alpha)}
              {result.random_start ? ", starting from a random point inside the budget" : ""}. After each step it
              clipped every pixel back to within {epsilonLabel(result.epsilon)} of the original.
            </>
          ) : (
            <>
              <span className="text-ink">FGSM</span> took one single step, moving every pixel by exactly{" "}
              {epsilonLabel(result.epsilon)} in the harmful direction.
            </>
          )}
        </li>
        <li>
          The biggest change to any pixel is {epsilonLabel(item.linf)} of brightness. That is why the attacked image
          looks the same as the original to you.
        </li>
        <li>{confidenceChange(item)}</li>
      </ol>
      {item.fooled ? (
        <p className="mt-4 text-sm">
          Try the same image on a defended model with <span className="font-medium">Compare with another model</span>,
          or lower the budget to find where it stops working.
        </p>
      ) : (
        <p className="mt-4 text-sm">Raise the budget or switch to PGD with more steps to see if the model eventually breaks.</p>
      )}
    </Card>
  );
}

function ComparePanel({
  primary,
  other,
  primaryModel,
  otherModel,
  busy,
}: {
  primary: AttackResponse;
  other: AttackResponse | null;
  primaryModel?: ModelEntry;
  otherModel?: ModelEntry;
  busy: boolean;
}) {
  const columns = [
    { model: primaryModel, result: primary },
    { model: otherModel, result: other },
  ];
  return (
    <Card
      title="Side by side"
      description="Same image, same budget. Each model is attacked with its own gradients, so this is a fair fight for both."
    >
      <div className={`grid gap-4 sm:grid-cols-2 ${busy ? "opacity-60" : ""}`}>
        {columns.map(({ model, result }, index) => {
          const item = result?.items[0];
          return (
            <figure key={index} className="rounded-lg border border-line bg-paper p-3">
              <figcaption className="mb-2 text-sm font-medium">{modelName(model) || "Loading"}</figcaption>
              {item ? (
                <>
                  <img
                    src={item.adversarial_image}
                    alt={`Attacked image, ${model?.label ?? "model"} predicts ${item.adversarial_prediction.label}`}
                    className={`pixelated aspect-square w-full rounded border-2 ${item.fooled ? "border-adv" : "border-clean"}`}
                  />
                  <p className={`mt-2 text-sm font-semibold ${item.fooled ? "text-adv" : "text-clean"}`}>
                    {item.fooled ? "Fooled" : "Held"}: says "{item.adversarial_prediction.label}" ({percent(item.adversarial_prediction.confidence, 0)})
                  </p>
                </>
              ) : (
                <Spinner label="Attacking" />
              )}
            </figure>
          );
        })}
      </div>
    </Card>
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
    <Card
      title="Batch check"
      description={`One image can be lucky. Attack ${BATCH} random test images with the current settings to see how often the attack wins.`}
      actions={
        <Button variant="secondary" busy={busy} onClick={run} disabled={!settings.modelId}>
          Attack {BATCH} random images
        </Button>
      }
    >
      {problem && <Notice tone="error">{problem.message}</Notice>}
      {!result && !problem && <p className="text-sm text-muted">No batch run yet.</p>}
      {result && (
        <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <div>
            <p className="text-sm text-muted">
              <span className="font-mono text-3xl font-semibold text-adv tabular-nums">{percent(result.success_rate, 0)}</span>{" "}
              of images fooled, in {seconds(result.seconds)}.
            </p>
            <ul className="mt-4 grid grid-cols-8 gap-1.5">
              {result.items.map((item) => (
                <li key={item.sample_index} className="text-center">
                  <img
                    src={item.adversarial_image}
                    alt={`Attacked ${item.true_label}, now predicted ${item.adversarial_prediction.label}`}
                    className={`pixelated aspect-square w-full rounded border-2 ${item.fooled ? "border-adv" : "border-clean"}`}
                  />
                  <span className={`text-[11px] ${item.fooled ? "text-adv" : "text-clean"}`}>{item.fooled ? "fooled" : "held"}</span>
                </li>
              ))}
            </ul>
          </div>
          {result.success_per_step.length > 1 && (
            <figure>
              <figcaption className="mb-1 text-sm font-medium">Share of the batch fooled after each PGD step</figcaption>
              <SuccessPerStep values={result.success_per_step} />
            </figure>
          )}
        </div>
      )}
    </Card>
  );
}

export function AttackLab() {
  const [params] = useSearchParams();
  const models = useResource(api.models);
  const attackable = useMemo(() => (models.data ?? []).filter((model) => model.has_weights), [models.data]);

  const [settings, setSettings] = useState<Settings>({
    modelId: "",
    compareId: "",
    compare: false,
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
  const [compareResult, setCompareResult] = useState<AttackResponse | null>(null);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);
  const requestId = useRef(0);

  const update = (patch: Partial<Settings>) => setSettings((current) => ({ ...current, ...patch }));
  const preset: Preset = PRESETS.find((item) => item.levels === settings.levels)?.value ?? "custom";

  useEffect(() => {
    if (!settings.modelId && attackable.length) {
      const baseline = attackable.find((model) => model.mode === "clean") ?? attackable[0];
      const defended = attackable.find((model) => model.id !== baseline.id && model.mode !== "clean") ?? attackable.find((model) => model.id !== baseline.id);
      update({ modelId: baseline.id, compareId: defended?.id ?? "" });
    }
  }, [attackable, settings.modelId]);

  const comparing = settings.compare && Boolean(settings.compareId) && settings.compareId !== settings.modelId;
  const request = useMemo(() => (settings.modelId ? toRequest(settings, source) : null), [settings, source]);
  const debounced = useDebounced(request, 300);

  async function run(next: AttackRequest | null = request) {
    if (!next) return;
    const id = ++requestId.current;
    setBusy(true);
    try {
      const [response, other] = await Promise.all([
        api.attack(settings.kind, next),
        comparing ? api.attack(settings.kind, { ...next, model_id: settings.compareId }) : Promise.resolve(null),
      ]);
      if (id === requestId.current) {
        setResult(response);
        setCompareResult(other);
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
  }, [debounced, live, comparing, settings.compareId]);

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
  const selected = attackable.find((model) => model.id === settings.modelId);
  const compareModel = attackable.find((model) => model.id === settings.compareId);

  if (models.loading) return <Spinner label="Loading models" />;
  if (models.error) return <Notice tone="error">{models.error.message}</Notice>;
  if (!attackable.length) {
    return (
      <EmptyState
        title="No model weights to attack yet"
        action={
          <>
            <Link to="/training" className={buttonClass("primary")}>
              Train a quick model (about 1 minute)
            </Link>
            <Link to="/guide#setup" className={buttonClass("secondary")}>
              Check setup status
            </Link>
          </>
        }
      >
        <p>Live attacks need a trained model file (best.pt). Train a small one here, or install the reported ResNet-18 weights:</p>
        <div className="mx-auto mt-3 max-w-md">
          <CopyCommand command="python scripts/install_checkpoints.py <Drive download>" />
        </div>
      </EmptyState>
    );
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Attack lab">
        Pick a model and an image, then choose how hard to attack. The result updates as you move the controls.
        Rose means the attack fooled the model; ice blue means the model held.
      </PageHeader>

      <div className="grid gap-6 lg:grid-cols-[22rem_1fr]">
        <form className="panel h-fit space-y-6 p-5 lg:sticky lg:top-6" onSubmit={(event) => event.preventDefault()} aria-label="Attack settings">
          <div className="space-y-3">
            <StepHeader step={1} title="Choose a model">
              Undefended models break easily. Defended ones were trained to resist.
            </StepHeader>
            <label htmlFor="model" className="sr-only">
              Model to attack
            </label>
            <select id="model" className="input" value={settings.modelId} onChange={(event) => update({ modelId: event.target.value })}>
              {attackable.map((model) => (
                <option key={model.id} value={model.id}>
                  {modelName(model)}
                </option>
              ))}
            </select>
            {selected?.summary && (
              <p className="text-xs text-muted">
                Reported: {percent(selected.summary.clean_accuracy, 0)} clean, {percent(selected.summary.pgd_accuracy, 0)} under PGD at{" "}
                {epsilonLabel(selected.summary.epsilon)}.
              </p>
            )}
            {attackable.length > 1 && (
              <div className="rounded-lg border border-line bg-paper p-3">
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input type="checkbox" checked={settings.compare} onChange={(event) => update({ compare: event.target.checked })} />
                  Compare with another model
                </label>
                {settings.compare && (
                  <>
                    <label htmlFor="compare-model" className="sr-only">
                      Model to compare with
                    </label>
                    <select
                      id="compare-model"
                      className="input mt-2"
                      value={settings.compareId}
                      onChange={(event) => update({ compareId: event.target.value })}
                    >
                      {attackable
                        .filter((model) => model.id !== settings.modelId)
                        .map((model) => (
                          <option key={model.id} value={model.id}>
                            {modelName(model)}
                          </option>
                        ))}
                    </select>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="space-y-3 border-t border-line pt-5">
            <StepHeader step={2} title="Choose an image">
              A random CIFAR-10 test image the model never trained on.
            </StepHeader>
            <div className="flex gap-2">
              <label htmlFor="pick-class" className="sr-only">
                Class to pick from
              </label>
              <select id="pick-class" className="input" value={pickClass} onChange={(event) => setPickClass(event.target.value)}>
                <option value="">Any class</option>
                {CLASSES.map((name) => (
                  <option key={name} value={name}>
                    {sentenceCase(name)}
                  </option>
                ))}
              </select>
              <Button variant="secondary" onClick={pickRandom} className="shrink-0">
                New image
              </Button>
            </div>
            <details className="group text-sm">
              <summary className="cursor-pointer text-muted hover:text-ink">
                <span className="mr-1 inline-block transition-transform group-open:rotate-90" aria-hidden="true">
                  &#9656;
                </span>
                Or upload your own photo
              </summary>
              <div className="mt-2 space-y-2">
                <label htmlFor="upload" className="field-hint block">
                  PNG or JPEG. It is shrunk to 32x32 pixels, like CIFAR-10.
                </label>
                <input id="upload" type="file" accept="image/png,image/jpeg" onChange={onUpload} className="block w-full text-sm text-muted" />
                {source.type === "upload" && (
                  <div>
                    <label htmlFor="upload-label" className="field-hint">
                      What is in {source.name}?
                    </label>
                    <select
                      id="upload-label"
                      className="input mt-1"
                      value={source.label ?? ""}
                      onChange={(event) => setSource({ ...source, label: event.target.value || null })}
                    >
                      <option value="">Not sure: attack whatever the model sees</option>
                      {CLASSES.map((name) => (
                        <option key={name} value={name}>
                          {sentenceCase(name)}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            </details>
          </div>

          <div className="space-y-4 border-t border-line pt-5">
            <StepHeader step={3} title="Set the attack">
              Bigger budgets fool models more often but start to become visible.
            </StepHeader>
            <Segmented
              legend={
                <span className="flex items-center gap-1.5">
                  Method <InfoTip term={settings.kind === "pgd" ? "pgd" : "fgsm"} />
                </span>
              }
              name="kind"
              tone="adv"
              value={settings.kind}
              onChange={(kind) => update({ kind })}
              options={[
                { value: "fgsm", label: "FGSM", hint: "One big step, fast" },
                { value: "pgd", label: "PGD", hint: "Many steps, stronger" },
              ]}
            />
            <Segmented
              legend={
                <span className="flex items-center gap-1.5">
                  Strength <InfoTip term="epsilon" />
                </span>
              }
              name="preset"
              tone="adv"
              columns={2}
              value={preset}
              onChange={(value) => {
                const levels = PRESETS.find((item) => item.value === value)?.levels;
                if (levels !== undefined) update({ levels });
              }}
              options={PRESETS.map((item) => ({ value: item.value, label: item.label, hint: item.hint }))}
            />
            <Slider
              id="epsilon"
              label="Budget per pixel"
              tone="adv"
              min={0}
              max={16}
              step={0.5}
              value={settings.levels}
              onChange={(levels) => update({ levels })}
              display={epsilonLabel(settings.levels / 255)}
              hint="Brightness levels out of 255. The defended models were trained at 2/255."
            />

            {settings.kind === "pgd" && (
              <details className="group rounded-lg border border-line bg-paper px-3 py-2 text-sm">
                <summary className="cursor-pointer py-1 font-medium text-muted hover:text-ink">
                  <span className="mr-1 inline-block transition-transform group-open:rotate-90" aria-hidden="true">
                    &#9656;
                  </span>
                  Advanced PGD settings
                </summary>
                <div className="mt-3 space-y-4 pb-2">
                  <Slider
                    id="steps"
                    label="Steps"
                    min={1}
                    max={50}
                    value={settings.steps}
                    onChange={(steps) => update({ steps })}
                    hint="More steps make a stronger attack. 10 matches the reported results."
                    tone="adv"
                  />
                  <div>
                    <Slider
                      id="alpha"
                      label="Step size"
                      term="alpha"
                      min={0.25}
                      max={8}
                      step={0.25}
                      disabled={settings.alphaAuto}
                      value={settings.alphaAuto ? Math.min(autoAlpha, 8) : settings.alphaLevels}
                      onChange={(alphaLevels) => update({ alphaLevels })}
                      display={epsilonLabel((settings.alphaAuto ? autoAlpha : settings.alphaLevels) / 255)}
                      tone="adv"
                    />
                    <label className="mt-1.5 flex items-center gap-2 text-xs text-muted">
                      <input type="checkbox" checked={settings.alphaAuto} onChange={(event) => update({ alphaAuto: event.target.checked })} />
                      Automatic (2.5 x budget / steps)
                    </label>
                  </div>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={settings.randomStart} onChange={(event) => update({ randomStart: event.target.checked })} />
                    Random start
                    <InfoTip term="randomStart" />
                  </label>
                </div>
              </details>
            )}
          </div>

          <div className="flex items-center justify-between gap-3 border-t border-line pt-4">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={live} onChange={(event) => setLive(event.target.checked)} />
              Update automatically
            </label>
            <Button variant="attack" onClick={() => run()} busy={busy}>
              {live ? "Run again" : "Run attack"}
            </Button>
          </div>
        </form>

        <section aria-labelledby="result-heading" aria-busy={busy} className="min-w-0 space-y-5">
          <h2 id="result-heading" className="sr-only">
            Attack result
          </h2>
          {problem && <Notice tone="error">{problem.message}</Notice>}
          {!item && !problem && (
            <div className="panel space-y-5 p-5 sm:p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <Spinner label={busy ? "Running the attack on the server" : "Preparing the first attack"} />
                <p className="text-xs text-muted">On a small CPU this can take a minute — play below meanwhile.</p>
              </div>
              <BootPlayground />
            </div>
          )}
          {item && result && busy && (
            <Notice tone="warn" title="Attack running">
              Updating the images on the server. You can keep painting the patch below while you wait.
              <div className="mt-4">
                <BootPlayground />
              </div>
            </Notice>
          )}
          {item && result && (
            <>
              <Verdict item={item} />

              <div className={`grid gap-4 sm:grid-cols-3 transition-opacity ${busy ? "opacity-60" : ""}`}>
                <figure className="panel p-3">
                  <figcaption className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">Original</span>
                    <span className="capitalize text-muted">
                      {item.true_label ? item.true_label : source.type === "upload" ? source.name : ""}
                    </span>
                  </figcaption>
                  <img
                    src={item.clean_image}
                    alt={`Original image${item.true_label ? ` of a ${item.true_label}` : ""}`}
                    className="pixelated aspect-square w-full rounded border-2 border-clean/60"
                  />
                  <div className="mt-3">
                    <Predictions title="Model's answer before" items={item.clean_top} tone="clean" />
                  </div>
                </figure>

                <figure className="panel p-3">
                  <figcaption className="mb-2 flex items-center justify-between text-sm">
                    <span className="font-medium">The change</span>
                    <span className="flex gap-1 text-xs" role="group" aria-label="How to show the change">
                      {(["heatmap", "signed"] as const).map((option) => (
                        <button
                          key={option}
                          type="button"
                          aria-pressed={view === option}
                          onClick={() => setView(option)}
                          className={`rounded px-1.5 py-0.5 ${view === option ? "bg-raised text-ink" : "text-muted hover:text-ink"}`}
                        >
                          {option === "heatmap" ? "Size" : "Direction"}
                        </button>
                      ))}
                    </span>
                  </figcaption>
                  <img
                    src={view === "heatmap" ? item.heatmap_image : item.perturbation_image}
                    alt={view === "heatmap" ? "Heatmap of how much each pixel changed" : "Direction of change per pixel, gray is no change"}
                    className="pixelated aspect-square w-full rounded border-2 border-line"
                  />
                  <p className="mt-3 text-xs leading-relaxed text-muted">
                    {view === "heatmap"
                      ? "Amplified so you can see it. Brighter pixels changed more; yellow is the full budget."
                      : "Amplified. Gray is unchanged; colours show which way each channel was pushed."}
                  </p>
                </figure>

                <figure className="panel p-3">
                  <figcaption className="mb-2 flex justify-between text-sm">
                    <span className="font-medium">Attacked</span>
                    <span className="text-muted">
                      {result.attack.toUpperCase()}, {epsilonLabel(result.epsilon)}
                    </span>
                  </figcaption>
                  <img
                    src={item.adversarial_image}
                    alt={`Attacked image, predicted as ${item.adversarial_prediction.label}`}
                    className={`pixelated aspect-square w-full rounded border-2 ${item.fooled ? "border-adv" : "border-clean/60"}`}
                  />
                  <div className="mt-3">
                    <Predictions title="Model's answer after" items={item.adversarial_top} tone="adv" />
                  </div>
                </figure>
              </div>

              {comparing && <ComparePanel primary={result} other={compareResult} primaryModel={selected} otherModel={compareModel} busy={busy} />}

              <Explainer result={result} item={item} />

              <Card title="Measurements">
                <div className="grid gap-6 md:grid-cols-[1fr_1.3fr]">
                  <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                    <Stat label={<span className="inline-flex items-center gap-1">Largest change <InfoTip term="linf" /></span>} value={epsilonLabel(item.linf)} />
                    <Stat label={<span className="inline-flex items-center gap-1">Total change <InfoTip term="l2" /></span>} value={item.l2.toFixed(3)} />
                    <Stat label="Run time" value={seconds(result.seconds)} />
                    <Stat label="Steps" value={result.steps} />
                    <Stat label="Step size" value={epsilonLabel(result.alpha)} />
                    <Stat label="Random start" value={result.random_start ? "Yes" : "No"} />
                  </dl>
                  {result.success_per_step.length > 1 ? (
                    <figure>
                      <figcaption className="mb-1 text-sm font-medium">Was the model fooled after each step?</figcaption>
                      <SuccessPerStep values={result.success_per_step} height={150} />
                    </figure>
                  ) : (
                    <p className="self-center text-sm text-muted">
                      FGSM is a single step, so there is no per-step curve. Switch to PGD to watch the attack converge.
                    </p>
                  )}
                </div>
              </Card>
            </>
          )}
        </section>
      </div>

      <BatchCheck settings={settings} />
    </div>
  );
}
