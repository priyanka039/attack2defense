import { useEffect, useState, type FormEvent } from "react";
import { Link, useLocation } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type CertifyJob } from "../api/types";
import { useAuth } from "../auth/AuthContext";
import { useToast } from "../components/Toast";
import { CertifiedAccuracy } from "../components/charts";
import {
  Button,
  buttonClass,
  Card,
  EmptyState,
  Notice,
  PageHeader,
  Segmented,
  Slider,
  Spinner,
  StatusBadge,
  Term,
} from "../components/ui";
import { dateTime, percent } from "../lib/format";
import { JobWaitPanel } from "../components/WaitPanel";
import { usePolling, useResource } from "../lib/hooks";
import { architectureName } from "../lib/results";

type PresetId = "quick" | "careful";
// Keep Monte Carlo counts small enough for a free CPU host.
const PRESETS: Record<PresetId, { n0: number; n: number; perClass: number }> = {
  quick: { n0: 10, n: 50, perClass: 2 },
  careful: { n0: 25, n: 100, perClass: 5 },
};

function CertifyForm({ onStarted }: { onStarted: (job: CertifyJob) => void }) {
  const { notify } = useToast();
  const models = useResource(api.models);
  const usable = (models.data ?? []).filter((model) => model.has_weights);
  const [modelId, setModelId] = useState("");
  const [sigma, setSigma] = useState(0.25);
  const [preset, setPreset] = useState<PresetId>("quick");
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!modelId && usable.length) setModelId((usable.find((model) => model.mode === "gaussian") ?? usable[0]).id);
  }, [modelId, usable]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    const { n0, n, perClass } = PRESETS[preset];
    try {
      onStarted(await api.startCertify({ model_id: modelId, sigma, n0, n, alpha: 0.001, images_per_class: perClass }));
      notify("Certification started.", "success");
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
        title="No model weights to certify"
        action={
          <Link to="/training" className={buttonClass("primary")}>
            Train a noise-trained model
          </Link>
        }
      >
        Certification needs a model file. Noise training (the Gaussian option on the Training page) is the matching defense.
      </EmptyState>
    );
  }

  return (
    <form onSubmit={submit} className="panel space-y-6 p-5 sm:p-6" aria-label="New certification">
      <div className="grid gap-5 md:grid-cols-2">
        <div>
          <label htmlFor="cert-model" className="field-label">
            Model
          </label>
          <select id="cert-model" className="input mt-1" value={modelId} onChange={(event) => setModelId(event.target.value)}>
            {usable.map((model) => (
              <option key={model.id} value={model.id}>
                {model.label} ({architectureName(model.architecture)})
              </option>
            ))}
          </select>
          <p className="field-hint mt-1">Any model can be certified; noise-trained models get the largest radii.</p>
        </div>
        <Slider
          id="sigma"
          label="Noise sigma"
          term="sigma"
          min={0.05}
          max={1}
          step={0.05}
          value={sigma}
          onChange={setSigma}
          display={sigma.toFixed(2)}
          hint="Match the sigma the model was noise-trained with (0.25 by default)."
        />
      </div>
      <Segmented
        legend="How careful"
        name="cert-preset"
        value={preset}
        onChange={setPreset}
        options={[
          { value: "quick", label: "Quick", hint: "50 images, 100 noisy copies each" },
          { value: "careful", label: "Careful", hint: "100 images, 500 noisy copies each" },
        ]}
      />
      {problem && <Notice tone="error">{problem.message}</Notice>}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
        <p className="text-sm text-muted">More noisy copies give tighter, larger guarantees but take longer.</p>
        <Button type="submit" busy={busy} disabled={!modelId}>
          Certify model
        </Button>
      </div>
    </form>
  );
}

function Jobs() {
  const { notify } = useToast();
  const jobs = useResource(api.certifyJobs);
  const [selected, setSelected] = useState<string | null>(null);
  const list = jobs.data ?? [];
  const current = list.find((job) => job.id === selected) ?? list[0];

  const active = list.some((job) => isActive(job.status));
  usePolling(() => void jobs.reload(), active, list.some((job) => job.status === "queued") ? 800 : 1500);

  async function cancel(id: string) {
    try {
      await api.cancelCertify(id);
      await jobs.reload();
    } catch (caught) {
      notify(toProblem(caught).message, "error");
    }
  }

  return (
    <>
      <CertifyForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />
      <section aria-labelledby="cert-jobs-heading">
        <h2 id="cert-jobs-heading" className="text-xl font-semibold">
          Your certifications
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading certifications" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No certifications yet">A quick certification takes a few minutes on a CPU.</EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[17rem_1fr]">
            <ul className="space-y-1.5" aria-label="Certifications">
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
                      <span className="truncate font-medium">{job.params.model_id}</span>
                      <StatusBadge status={job.status} />
                    </span>
                    <span className="mt-0.5 block text-xs text-muted">
                      sigma {job.params.sigma}, {dateTime(job.created_at)}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            <Card>
              {isActive(current.status) && (
                <div className="space-y-3">
                  <div className="flex justify-end">
                    <Button variant="danger" size="sm" onClick={() => cancel(current.id)} disabled={current.status === "cancelling"}>
                      Stop certification
                    </Button>
                  </div>
                  <JobWaitPanel
                    status={current.status}
                    progress={current.progress}
                    message={current.message}
                    startedAt={current.started_at}
                    createdAt={current.created_at}
                    label="Certification progress"
                  />
                </div>
              )}
              {current.error && <Notice tone="error" title="Certification failed">{current.error}</Notice>}
              {current.status === "cancelled" && <Notice>This certification was stopped before it finished.</Notice>}
              {current.status === "succeeded" && current.result && (
                <div className="space-y-4">
                  <p className="text-sm text-muted">
                    {current.result.images} images, sigma {current.result.sigma}, {current.result.n} noisy copies per image. Accuracy of the
                    smoothed model with no radius required: <span className="font-mono text-ink">{percent(current.result.naive_accuracy)}</span>.
                  </p>
                  <CertifiedAccuracy curve={current.result.curve} />
                  <p className="text-xs text-muted">{current.result.note}</p>
                </div>
              )}
            </Card>
          </div>
        )}
      </section>
    </>
  );
}

export function Certified() {
  const { user, ready } = useAuth();
  const location = useLocation();
  const publicResults = useResource(api.publicResults);
  const archived = publicResults.data?.certified ?? [];

  return (
    <div className="space-y-8">
      <PageHeader title="Certified defense">
        Attacks can only show that a model <em>failed</em>. They cannot prove it is safe, because a cleverer attack
        might exist. <Term id="smoothing">Randomized smoothing</Term> gives a mathematical guarantee instead.
      </PageHeader>

      <section aria-label="Tested versus proven" className="grid gap-4 md:grid-cols-2">
        <div className="panel p-5">
          <p className="font-semibold text-adv">Tested (Attack lab, Evaluation)</p>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            "PGD could not fool the model on 79% of images." Strong evidence, but only against the attacks we tried, and
            measured in the L-infinity sense (biggest single pixel change).
          </p>
        </div>
        <div className="panel p-5">
          <p className="font-semibold text-clean">Proven (this page)</p>
          <p className="mt-2 text-sm leading-relaxed text-muted">
            "No change of total size below r can alter this answer, whatever the attacker does." The curve shows the share
            of images with such a <Term id="certifiedRadius">certified radius</Term> of at least r, measured in L2.
          </p>
        </div>
      </section>

      {archived.length > 0 && (
        <Card title="Reported certification" description="From the Colab results. No login needed.">
          <div className="space-y-6">
            {archived.map((item) => (
              <div key={item.source}>
                <h3 className="mb-2 font-mono text-sm text-muted">{item.source}</h3>
                <CertifiedAccuracy curve={item.curve} />
              </div>
            ))}
          </div>
        </Card>
      )}

      {!ready ? (
        <Spinner label="Checking your session" />
      ) : user ? (
        <Jobs />
      ) : (
        <EmptyState
          title="Log in to certify a model"
          action={
            <Link to="/login" state={{ from: location.pathname }} className={buttonClass("primary")}>
              Log in or create an account
            </Link>
          }
        >
          Certification runs on the server and is saved to your account.
        </EmptyState>
      )}
    </div>
  );
}
