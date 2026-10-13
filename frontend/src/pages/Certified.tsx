import { useEffect, useState, type FormEvent } from "react";

import { toProblem, type ApiProblem } from "../api/client";
import { api } from "../api/endpoints";
import { isActive, type CertifyJob } from "../api/types";
import { useToast } from "../components/Toast";
import { CertifiedAccuracy } from "../components/charts";
import { Button, EmptyState, Notice, PageHeader, ProgressBar, Spinner, StatusBadge } from "../components/ui";
import { dateTime, percent } from "../lib/format";
import { usePolling, useResource } from "../lib/hooks";

function CertifyForm({ onStarted }: { onStarted: (job: CertifyJob) => void }) {
  const { notify } = useToast();
  const models = useResource(api.models);
  const usable = (models.data ?? []).filter((model) => model.has_weights);
  const [modelId, setModelId] = useState("");
  const [sigma, setSigma] = useState(0.25);
  const [n0, setN0] = useState(25);
  const [n, setN] = useState(100);
  const [perClass, setPerClass] = useState(5);
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!modelId && usable.length) setModelId(usable[0].id);
  }, [modelId, usable]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      onStarted(
        await api.startCertify({
          model_id: modelId,
          sigma,
          n0,
          n,
          alpha: 0.001,
          images_per_class: perClass,
        }),
      );
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
      <EmptyState title="No model weights to certify">
        Add Colab checkpoints to checkpoints/ or train a Gaussian-noise model first.
      </EmptyState>
    );
  }

  return (
    <form onSubmit={submit} className="panel space-y-5 p-5" aria-label="New certification">
      <div>
        <label htmlFor="cert-model" className="field-label">
          Model
        </label>
        <select
          id="cert-model"
          className="input mt-1"
          value={modelId}
          onChange={(event) => setModelId(event.target.value)}
        >
          {usable.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label} ({model.id})
            </option>
          ))}
        </select>
        <p className="field-hint">Gaussian-noise training is the matching defense; any checkpoint can be certified.</p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <label htmlFor="sigma" className="field-label">
            Noise sigma
          </label>
          <input
            id="sigma"
            type="number"
            min={0.05}
            max={1}
            step={0.05}
            value={sigma}
            onChange={(event) => setSigma(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
        </div>
        <div>
          <label htmlFor="n0" className="field-label">
            Selection samples
          </label>
          <input
            id="n0"
            type="number"
            min={10}
            max={200}
            value={n0}
            onChange={(event) => setN0(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
        </div>
        <div>
          <label htmlFor="n" className="field-label">
            Bound samples
          </label>
          <input
            id="n"
            type="number"
            min={20}
            max={2000}
            value={n}
            onChange={(event) => setN(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
        </div>
        <div>
          <label htmlFor="cert-per-class" className="field-label">
            Images per class
          </label>
          <input
            id="cert-per-class"
            type="number"
            min={1}
            max={50}
            value={perClass}
            onChange={(event) => setPerClass(Number(event.target.value))}
            className="input mt-1 tabular-nums"
          />
        </div>
      </div>
      {problem && <Notice tone="error">{problem.message}</Notice>}
      <div className="flex justify-end border-t border-line pt-4">
        <Button type="submit" busy={busy} disabled={!modelId}>
          Certify model
        </Button>
      </div>
    </form>
  );
}

export function Certified() {
  const { notify } = useToast();
  const publicResults = useResource(api.publicResults);
  const jobs = useResource(api.certifyJobs);
  const [selected, setSelected] = useState<string | null>(null);
  const list = jobs.data ?? [];
  const current = list.find((job) => job.id === selected) ?? list[0];
  const archived = publicResults.data?.certified ?? [];

  usePolling(() => void jobs.reload(), list.some((job) => isActive(job.status)));

  async function cancel(id: string) {
    try {
      await api.cancelCertify(id);
      await jobs.reload();
    } catch (caught) {
      notify(toProblem(caught).message, "error");
    }
  }

  return (
    <div className="space-y-8">
      <PageHeader title="Certified defense">
        Randomized smoothing gives a provable L2 radius. The curve below is not empirical accuracy after FGSM
        or PGD; it is the share of images that stay correct for every perturbation inside that radius.
      </PageHeader>

      <CertifyForm
        onStarted={(job) => {
          setSelected(job.id);
          void jobs.reload();
        }}
      />

      {archived.length > 0 && (
        <section aria-labelledby="archived-cert-heading">
          <h2 id="archived-cert-heading" className="text-lg font-semibold">
            Reported certification
          </h2>
          <p className="mt-1 text-sm text-muted">From the Colab results. No login needed.</p>
          <div className="mt-4 space-y-6">
            {archived.map((item) => (
              <div key={item.source} className="panel p-5">
                <h3 className="font-medium">{item.source}</h3>
                <CertifiedAccuracy curve={item.curve} />
              </div>
            ))}
          </div>
        </section>
      )}

      <section aria-labelledby="cert-jobs-heading">
        <h2 id="cert-jobs-heading" className="text-lg font-semibold">
          Your certifications
        </h2>
        {jobs.loading && !jobs.data && <Spinner label="Loading certifications" />}
        {jobs.error && <Notice tone="error">{jobs.error.message}</Notice>}
        {jobs.data && !list.length && (
          <div className="mt-4">
            <EmptyState title="No certifications yet">
              5 images per class with 100 Monte Carlo samples takes a few minutes on a CPU.
            </EmptyState>
          </div>
        )}
        {current && (
          <div className="mt-4 grid gap-6 lg:grid-cols-[16rem_1fr]">
            <ul className="space-y-1" aria-label="Certifications">
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
                      <span className="font-medium">{job.params.model_id}</span>
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
                      Stop certification
                    </Button>
                  </div>
                  <ProgressBar value={current.progress} label="Certification progress" />
                </div>
              )}
              {current.error && <Notice tone="error">{current.error}</Notice>}
              {current.status === "cancelled" && <Notice>This certification was stopped before it finished.</Notice>}
              {current.status === "succeeded" && current.result && (
                <div className="space-y-4">
                  <Notice>{current.result.note}</Notice>
                  <p className="text-sm text-muted">
                    {current.result.images} images, sigma {current.result.sigma}, p_A bound from{" "}
                    {current.result.n} samples. Majority-class accuracy {percent(current.result.naive_accuracy)}.
                  </p>
                  <CertifiedAccuracy curve={current.result.curve} />
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
