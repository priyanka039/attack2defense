import { useEffect, useState } from "react";

import type { JobStatus } from "../api/types";
import { BootPlayground } from "./BootPlayground";
import { ProgressBar } from "./ui";

const TIPS = [
  "An invisible change of 2/255 brightness can drop an undefended model from ~94% to ~6%.",
  "PGD is stronger than FGSM because it takes many small steps instead of one big one.",
  "Adversarial training teaches the model on attacked images, so it keeps accuracy under attack.",
  "The first open after a restart loads PyTorch into memory — that can take about a minute on a small CPU.",
  "Only one heavy job runs at a time on this server, so a second run waits in line.",
  "The reported ResNet-18 numbers on the Dashboard came from a GPU; demos here use a Small CNN.",
];

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m <= 0) return `${s}s`;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}

function phaseCopy(status: JobStatus, message: string): { title: string; detail: string } {
  if (status === "queued") {
    return {
      title: "In the queue",
      detail: "One job runs at a time on this server. Yours will start as soon as the worker is free.",
    };
  }
  if (status === "cancelling") {
    return { title: "Stopping", detail: "Finishing the current batch, then stopping." };
  }
  const lower = message.toLowerCase();
  if (lower.includes("download") || lower.includes("cifar")) {
    return {
      title: "Preparing data",
      detail: "First run may download CIFAR-10 (~170 MB). Later runs reuse it until the host restarts.",
    };
  }
  if (lower.includes("starting") || message === "Waiting to start") {
    return {
      title: "Starting",
      detail: "Loading the model and dataset. Progress ticks appear after the first batch.",
    };
  }
  return {
    title: "Working",
    detail: message || "Running on the server. Keep this tab open — it updates automatically.",
  };
}

/** Progress + elapsed time + rotating facts so a long wait feels intentional. */
export function JobWaitPanel({
  status,
  progress,
  message,
  startedAt,
  createdAt,
  label = "Job progress",
}: {
  status: JobStatus;
  progress: number;
  message: string;
  startedAt?: string | null;
  createdAt: string;
  label?: string;
}) {
  const origin = Date.parse(startedAt || createdAt);
  const [elapsed, setElapsed] = useState(() => Math.max(0, Math.floor((Date.now() - origin) / 1000)));
  const [tipIndex, setTipIndex] = useState(0);
  const phase = phaseCopy(status, message);
  const indeterminate = status === "queued" || (status === "running" && progress < 0.02);

  useEffect(() => {
    const id = window.setInterval(() => {
      setElapsed(Math.max(0, Math.floor((Date.now() - origin) / 1000)));
    }, 1000);
    return () => window.clearInterval(id);
  }, [origin]);

  useEffect(() => {
    const id = window.setInterval(() => setTipIndex((i) => (i + 1) % TIPS.length), 8000);
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="space-y-3 rounded-lg border border-line bg-raised/40 px-4 py-3" role="status" aria-live="polite">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-sm font-semibold text-ink">{phase.title}</p>
          <p className="mt-0.5 text-sm text-muted">{phase.detail}</p>
        </div>
        <p className="font-mono text-xs tabular-nums text-muted">elapsed {formatElapsed(elapsed)}</p>
      </div>
      <div>
        <div className="mb-1.5 flex justify-between text-sm">
          <span className="text-muted">{message || phase.title}</span>
          {!indeterminate && <span className="font-mono tabular-nums">{Math.round(progress * 100)}%</span>}
        </div>
        <ProgressBar value={progress} label={label} indeterminate={indeterminate} />
      </div>
      <p className="text-xs leading-relaxed text-muted">
        <span className="font-medium text-ink">While you wait · </span>
        {TIPS[tipIndex]}
      </p>
    </div>
  );
}

/** Full wait panel when the API is still booting (PyTorch import) or unreachable. */
export function HostWakeBanner({
  offline,
  onRetry,
  retrying,
}: {
  offline: boolean;
  onRetry: () => void;
  retrying?: boolean;
}) {
  const [seconds, setSeconds] = useState(0);
  const [tipIndex, setTipIndex] = useState(0);

  useEffect(() => {
    if (!offline) {
      setSeconds(0);
      return;
    }
    const id = window.setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => window.clearInterval(id);
  }, [offline]);

  useEffect(() => {
    if (!offline) return;
    const id = window.setInterval(() => onRetry(), 6000);
    return () => window.clearInterval(id);
  }, [offline, onRetry]);

  useEffect(() => {
    if (!offline) return;
    const id = window.setInterval(() => setTipIndex((i) => (i + 1) % TIPS.length), 7000);
    return () => window.clearInterval(id);
  }, [offline]);

  if (!offline) return null;

  const local =
    typeof window !== "undefined" &&
    (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");

  return (
    <div className="border-b border-line bg-surface" role="status" aria-live="polite">
      <div className="mx-auto grid max-w-6xl gap-8 px-4 py-6 sm:px-6 lg:grid-cols-[1.1fr_0.9fr] lg:items-center lg:px-10 lg:py-8">
        <div>
          <p className="font-display text-xl font-semibold text-ink sm:text-2xl">
            {local ? "API not reachable" : "Starting the model server"}
          </p>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
            {local
              ? "Start the backend with uvicorn, then click Try now."
              : "PyTorch is loading into memory. This often takes under a minute on a small CPU — then the site stays responsive. Play the pixel game while you wait."}
          </p>
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <div className="min-w-[12rem] flex-1">
              <ProgressBar value={0} label="API starting" indeterminate />
            </div>
            <span className="font-mono text-xs tabular-nums text-muted">{formatElapsed(seconds)}</span>
            <button
              type="button"
              onClick={onRetry}
              disabled={retrying}
              className="rounded-lg border border-line-strong bg-raised px-3 py-1.5 text-xs font-semibold text-ink hover:border-muted disabled:opacity-60"
            >
              {retrying ? "Checking…" : "Try now"}
            </button>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-muted">
            <span className="font-medium text-ink">Did you know · </span>
            {TIPS[tipIndex]}
          </p>
        </div>
        <div className="justify-self-center lg:justify-self-end">
          <BootPlayground />
        </div>
      </div>
    </div>
  );
}
