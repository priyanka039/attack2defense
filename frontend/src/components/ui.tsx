import { useId, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

import type { JobStatus } from "../api/types";
import { glossary, type GlossaryEntry, type GlossaryKey } from "../lib/glossary";

type Variant = "primary" | "attack" | "secondary" | "danger" | "ghost";
type Size = "sm" | "md";

const variants: Record<Variant, string> = {
  // Fire-orange accent: the only loud colour, reserved for primary CTAs.
  primary: "bg-accent text-accent-ink hover:bg-accent/90 disabled:bg-accent/30 disabled:text-accent-ink/60",
  attack: "bg-accent text-accent-ink hover:bg-accent/90 disabled:bg-accent/30 disabled:text-accent-ink/60",
  secondary: "border border-line-strong bg-raised text-ink hover:border-muted disabled:text-muted",
  danger: "border border-flag/50 bg-transparent text-flag hover:bg-flag-soft disabled:opacity-50",
  ghost: "text-ink hover:bg-raised disabled:text-muted",
};

const sizes: Record<Size, string> = {
  sm: "px-3 py-1.5 text-sm",
  md: "px-4 py-2 text-sm",
};

/** Class names for a link that should look like a button. */
export function buttonClass(variant: Variant = "primary", size: Size = "md"): string {
  return `inline-flex items-center justify-center gap-2 rounded-lg font-semibold transition-colors disabled:cursor-not-allowed ${variants[variant]} ${sizes[size]}`;
}

export function Button({
  variant = "primary",
  size = "md",
  busy = false,
  className = "",
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; busy?: boolean }) {
  return (
    <button
      type="button"
      {...props}
      disabled={props.disabled || busy}
      aria-busy={busy || undefined}
      className={`${buttonClass(variant, size)} ${className}`}
    >
      {busy && <Spinner />}
      {children}
    </button>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role={label ? "status" : undefined}>
      <span
        aria-hidden="true"
        className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
      />
      {label && <span>{label}</span>}
    </span>
  );
}

export function PageHeader({
  title,
  children,
  actions,
}: {
  title: string;
  children?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <h1 className="text-3xl font-semibold sm:text-4xl">{title}</h1>
        {children && <div className="mt-3 text-base leading-relaxed text-muted">{children}</div>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </header>
  );
}

type Tone = "info" | "error" | "warn" | "success";

const noticeStyles: Record<Tone, string> = {
  info: "border-line bg-raised text-ink",
  error: "border-flag/40 bg-flag-soft text-flag",
  warn: "border-warn/40 bg-warn-soft text-warn",
  success: "border-clean/40 bg-clean-soft text-clean",
};

export function Notice({ tone = "info", title, children }: { tone?: Tone; title?: string; children: ReactNode }) {
  return (
    <div role={tone === "error" ? "alert" : undefined} className={`rounded-lg border px-4 py-3 text-sm ${noticeStyles[tone]}`}>
      {title && <p className="mb-0.5 font-semibold">{title}</p>}
      <div className={title ? "text-ink/90" : undefined}>{children}</div>
    </div>
  );
}

export function EmptyState({ title, children, action }: { title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line-strong bg-surface/50 px-6 py-10 text-center">
      <p className="font-display text-lg font-semibold">{title}</p>
      {children && <div className="mx-auto mt-2 max-w-lg text-sm leading-relaxed text-muted">{children}</div>}
      {action && <div className="mt-5 flex flex-wrap justify-center gap-2">{action}</div>}
    </div>
  );
}

const statusStyles: Record<JobStatus, string> = {
  queued: "bg-raised text-muted",
  running: "bg-clean-soft text-clean",
  cancelling: "bg-warn-soft text-warn",
  succeeded: "bg-clean-soft text-clean",
  failed: "bg-flag-soft text-flag",
  cancelled: "bg-raised text-muted",
};

const statusText: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running",
  cancelling: "Stopping",
  succeeded: "Done",
  failed: "Failed",
  cancelled: "Stopped",
};

export function StatusBadge({ status }: { status: JobStatus }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium ${statusStyles[status]}`}>
      {status === "running" && <span aria-hidden="true" className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
      {statusText[status]}
    </span>
  );
}

export function Badge({ tone = "info", children }: { tone?: Tone | "adv"; children: ReactNode }) {
  const styles = {
    info: "border-line-strong text-muted",
    error: "border-flag/40 text-flag",
    warn: "border-warn/40 text-warn",
    success: "border-clean/40 text-clean",
    adv: "border-adv/40 text-adv",
  }[tone];
  return <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${styles}`}>{children}</span>;
}

export function ProgressBar({
  value,
  label,
  tone = "clean",
  indeterminate = false,
}: {
  value: number;
  label: string;
  tone?: "clean" | "adv";
  /** Pulse animation when the server has not reported real progress yet (queue / startup). */
  indeterminate?: boolean;
}) {
  const percent = Math.round(value * 100);
  const fill = tone === "clean" ? "bg-clean" : "bg-adv";
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : percent}
      aria-valuetext={indeterminate ? "Waiting" : undefined}
      className="h-2 overflow-hidden rounded-full bg-raised"
    >
      {indeterminate ? (
        <div className={`h-full w-1/3 rounded-full ${fill} animate-[a2d-indeterminate_1.2s_ease-in-out_infinite]`} />
      ) : (
        <div className={`h-full rounded-full transition-[width] duration-500 ${fill}`} style={{ width: `${percent}%` }} />
      )}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: ReactNode; value: ReactNode; hint?: string }) {
  return (
    <div>
      <dt className="text-xs text-muted">{label}</dt>
      <dd className="mt-0.5 font-mono text-lg font-semibold tabular-nums">{value}</dd>
      {hint && <dd className="text-xs text-muted">{hint}</dd>}
    </div>
  );
}

/** A titled panel. Use for each distinct block of a page. */
export function Card({
  title,
  description,
  actions,
  children,
  className = "",
  id,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  id?: string;
}) {
  const headingId = useId();
  return (
    <section aria-labelledby={title ? headingId : undefined} id={id} className={`panel p-5 sm:p-6 ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="max-w-2xl">
            {title && (
              <h2 id={headingId} className="text-lg font-semibold">
                {title}
              </h2>
            )}
            {description && <div className="mt-1 text-sm leading-relaxed text-muted">{description}</div>}
          </div>
          {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** One headline metric. ``tone`` ties the number to the clean/attack colour language. */
export function KpiCard({
  label,
  value,
  hint,
  tone = "neutral",
}: {
  label: ReactNode;
  value: ReactNode;
  hint?: ReactNode;
  tone?: "neutral" | "clean" | "adv";
}) {
  const color = { neutral: "text-ink", clean: "text-clean", adv: "text-adv" }[tone];
  return (
    <div className="panel p-4">
      <p className="text-sm text-muted">{label}</p>
      <p className={`mt-1 font-mono text-3xl font-semibold tabular-nums ${color}`}>{value}</p>
      {hint && <p className="mt-1 text-xs leading-relaxed text-muted">{hint}</p>}
    </div>
  );
}

export interface SegmentOption<T extends string> {
  value: T;
  label: ReactNode;
  hint?: string;
}

/** A row of mutually exclusive choices (radio group styled as tabs). */
export function Segmented<T extends string>({
  legend,
  options,
  value,
  onChange,
  name,
  tone = "clean",
  columns,
}: {
  legend: ReactNode;
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  name: string;
  tone?: "clean" | "adv";
  columns?: number;
}) {
  const active = tone === "clean" ? "border-clean bg-clean-soft text-ink" : "border-adv bg-adv-soft text-ink";
  return (
    <fieldset>
      <legend className="field-label">{legend}</legend>
      <div
        className="mt-1.5 grid gap-1.5"
        style={{ gridTemplateColumns: `repeat(${columns ?? options.length}, minmax(0, 1fr))` }}
      >
        {options.map((option) => (
          <label
            key={option.value}
            className={`cursor-pointer rounded-lg border px-3 py-2 text-sm transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-clean ${
              value === option.value ? active : "border-line bg-paper text-muted hover:border-line-strong hover:text-ink"
            }`}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={value === option.value}
              onChange={() => onChange(option.value)}
              className="sr-only"
            />
            <span className="block font-semibold">{option.label}</span>
            {option.hint && <span className="mt-0.5 block text-xs text-muted">{option.hint}</span>}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

/** A labelled range input with a live value readout and an optional glossary tip. */
export function Slider({
  id,
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  display,
  hint,
  term,
  tone = "clean",
  disabled,
}: {
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  display?: ReactNode;
  hint?: ReactNode;
  term?: GlossaryKey;
  tone?: "clean" | "adv";
  disabled?: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="flex items-center gap-1.5">
          <label htmlFor={id} className="field-label">
            {label}
          </label>
          {term && <InfoTip term={term} />}
        </span>
        <output htmlFor={id} className="font-mono text-sm font-semibold tabular-nums">
          {display ?? value}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        className={`mt-1.5 w-full disabled:opacity-40 ${tone === "adv" ? "range-adv" : ""}`}
        aria-describedby={hint ? `${id}-hint` : undefined}
      />
      {hint && (
        <p id={`${id}-hint`} className="field-hint mt-1">
          {hint}
        </p>
      )}
    </div>
  );
}

/** A terminal command with a copy button. */
export function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }
  return (
    <div className="flex items-stretch overflow-hidden rounded-lg border border-line bg-paper text-left">
      <code className="flex-1 overflow-x-auto whitespace-pre px-3 py-2 font-mono text-xs leading-relaxed text-ink">{command}</code>
      <button
        type="button"
        onClick={copy}
        className="shrink-0 border-l border-line px-3 text-xs font-medium text-muted hover:bg-raised hover:text-ink"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}

/** Header for one step of a genuine sequence (for example: pick a model, then an image, then the attack). */
export function StepHeader({ step, title, children }: { step: number; title: string; children?: ReactNode }) {
  return (
    <div className="flex gap-3">
      <span
        aria-hidden="true"
        className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-line-strong font-mono text-xs font-semibold text-muted"
      >
        {step}
      </span>
      <div>
        <h3 className="text-base font-semibold">
          <span className="sr-only">Step {step}: </span>
          {title}
        </h3>
        {children && <p className="mt-0.5 text-xs leading-relaxed text-muted">{children}</p>}
      </div>
    </div>
  );
}

/** A small "?" button that explains a glossary term on hover, focus, or click. */
export function InfoTip({ term }: { term: GlossaryKey }) {
  const entry: GlossaryEntry = glossary[term];
  const [open, setOpen] = useState(false);
  const tipId = useId();
  return (
    <span className="relative inline-flex" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-label={`What is ${entry.term}?`}
        aria-describedby={open ? tipId : undefined}
        aria-expanded={open}
        onClick={() => setOpen((current) => !current)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === "Escape") setOpen(false);
        }}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-line-strong text-[10px] font-semibold leading-none text-muted hover:border-clean hover:text-clean"
      >
        ?
      </button>
      {open && (
        <span
          id={tipId}
          role="tooltip"
          className="absolute left-1/2 top-6 z-30 w-64 -translate-x-1/2 rounded-lg border border-line-strong bg-raised p-3 text-left text-xs font-normal leading-relaxed text-ink shadow-xl shadow-black/40"
        >
          <span className="mb-1 block font-semibold">
            {entry.term}
            {entry.abbr && <span className="font-normal text-muted"> ({entry.abbr})</span>}
          </span>
          {entry.short}
        </span>
      )}
    </span>
  );
}

/** Inline term followed by its glossary tip. */
export function Term({ id, children }: { id: GlossaryKey; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      {children}
      <InfoTip term={id} />
    </span>
  );
}
