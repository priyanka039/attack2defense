import { useEffect, useState } from "react";

import { percent } from "../lib/format";

export interface TrackRow {
  label: string;
  detail: string;
  value: number;
  tone: "clean" | "adv";
}

/**
 * The project's story on one shared 0-100% axis: accuracy before the attack,
 * after the attack, and after the defense, with the target drawn across all rows.
 */
export function AccuracyTrack({ rows, target }: { rows: TrackRow[]; target?: number }) {
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const id = window.requestAnimationFrame(() => setShown(true));
    return () => window.cancelAnimationFrame(id);
  }, []);

  return (
    <figure className="relative">
      <ul className="space-y-4">
        {rows.map((row) => (
          <li key={row.label}>
            <div className="mb-1.5 flex items-baseline justify-between gap-3">
              <span className="text-sm">
                <span className="font-semibold">{row.label}</span>
                <span className="text-muted"> {row.detail}</span>
              </span>
              <span className={`font-mono text-2xl font-semibold tabular-nums ${row.tone === "clean" ? "text-clean" : "text-adv"}`}>
                {percent(row.value)}
              </span>
            </div>
            <div className="relative h-3 overflow-hidden rounded-full bg-raised" aria-hidden="true">
              <div
                className={`h-full rounded-full transition-[width] duration-1000 ease-out motion-reduce:transition-none ${row.tone === "clean" ? "bg-clean" : "bg-adv"}`}
                style={{ width: shown ? `${row.value * 100}%` : "0%" }}
              />
              {target !== undefined && (
                <span className="absolute inset-y-0 w-0.5 bg-warn" style={{ left: `${target * 100}%` }} />
              )}
            </div>
          </li>
        ))}
      </ul>
      {target !== undefined && (
        <figcaption className="mt-3 flex items-center gap-2 text-xs text-muted">
          <span aria-hidden="true" className="inline-block h-3 w-0.5 bg-warn" />
          Amber mark: the {percent(target, 0)} robust-accuracy target.
        </figcaption>
      )}
    </figure>
  );
}
