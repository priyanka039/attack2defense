import { useCallback, useEffect, useMemo, useState } from "react";

const SIZE = 8;
const WIN = 12;

function emptyGrid(): boolean[] {
  return Array.from({ length: SIZE * SIZE }, () => false);
}

/** Tiny click game for API cold start: paint an adversarial “patch” on a pixel grid. */
export function BootPlayground() {
  const [cells, setCells] = useState(emptyGrid);
  const [moves, setMoves] = useState(0);
  const [won, setWon] = useState(false);
  const [pulse, setPulse] = useState(0);

  const flipped = useMemo(() => cells.filter(Boolean).length, [cells]);

  useEffect(() => {
    const id = window.setInterval(() => setPulse((p) => p + 1), 900);
    return () => window.clearInterval(id);
  }, []);

  useEffect(() => {
    if (!won && flipped >= WIN) setWon(true);
  }, [flipped, won]);

  const toggle = useCallback((index: number) => {
    setCells((current) => {
      const next = [...current];
      next[index] = !next[index];
      return next;
    });
    setMoves((m) => m + 1);
  }, []);

  const reset = () => {
    setCells(emptyGrid());
    setMoves(0);
    setWon(false);
  };

  return (
    <div className="w-full max-w-sm">
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <p className="font-display text-base font-semibold text-ink">Paint the attack</p>
          <p className="mt-0.5 text-xs leading-relaxed text-muted">
            Click pixels to build a tiny adversarial patch while the API loads. Goal: {WIN} pink cells.
          </p>
        </div>
        <button
          type="button"
          onClick={reset}
          className="shrink-0 rounded-md border border-line px-2 py-1 text-[11px] font-medium text-muted hover:border-line-strong hover:text-ink"
        >
          Reset
        </button>
      </div>

      <div
        className="grid aspect-square w-full gap-1 rounded-xl border border-line bg-paper p-2"
        style={{ gridTemplateColumns: `repeat(${SIZE}, minmax(0, 1fr))` }}
        role="group"
        aria-label="Adversarial pixel patch game"
      >
        {cells.map((on, index) => {
          const idleGlow = !on && (index + pulse) % 7 === 0;
          return (
            <button
              key={index}
              type="button"
              aria-pressed={on}
              aria-label={on ? `Pixel ${index + 1} attacked` : `Pixel ${index + 1} clean`}
              onClick={() => toggle(index)}
              className={`rounded-sm transition-colors duration-150 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-clean ${
                on
                  ? "bg-adv shadow-[inset_0_0_0_1px_rgba(196,138,154,0.5)]"
                  : idleGlow
                    ? "bg-raised animate-pulse"
                    : "bg-surface hover:bg-raised"
              }`}
            />
          );
        })}
      </div>

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs">
        <p className="font-mono tabular-nums text-muted">
          <span className="text-adv">{flipped}</span>/{WIN} attacked · {moves} clicks
        </p>
        {won ? (
          <p className="font-medium text-clean">Patch ready — model still waking…</p>
        ) : (
          <p className="text-muted">Teal will return when the server answers.</p>
        )}
      </div>
    </div>
  );
}
