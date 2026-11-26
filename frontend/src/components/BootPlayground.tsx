import { useCallback, useEffect, useRef, useState } from "react";

type Phase = "ready" | "playing" | "over";

interface Blob {
  x: number;
  y: number;
  r: number;
  vy: number;
  id: number;
}

const W = 320;
const H = 220;
const SHIELD_W = 56;
const SHIELD_H = 10;
const MAX_LIVES = 3;

/** Real mini-game while waiting: block falling adversarial noise with a moving shield. */
export function BootPlayground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef({
    shieldX: W / 2,
    blobs: [] as Blob[],
    score: 0,
    lives: MAX_LIVES,
    phase: "ready" as Phase,
    spawnIn: 0,
    nextId: 1,
    pointer: false,
  });
  const [hud, setHud] = useState({ score: 0, lives: MAX_LIVES, phase: "ready" as Phase, best: 0 });

  const bestKey = "a2d.defendBest";

  useEffect(() => {
    const saved = Number(localStorage.getItem(bestKey) || 0);
    setHud((h) => ({ ...h, best: Number.isFinite(saved) ? saved : 0 }));
  }, []);

  const start = useCallback(() => {
    const s = stateRef.current;
    s.shieldX = W / 2;
    s.blobs = [];
    s.score = 0;
    s.lives = MAX_LIVES;
    s.phase = "playing";
    s.spawnIn = 40;
    setHud((h) => ({ ...h, score: 0, lives: MAX_LIVES, phase: "playing" }));
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame = 0;
    let raf = 0;

    const onKey = (event: KeyboardEvent) => {
      const s = stateRef.current;
      if (event.key === "ArrowLeft" || event.key === "a" || event.key === "A") {
        s.shieldX = Math.max(SHIELD_W / 2, s.shieldX - 18);
        event.preventDefault();
      }
      if (event.key === "ArrowRight" || event.key === "d" || event.key === "D") {
        s.shieldX = Math.min(W - SHIELD_W / 2, s.shieldX + 18);
        event.preventDefault();
      }
      if ((event.key === " " || event.key === "Enter") && s.phase !== "playing") {
        start();
        event.preventDefault();
      }
    };

    const moveTo = (clientX: number) => {
      const rect = canvas.getBoundingClientRect();
      const x = ((clientX - rect.left) / rect.width) * W;
      stateRef.current.shieldX = Math.min(W - SHIELD_W / 2, Math.max(SHIELD_W / 2, x));
    };

    const onPointerDown = (event: PointerEvent) => {
      stateRef.current.pointer = true;
      canvas.setPointerCapture(event.pointerId);
      moveTo(event.clientX);
      if (stateRef.current.phase !== "playing") start();
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!stateRef.current.pointer && event.buttons === 0) return;
      moveTo(event.clientX);
    };
    const onPointerUp = () => {
      stateRef.current.pointer = false;
    };

    window.addEventListener("keydown", onKey);
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);

    const tick = () => {
      frame += 1;
      const s = stateRef.current;

      // Clear
      ctx.fillStyle = "#0f1319";
      ctx.fillRect(0, 0, W, H);
      // Grid feel
      ctx.strokeStyle = "rgba(43,52,65,0.9)";
      ctx.lineWidth = 1;
      for (let x = 0; x < W; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }

      if (s.phase === "playing") {
        s.spawnIn -= 1;
        if (s.spawnIn <= 0) {
          const speed = 1.4 + Math.min(2.8, s.score * 0.04);
          s.blobs.push({
            id: s.nextId++,
            x: 16 + Math.random() * (W - 32),
            y: -12,
            r: 7 + Math.random() * 5,
            vy: speed,
          });
          s.spawnIn = Math.max(18, 48 - Math.floor(s.score / 3));
        }

        const next: Blob[] = [];
        let score = s.score;
        let lives = s.lives;
        for (const blob of s.blobs) {
          blob.y += blob.vy;
          const shieldTop = H - 28;
          const shieldLeft = s.shieldX - SHIELD_W / 2;
          const shieldRight = s.shieldX + SHIELD_W / 2;
          const hit =
            blob.y + blob.r >= shieldTop &&
            blob.y - blob.r <= shieldTop + SHIELD_H &&
            blob.x >= shieldLeft &&
            blob.x <= shieldRight;
          if (hit) {
            score += 1;
            continue;
          }
          if (blob.y - blob.r > H) {
            lives -= 1;
            continue;
          }
          next.push(blob);
        }
        s.blobs = next;
        s.score = score;
        s.lives = lives;
        if (lives <= 0) {
          s.phase = "over";
          const best = Math.max(Number(localStorage.getItem(bestKey) || 0), score);
          localStorage.setItem(bestKey, String(best));
          setHud({ score, lives: 0, phase: "over", best });
        } else if (frame % 8 === 0) {
          setHud((h) => ({ ...h, score, lives, phase: "playing" }));
        }
      }

      // Blobs (attacks)
      for (const blob of s.blobs) {
        const g = ctx.createRadialGradient(blob.x, blob.y, 1, blob.x, blob.y, blob.r);
        g.addColorStop(0, "#e8b4c0");
        g.addColorStop(1, "#c48a9a");
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(blob.x, blob.y, blob.r, 0, Math.PI * 2);
        ctx.fill();
      }

      // Shield (defense)
      const sx = s.shieldX - SHIELD_W / 2;
      const sy = H - 28;
      ctx.fillStyle = "#7eb6e0";
      ctx.fillRect(sx, sy, SHIELD_W, SHIELD_H);
      ctx.fillStyle = "rgba(126,182,224,0.25)";
      ctx.fillRect(sx - 4, sy - 3, SHIELD_W + 8, SHIELD_H + 6);

      // Classifier “core”
      ctx.fillStyle = "#1f2630";
      ctx.fillRect(W / 2 - 18, H - 14, 36, 8);
      ctx.fillStyle = "#8b97a8";
      ctx.font = "10px IBM Plex Sans, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("model", W / 2, H - 5);

      if (s.phase === "ready") {
        ctx.fillStyle = "rgba(15,19,25,0.72)";
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = "#e7edf4";
        ctx.font = "600 15px Archivo, IBM Plex Sans, sans-serif";
        ctx.fillText("Shield the classifier", W / 2, H / 2 - 18);
        ctx.fillStyle = "#8b97a8";
        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.fillText("Stop pink attacks before they hit the model", W / 2, H / 2 + 4);
        ctx.fillStyle = "#ff5c1a";
        ctx.font = "600 12px IBM Plex Sans, sans-serif";
        ctx.fillText("Click / tap or press Space to play", W / 2, H / 2 + 28);
      }

      if (s.phase === "over") {
        ctx.fillStyle = "rgba(15,19,25,0.75)";
        ctx.fillRect(0, 0, W, H);
        ctx.fillStyle = "#e7edf4";
        ctx.font = "600 15px Archivo, IBM Plex Sans, sans-serif";
        ctx.fillText("Model fooled!", W / 2, H / 2 - 12);
        ctx.fillStyle = "#c48a9a";
        ctx.font = "12px IBM Plex Sans, sans-serif";
        ctx.fillText(`Blocked ${s.score} attacks`, W / 2, H / 2 + 10);
        ctx.fillStyle = "#ff5c1a";
        ctx.font = "600 12px IBM Plex Sans, sans-serif";
        ctx.fillText("Click or Space to try again", W / 2, H / 2 + 32);
      }

      raf = window.requestAnimationFrame(tick);
    };

    raf = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("keydown", onKey);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
    };
  }, [start]);

  return (
    <div className="w-full max-w-sm">
      <div className="mb-3 flex items-end justify-between gap-3">
        <div>
          <p className="font-display text-base font-semibold text-ink">Shield the classifier</p>
          <p className="mt-0.5 text-xs leading-relaxed text-muted">
            Drag or use ← → to block pink attacks. Miss three and the model is fooled.
          </p>
        </div>
        <button
          type="button"
          onClick={start}
          className="shrink-0 rounded-md border border-line px-2 py-1 text-[11px] font-medium text-muted hover:border-line-strong hover:text-ink"
        >
          {hud.phase === "playing" ? "Restart" : "Play"}
        </button>
      </div>

      <canvas
        ref={canvasRef}
        width={W}
        height={H}
        className="w-full touch-none rounded-xl border border-line bg-paper"
        role="img"
        aria-label="Shield the classifier mini-game. Use arrow keys or drag to move the shield."
      />

      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs">
        <p className="font-mono tabular-nums text-muted">
          Score <span className="text-clean">{hud.score}</span>
          {" · "}
          Lives <span className="text-adv">{hud.lives}</span>
          {" · "}
          Best <span className="text-ink">{hud.best}</span>
        </p>
        <p className="text-muted">{hud.phase === "playing" ? "Defend!" : "Pass the time while the server works"}</p>
      </div>
    </div>
  );
}
