import { useCallback, useEffect, useRef, useState } from "react";

import { toProblem, type ApiProblem } from "../api/client";

export interface Resource<T> {
  data: T | undefined;
  error: ApiProblem | null;
  loading: boolean;
  reload: () => Promise<void>;
}

/** Load data once (and on ``reload``), tracking loading and error state. */
export function useResource<T>(load: () => Promise<T>, deps: unknown[] = []): Resource<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiProblem | null>(null);
  const [loading, setLoading] = useState(true);
  const loader = useRef(load);
  loader.current = load;

  const reload = useCallback(async () => {
    setLoading(true);
    try {
      setData(await loader.current());
      setError(null);
    } catch (caught) {
      setError(toProblem(caught));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, deps);

  return { data, error, loading, reload };
}

/** Call ``tick`` every ``intervalMs`` while ``active`` is true. */
export function usePolling(tick: () => void, active: boolean, intervalMs = 1500): void {
  const callback = useRef(tick);
  callback.current = tick;

  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => callback.current(), intervalMs);
    return () => window.clearInterval(id);
  }, [active, intervalMs]);
}

export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}
