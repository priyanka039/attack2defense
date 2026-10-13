import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

type Tone = "info" | "error" | "success";

interface ToastMessage {
  id: number;
  tone: Tone;
  text: string;
}

interface ToastApi {
  notify: (text: string, tone?: Tone) => void;
}

const ToastContext = createContext<ToastApi | null>(null);

const toneClasses: Record<Tone, string> = {
  info: "border-line bg-surface text-ink",
  error: "border-flag/40 bg-flag-soft text-flag",
  success: "border-clean/40 bg-clean-soft text-clean",
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [messages, setMessages] = useState<ToastMessage[]>([]);

  const dismiss = useCallback((id: number) => {
    setMessages((current) => current.filter((message) => message.id !== id));
  }, []);

  const notify = useCallback(
    (text: string, tone: Tone = "info") => {
      const id = Date.now() + Math.random();
      setMessages((current) => [...current.slice(-3), { id, tone, text }]);
      window.setTimeout(() => dismiss(id), tone === "error" ? 8000 : 4000);
    },
    [dismiss],
  );

  const value = useMemo(() => ({ notify }), [notify]);

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4" aria-live="polite">
        {messages.map((message) => (
          <div
            key={message.id}
            role={message.tone === "error" ? "alert" : "status"}
            className={`pointer-events-auto flex w-full max-w-md items-start gap-3 rounded-md border px-4 py-3 text-sm shadow-sm ${toneClasses[message.tone]}`}
          >
            <p className="flex-1">{message.text}</p>
            <button
              type="button"
              onClick={() => dismiss(message.id)}
              className="text-xs font-medium underline-offset-2 hover:underline"
            >
              Dismiss
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export function useToast(): ToastApi {
  const context = useContext(ToastContext);
  if (!context) throw new Error("useToast must be used inside ToastProvider");
  return context;
}
