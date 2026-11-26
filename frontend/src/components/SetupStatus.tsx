import { useSetup } from "../lib/setup";
import { Button, Notice, Spinner } from "./ui";

/** Visitor-friendly setup summary; developer details stay collapsed. */
export function SetupStatus() {
  const setup = useSetup();

  if (setup.loading && !setup.data) return <Spinner label="Checking the server" />;
  if (setup.error) {
    const local =
      typeof window !== "undefined" &&
      (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
    return (
      <Notice tone="error" title="Cannot reach the server">
        {local ? (
          <>
            Start it with{" "}
            <code className="font-mono">
              python -m uvicorn backend.app.main:create_app --factory --port 8000
            </code>
            , then reload this page.
          </>
        ) : (
          <>
            The model server is still starting. Wait a moment, then click Check again — or play the mini-game
            above.
          </>
        )}
      </Notice>
    );
  }
  const report = setup.data!;
  const liveOk = report.checks.some((check) => check.id === "weights" && check.ok);
  const resultsOk = report.checks.some((check) => check.id === "results" && check.ok);
  const demoReady = liveOk && resultsOk;

  return (
    <div className="space-y-4">
      <Notice tone={demoReady ? "success" : "warn"} title={demoReady ? "Demo ready" : "Still preparing"}>
        {demoReady ? (
          <p>
            You can use the Attack lab with the Small CNN models, and the Dashboard already shows the reported
            ResNet-18 numbers. Log in to run attacks and training.
          </p>
        ) : (
          <p>Published results or demo weights are still loading. Try Check again in a moment.</p>
        )}
      </Notice>

      <details className="rounded-lg border border-line bg-raised/30 px-4 py-3">
        <summary className="cursor-pointer text-sm font-medium text-ink">Advanced setup (developers)</summary>
        <ul className="mt-3 divide-y divide-line rounded-lg border border-line bg-surface">
          {report.checks.map((check) => (
            <li key={check.id} className="flex gap-3 px-4 py-3">
              <span
                aria-hidden="true"
                className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${
                  check.ok ? "bg-clean" : check.id === "reported_weights" ? "bg-warn" : "bg-flag"
                }`}
              />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-medium">
                  {check.label}
                  <span className={`ml-2 text-xs font-normal ${check.ok ? "text-clean" : "text-warn"}`}>
                    {check.ok ? "Ready" : "Optional / pending"}
                  </span>
                </p>
                {check.hint && <p className="mt-0.5 text-xs leading-relaxed text-muted">{check.hint}</p>}
              </div>
            </li>
          ))}
        </ul>
        {report.missing_reported_weights.length > 0 && (
          <p className="mt-3 text-xs leading-relaxed text-muted">
            ResNet-18 <code className="font-mono text-ink">best.pt</code> files are optional for the public demo.
            Without them, live ResNet attacks are unavailable, but Small CNN demos and the Dashboard still work.
          </p>
        )}
      </details>

      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>
          Server device: <span className="font-mono text-ink">{report.device}</span>
        </span>
        <Button variant="ghost" size="sm" onClick={() => void setup.reload()} busy={setup.loading}>
          Check again
        </Button>
      </div>
    </div>
  );
}
