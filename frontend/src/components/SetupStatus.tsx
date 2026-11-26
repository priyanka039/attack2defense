import { useSetup } from "../lib/setup";
import { Button, CopyCommand, Notice, Spinner } from "./ui";

/** Checklist of what is installed, with the exact fix for anything missing. */
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
            The hosted demo may be waking up or temporarily down. Wait about a minute, then click Check
            again.
          </>
        )}
      </Notice>
    );
  }
  const report = setup.data!;

  return (
    <div className="space-y-4">
      <ul className="divide-y divide-line rounded-lg border border-line">
        {report.checks.map((check) => (
          <li key={check.id} className="flex gap-3 px-4 py-3">
            <span
              aria-hidden="true"
              className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${check.ok ? "bg-clean" : check.id === "reported_weights" ? "bg-warn" : "bg-flag"}`}
            />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium">
                {check.label}
                <span className={`ml-2 text-xs font-normal ${check.ok ? "text-clean" : "text-warn"}`}>
                  {check.ok ? "Ready" : "Missing"}
                </span>
              </p>
              {check.hint && <p className="mt-0.5 text-xs leading-relaxed text-muted">{check.hint}</p>}
            </div>
          </li>
        ))}
      </ul>

      {report.missing_reported_weights.length > 0 && (
        <div className="space-y-2">
          <p className="text-sm text-muted">
            Reported numbers still show without these files, but the ResNet-18 models cannot be attacked live
            until each <code className="font-mono text-ink">best.pt</code> is in place:
          </p>
          <ul className="space-y-1 font-mono text-xs text-ink">
            {report.missing_reported_weights.map((item) => (
              <li key={item.id}>{item.expected_path}</li>
            ))}
          </ul>
          <CopyCommand command="python scripts/install_checkpoints.py <your Drive download folder or .zip>" />
        </div>
      )}

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
