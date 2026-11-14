import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";

import { toProblem, type ApiProblem } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Button, Notice } from "../components/ui";

type Mode = "login" | "register";

export function Login() {
  const { user, login, register } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? "/attack";

  const [mode, setMode] = useState<Mode>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [problem, setProblem] = useState<ApiProblem | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to={from} replace />;

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setProblem(null);
    try {
      await (mode === "login" ? login(email, password) : register(email, password));
      navigate(from, { replace: true });
    } catch (caught) {
      setProblem(toProblem(caught));
    } finally {
      setBusy(false);
    }
  }

  const fieldError = (field: string) => (problem?.field === field ? problem.message : null);

  return (
    <div className="mx-auto grid max-w-4xl items-start gap-10 py-6 md:grid-cols-[1fr_22rem]">
      <div className="order-2 md:order-1">
        <h2 className="text-xl font-semibold">Why an account?</h2>
        <ul className="mt-4 space-y-4 text-sm leading-relaxed text-muted">
          <li>
            <span className="font-medium text-ink">Your runs stay yours.</span> Training runs, evaluations and
            certifications are saved per account, so you can come back to them later.
          </li>
          <li>
            <span className="font-medium text-ink">The server does real work.</span> Attacks and training use the
            server's CPU or GPU. An account keeps that usage tied to a person.
          </li>
          <li>
            <span className="font-medium text-ink">Nothing else is needed.</span> Only an email and password, stored on
            this server. The Dashboard and Start here guide stay open without logging in.
          </li>
        </ul>
        <p className="mt-6 text-sm">
          <Link to="/guide" className="link">
            Read how the project works first
          </Link>
        </p>
      </div>

      <div className="order-1 md:order-2">
        <h1 className="text-2xl font-semibold">{mode === "login" ? "Log in" : "Create an account"}</h1>
        <p className="mt-1 text-sm text-muted">
          {mode === "login" ? "Welcome back. Continue to the attack lab." : "Takes ten seconds. No email confirmation."}
        </p>

        <form onSubmit={submit} className="panel mt-5 space-y-4 p-5" noValidate>
          <div>
            <label htmlFor="email" className="field-label">
              Email
            </label>
            <input
              id="email"
              type="email"
              autoComplete="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              aria-invalid={Boolean(fieldError("email"))}
              aria-describedby={fieldError("email") ? "email-error" : undefined}
              className="input mt-1"
            />
            {fieldError("email") && (
              <p id="email-error" className="mt-1 text-xs text-flag">
                {fieldError("email")}
              </p>
            )}
          </div>
          <div>
            <label htmlFor="password" className="field-label">
              Password
            </label>
            <input
              id="password"
              type="password"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
              required
              minLength={8}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              aria-invalid={Boolean(fieldError("password"))}
              aria-describedby="password-hint"
              className="input mt-1"
            />
            <p id="password-hint" className={`mt-1 text-xs ${fieldError("password") ? "text-flag" : "text-muted"}`}>
              {fieldError("password") ?? "At least 8 characters."}
            </p>
          </div>

          {problem && !problem.field && <Notice tone="error">{problem.message}</Notice>}

          <Button type="submit" busy={busy} className="w-full">
            {mode === "login" ? "Log in" : "Create account"}
          </Button>
        </form>

        <p className="mt-4 text-center text-sm text-muted">
          {mode === "login" ? "New here? " : "Already have an account? "}
          <button
            type="button"
            className="link"
            onClick={() => {
              setMode(mode === "login" ? "register" : "login");
              setProblem(null);
            }}
          >
            {mode === "login" ? "Create an account" : "Log in"}
          </button>
        </p>
      </div>
    </div>
  );
}
