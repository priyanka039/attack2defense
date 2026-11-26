import { useEffect, useState } from "react";
import { Link, NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";
import { useSetup } from "../lib/setup";
import { useTheme } from "../lib/theme";
import type { Theme } from "../lib/tokens";
import { HostWakeBanner } from "./WaitPanel";

interface NavItem {
  to: string;
  label: string;
  hint: string;
  end?: boolean;
  needsLogin?: boolean;
}

const groups: { title: string; items: NavItem[] }[] = [
  {
    title: "Learn",
    items: [
      { to: "/guide", label: "Start here", hint: "What this project does" },
      { to: "/", label: "Dashboard", hint: "The reported results", end: true },
    ],
  },
  {
    title: "Experiment",
    items: [
      { to: "/attack", label: "Attack lab", hint: "Fool a model, step by step", needsLogin: true },
      { to: "/training", label: "Training", hint: "Train a defended model", needsLogin: true },
      { to: "/evaluation", label: "Evaluation", hint: "Compare models fairly", needsLogin: true },
      { to: "/certified", label: "Certified defense", hint: "Provable robustness" },
    ],
  },
  {
    title: "Data",
    items: [{ to: "/models", label: "Models and data", hint: "Checkpoints and test images" }],
  },
];

function Brand() {
  return (
    <Link to="/guide" className="flex items-center gap-2.5" aria-label="Attack2Defense, start here">
      <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden="true">
        <rect width="32" height="32" rx="7" className="fill-raised" />
        <rect x="7" y="7" width="18" height="18" rx="2" fill="none" strokeWidth="2.5" className="stroke-clean" />
        <rect x="12" y="12" width="8" height="8" rx="1" className="fill-accent" />
      </svg>
      <span className="font-display text-lg font-semibold tracking-tight">Attack2Defense</span>
    </Link>
  );
}

function SetupDot() {
  const setup = useSetup();
  const report = setup.data;
  const state = setup.error ? "offline" : !report ? "checking" : !report.ready ? "blocked" : report.missing_reported_weights.length ? "partial" : "ready";
  const text = {
    offline: "API offline",
    checking: "Checking setup",
    blocked: "Setup needed",
    partial: "Demo models only",
    ready: "All models live",
  }[state];
  const color = {
    offline: "bg-flag",
    checking: "bg-muted",
    blocked: "bg-flag",
    partial: "bg-warn",
    ready: "bg-clean",
  }[state];
  return (
    <Link to="/guide#setup" className="flex items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted hover:bg-raised hover:text-ink">
      <span aria-hidden="true" className={`h-2 w-2 rounded-full ${color}`} />
      <span>{text}</span>
      {report && <span className="ml-auto font-mono text-[11px] uppercase">{report.device.split(" ")[0]}</span>}
    </Link>
  );
}

function ThemeIcon({ theme }: { theme: Theme }) {
  return theme === "light" ? (
    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <circle cx="10" cy="10" r="3.5" />
      <path d="M10 1.5v2M10 16.5v2M1.5 10h2M16.5 10h2M4 4l1.4 1.4M14.6 14.6L16 16M4 16l1.4-1.4M14.6 5.4L16 4" strokeLinecap="round" />
    </svg>
  ) : (
    <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
      <path d="M16.5 12.2A7 7 0 0 1 7.8 3.5a7 7 0 1 0 8.7 8.7Z" strokeLinejoin="round" />
    </svg>
  );
}

function ThemeSwitch() {
  const { theme, setTheme } = useTheme();
  const options: { value: Theme; label: string }[] = [
    { value: "dark", label: "Dark" },
    { value: "light", label: "Light" },
  ];
  return (
    <div role="group" aria-label="Colour theme" className="grid grid-cols-2 gap-1 rounded-lg border border-line bg-paper p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={theme === option.value}
          onClick={() => setTheme(option.value)}
          className={`flex items-center justify-center gap-1.5 rounded-md px-2 py-1 text-xs font-medium transition-colors ${
            theme === option.value ? "bg-raised text-ink" : "text-muted hover:text-ink"
          }`}
        >
          <ThemeIcon theme={option.value} />
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ThemeButton() {
  const { theme, toggle } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      className="rounded-lg border border-line p-2 text-muted hover:text-ink"
    >
      <ThemeIcon theme={next} />
    </button>
  );
}

export function Layout() {
  const { user, ready, logout } = useAuth();
  const setup = useSetup();
  const navigate = useNavigate();
  const location = useLocation();
  const [open, setOpen] = useState(false);

  useEffect(() => setOpen(false), [location.pathname]);

  const signOut = () => {
    logout();
    navigate("/guide");
  };

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `group flex items-start justify-between gap-2 rounded-lg px-3 py-2 transition-colors ${
      isActive ? "bg-raised text-ink" : "text-muted hover:bg-raised/60 hover:text-ink"
    }`;

  const sidebar = (
    <>
      <nav aria-label="Main" className="flex-1 space-y-6 overflow-y-auto px-3 py-2">
        {groups.map((group) => (
          <div key={group.title}>
            <p className="px-3 pb-1.5 text-xs font-medium text-muted/80">{group.title}</p>
            <ul className="space-y-0.5">
              {group.items.map((item) => (
                <li key={item.to}>
                  <NavLink to={item.to} end={item.end} className={linkClass}>
                    {({ isActive }) => (
                      <>
                        <span className="min-w-0">
                          <span className={`block text-sm font-medium ${isActive ? "text-ink" : ""}`}>
                            {isActive && <span aria-hidden="true" className="mr-2 inline-block h-1.5 w-1.5 rounded-full bg-clean align-middle" />}
                            {item.label}
                          </span>
                          <span className="block text-xs text-muted">{item.hint}</span>
                        </span>
                        {item.needsLogin && ready && !user && (
                          <span className="mt-0.5 shrink-0 rounded border border-line px-1.5 text-[10px] text-muted">login</span>
                        )}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>
      <div className="space-y-2 border-t border-line px-3 py-4">
        <ThemeSwitch />
        <SetupDot />
        <div className="min-h-[2.75rem] px-2 text-sm">
          {ready &&
            (user ? (
              <>
                <p className="truncate text-muted" title={user.email}>
                  Signed in as <span className="text-ink">{user.email}</span>
                </p>
                <button type="button" className="mt-1 text-sm font-medium text-clean hover:underline" onClick={signOut}>
                  Log out
                </button>
              </>
            ) : (
              <Link to="/login" state={{ from: location.pathname }} className="link text-sm">
                Log in to run experiments
              </Link>
            ))}
        </div>
      </div>
    </>
  );

  return (
    <>
      <HostWakeBanner offline={Boolean(setup.error)} onRetry={() => void setup.reload()} retrying={setup.loading} />
      <div className="min-h-screen lg:grid lg:grid-cols-[16.5rem_1fr]">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-clean focus:px-3 focus:py-2 focus:text-paper">
        Skip to content
      </a>

      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-line bg-paper/90 px-4 py-3 backdrop-blur lg:hidden">
        <Brand />
        <div className="flex items-center gap-2">
          <ThemeButton />
          <button
            type="button"
            aria-expanded={open}
            aria-controls="mobile-nav"
            onClick={() => setOpen((current) => !current)}
            className="rounded-lg border border-line px-3 py-1.5 text-sm font-medium"
          >
            {open ? "Close" : "Menu"}
          </button>
        </div>
      </header>

      {open && (
        <div id="mobile-nav" className="fixed inset-x-0 bottom-0 top-[57px] z-20 flex flex-col bg-surface lg:hidden">
          {sidebar}
        </div>
      )}

      <aside className="hidden border-r border-line bg-surface lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col">
        <div className="px-6 py-6">
          <Brand />
        </div>
        {sidebar}
      </aside>

      <main id="main" className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:px-10 lg:py-12">
        <Outlet />
      </main>
    </div>
    </>
  );
}
