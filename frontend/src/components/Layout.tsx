import { NavLink, Outlet, useNavigate } from "react-router-dom";

import { useAuth } from "../auth/AuthContext";

const links = [
  { to: "/", label: "Overview", end: true },
  { to: "/models", label: "Models and data" },
  { to: "/attack", label: "Attack lab" },
  { to: "/training", label: "Training" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/certified", label: "Certified defense" },
];

function Brand() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5" aria-label="Attack2Defense home">
      <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden="true">
        <rect width="32" height="32" rx="6" className="fill-ink" />
        <rect x="7" y="7" width="18" height="18" fill="none" strokeWidth="2.5" className="stroke-clean" />
        <rect x="11" y="11" width="10" height="10" className="fill-adv" />
      </svg>
      <span className="font-display text-lg font-semibold">Attack2Defense</span>
    </NavLink>
  );
}

export function Layout() {
  const { user, ready, logout } = useAuth();
  const navigate = useNavigate();
  const signOut = () => {
    logout();
    navigate("/");
  };

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `block whitespace-nowrap rounded-md px-3 py-2 text-sm transition-colors ${
      isActive ? "bg-ink text-white" : "text-ink hover:bg-ink/5"
    }`;

  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[15rem_1fr]">
      <aside className="border-b border-line bg-surface lg:sticky lg:top-0 lg:flex lg:h-screen lg:flex-col lg:border-b-0 lg:border-r">
        <div className="flex items-center justify-between px-4 py-4 lg:px-5 lg:py-6">
          <Brand />
        </div>
        <nav aria-label="Main" className="overflow-x-auto px-2 pb-3 lg:flex-1 lg:px-3">
          <ul className="flex gap-1 lg:flex-col">
            {links.map((link) => (
              <li key={link.to}>
                <NavLink to={link.to} end={link.end} className={linkClass}>
                  {link.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="hidden min-h-[4.5rem] border-t border-line px-5 py-4 text-sm lg:block">
          {ready &&
            (user ? (
              <>
                <p className="truncate text-muted" title={user.email}>
                  {user.email}
                </p>
                <button
                  type="button"
                  className="mt-1 font-medium underline-offset-2 hover:underline"
                  onClick={signOut}
                >
                  Log out
                </button>
              </>
            ) : (
              <NavLink to="/login" className="font-medium underline-offset-2 hover:underline">
                Log in to run experiments
              </NavLink>
            ))}
        </div>
      </aside>
      <main id="main" className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
        <div className="mb-6 flex min-h-5 justify-end text-sm lg:hidden">
          {ready &&
            (user ? (
              <button type="button" className="font-medium underline" onClick={signOut}>
                Log out ({user.email})
              </button>
            ) : (
              <NavLink to="/login" className="font-medium underline">
                Log in
              </NavLink>
            ))}
        </div>
        <Outlet />
      </main>
    </div>
  );
}
