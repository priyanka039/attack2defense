import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import { buttonClass, EmptyState, Spinner } from "../components/ui";
import { useAuth } from "./AuthContext";

/** Pages that start runs need an account; visitors get a pointer to the public results. */
export function RequireAuth({ action, children }: { action: string; children: ReactNode }) {
  const { user, ready } = useAuth();
  const location = useLocation();

  if (!ready) return <Spinner label="Checking your session" />;
  if (!user) {
    return (
      <EmptyState
        title={`Log in to ${action}`}
        action={
          <>
            <Link to="/login" state={{ from: location.pathname }} className={buttonClass("primary")}>
              Log in or create an account
            </Link>
            <Link to="/" className={buttonClass("secondary")}>
              See the reported results
            </Link>
          </>
        }
      >
        Experiments run on the server and are saved to your account, so each person sees only their own runs. The
        reported results and the Start here guide need no account.
      </EmptyState>
    );
  }
  return <>{children}</>;
}
