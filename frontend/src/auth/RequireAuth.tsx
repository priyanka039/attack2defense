import type { ReactNode } from "react";
import { Link, useLocation } from "react-router-dom";

import { EmptyState, Spinner } from "../components/ui";
import { useAuth } from "./AuthContext";

/** Pages that start runs need an account; visitors get a pointer to the public results. */
export function RequireAuth({ action, children }: { action: string; children: ReactNode }) {
  const { user, ready } = useAuth();
  const location = useLocation();

  if (!ready) return <Spinner label="Checking your session" />;
  if (!user) {
    return (
      <EmptyState title={`Log in to ${action}`}>
        <p>
          Runs are kept per account.{" "}
          <Link to="/login" state={{ from: location.pathname }} className="font-medium text-clean underline">
            Log in or create an account
          </Link>
          , or browse the{" "}
          <Link to="/" className="font-medium text-clean underline">
            published results
          </Link>{" "}
          without one.
        </p>
      </EmptyState>
    );
  }
  return <>{children}</>;
}
