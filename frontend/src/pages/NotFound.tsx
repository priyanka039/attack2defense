import { Link } from "react-router-dom";

import { buttonClass, EmptyState } from "../components/ui";

export function NotFound() {
  return (
    <EmptyState
      title="This page does not exist"
      action={
        <>
          <Link to="/guide" className={buttonClass("primary")}>
            Go to Start here
          </Link>
          <Link to="/" className={buttonClass("secondary")}>
            Open the dashboard
          </Link>
        </>
      }
    >
      The link may be old, or the address may have a typo.
    </EmptyState>
  );
}
