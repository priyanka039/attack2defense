import { Link } from "react-router-dom";

import { EmptyState } from "../components/ui";

export function NotFound() {
  return (
    <EmptyState title="This page does not exist">
      <Link to="/" className="font-medium text-clean underline">
        Go to the overview
      </Link>
    </EmptyState>
  );
}
