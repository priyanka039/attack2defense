import { createContext, useContext, useEffect, type ReactNode } from "react";

import { api } from "../api/endpoints";
import type { SetupReport } from "../api/types";
import { useResource, type Resource } from "./hooks";

const SetupContext = createContext<Resource<SetupReport> | null>(null);

/** Loads ``GET /api/setup`` for the sidebar status dot and the Start here checklist. */
export function SetupProvider({ children }: { children: ReactNode }) {
  const setup = useResource(api.setup);

  // Free hosts sleep when idle; keep probing so the wake banner can clear without a full reload.
  useEffect(() => {
    if (!setup.error || setup.loading) return;
    const id = window.setTimeout(() => void setup.reload(), 8000);
    return () => window.clearTimeout(id);
  }, [setup.error, setup.loading, setup.reload]);

  return <SetupContext.Provider value={setup}>{children}</SetupContext.Provider>;
}

export function useSetup(): Resource<SetupReport> {
  const context = useContext(SetupContext);
  if (!context) throw new Error("useSetup must be used inside SetupProvider");
  return context;
}

const SEEN_GUIDE = "a2d.seenGuide";

export const guideFlag = {
  seen: () => localStorage.getItem(SEEN_GUIDE) === "1",
  markSeen: () => localStorage.setItem(SEEN_GUIDE, "1"),
};
