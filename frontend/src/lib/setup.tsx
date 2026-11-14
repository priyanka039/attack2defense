import { createContext, useContext, type ReactNode } from "react";

import { api } from "../api/endpoints";
import type { SetupReport } from "../api/types";
import { useResource, type Resource } from "./hooks";

const SetupContext = createContext<Resource<SetupReport> | null>(null);

/** Loads ``GET /api/setup`` once for the sidebar status dot and the Start here checklist. */
export function SetupProvider({ children }: { children: ReactNode }) {
  const setup = useResource(api.setup);
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
