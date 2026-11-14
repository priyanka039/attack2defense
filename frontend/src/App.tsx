import { lazy, Suspense, type ComponentType, type ReactNode } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { RequireAuth } from "./auth/RequireAuth";
import { Layout } from "./components/Layout";
import { Spinner } from "./components/ui";
import { guideFlag, SetupProvider } from "./lib/setup";
import { Login } from "./pages/Login";
import { NotFound } from "./pages/NotFound";

const named = <K extends string>(load: () => Promise<Record<K, ComponentType>>, name: K) =>
  lazy(() => load().then((module) => ({ default: module[name] })));

const Guide = named(() => import("./pages/Guide"), "Guide");
const Overview = named(() => import("./pages/Overview"), "Overview");
const ModelsData = named(() => import("./pages/ModelsData"), "ModelsData");
const AttackLab = named(() => import("./pages/AttackLab"), "AttackLab");
const Training = named(() => import("./pages/Training"), "Training");
const Evaluation = named(() => import("./pages/Evaluation"), "Evaluation");
const Certified = named(() => import("./pages/Certified"), "Certified");

function Page({ children, action }: { children: ReactNode; action?: string }) {
  const content = <Suspense fallback={<Spinner label="Loading" />}>{children}</Suspense>;
  return action ? <RequireAuth action={action}>{content}</RequireAuth> : content;
}

/** First-time visitors land on the guide; after that the dashboard is home. */
function Home() {
  if (!guideFlag.seen()) return <Navigate to="/guide" replace />;
  return (
    <Page>
      <Overview />
    </Page>
  );
}

export function App() {
  return (
    <SetupProvider>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="guide" element={<Page><Guide /></Page>} />
          <Route path="login" element={<Login />} />
          <Route path="models" element={<Page><ModelsData /></Page>} />
          <Route path="attack" element={<Page action="attack a model"><AttackLab /></Page>} />
          <Route path="training" element={<Page action="train a model"><Training /></Page>} />
          <Route path="evaluation" element={<Page action="run an evaluation"><Evaluation /></Page>} />
          <Route path="certified" element={<Page><Certified /></Page>} />
          <Route path="*" element={<NotFound />} />
        </Route>
      </Routes>
    </SetupProvider>
  );
}
