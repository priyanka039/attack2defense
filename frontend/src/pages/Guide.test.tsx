import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/endpoints";
import type { ModelEntry, PublicResults, SetupReport } from "../api/types";
import { App } from "../App";
import { AuthProvider } from "../auth/AuthContext";
import { ToastProvider } from "../components/Toast";
import { guideFlag } from "../lib/setup";

vi.mock("../api/endpoints", () => ({
  api: {
    me: vi.fn(),
    publicResults: vi.fn(),
    setup: vi.fn(),
    examplesUrl: (id: string) => `/api/models/${id}/examples.png`,
  },
}));

function model(id: string, mode: string, clean: number, pgd: number): ModelEntry {
  return {
    id,
    architecture: "resnet18",
    dataset: "cifar10",
    mode,
    label: mode === "clean" ? "No defense" : "PGD adversarial training",
    tag: mode,
    epsilon: 2 / 255,
    epochs: 30,
    has_weights: false,
    sweep: [],
    summary: {
      model: "resnet18",
      mode,
      epsilon: 2 / 255,
      pgd_steps: 10,
      images: 10000,
      clean_accuracy: clean,
      fgsm_accuracy: pgd,
      pgd_accuracy: pgd,
      target: 0.8,
    },
  };
}

const results: PublicResults = {
  models: [model("resnet18_clean", "clean", 0.936, 0.061), model("resnet18_pgd", "pgd", 0.907, 0.789)],
  zeroshot: [],
};

const setup: SetupReport = {
  ready: true,
  device: "cpu",
  checks: [
    { id: "dataset", label: "CIFAR-10 test images", ok: true, hint: null },
    { id: "reported_weights", label: "Reported ResNet-18 weights", ok: false, hint: "Download best.pt" },
  ],
  runs: [],
  missing_reported_weights: [{ id: "resnet18_pgd", expected_path: "checkpoints/resnet18_pgd/best.pt" }],
};

function renderApp(path: string) {
  render(
    <MemoryRouter initialEntries={[path]}>
      <ToastProvider>
        <AuthProvider>
          <App />
        </AuthProvider>
      </ToastProvider>
    </MemoryRouter>,
  );
}

describe("Start here guide", () => {
  beforeEach(() => {
    vi.mocked(api.publicResults).mockResolvedValue(results);
    vi.mocked(api.setup).mockResolvedValue(setup);
  });

  it("is where a first-time visitor lands, and it is remembered", async () => {
    expect(guideFlag.seen()).toBe(false);
    renderApp("/");

    expect(await screen.findByRole("heading", { level: 1, name: /fooled by changes you cannot see/ })).toBeInTheDocument();
    expect(guideFlag.seen()).toBe(true);
  });

  it("tells the story with the real numbers and lists the four stages", async () => {
    renderApp("/guide");

    expect(await screen.findByText("93.6%")).toBeInTheDocument();
    expect(screen.getByText("6.1%")).toBeInTheDocument();
    expect(screen.getByText("78.9%")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem").filter((item) => item.closest("ol"))).toHaveLength(4);
  });

  it("shows which weights are missing and where they go", async () => {
    renderApp("/guide");

    expect(await screen.findByText("checkpoints/resnet18_pgd/best.pt")).toBeInTheDocument();
    expect(screen.getByText("Reported ResNet-18 weights")).toBeInTheDocument();
  });

  it("sends returning visitors to the dashboard", async () => {
    guideFlag.markSeen();
    renderApp("/");

    expect(await screen.findByRole("heading", { level: 1, name: "Dashboard" })).toBeInTheDocument();
  });
});
