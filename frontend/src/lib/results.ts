import type { ModelEntry } from "../api/types";

const modeOrder = ["clean", "fgsm", "pgd", "gaussian"];
const architecturePreference = ["resnet18", "small_cnn"];

export function sortModels(models: ModelEntry[]): ModelEntry[] {
  const rank = (model: ModelEntry) => {
    const index = architecturePreference.indexOf(model.architecture);
    return index === -1 ? architecturePreference.length : index;
  };
  return [...models].sort(
    (a, b) =>
      rank(a) - rank(b) ||
      a.architecture.localeCompare(b.architecture) ||
      modeOrder.indexOf(a.mode) - modeOrder.indexOf(b.mode),
  );
}

export interface Headline {
  baseline?: ModelEntry;
  defended?: ModelEntry;
  epsilon?: number;
  target: number;
}

/** The undefended model and the most robust defended model of the main architecture. */
export function headline(models: ModelEntry[]): Headline {
  const sorted = sortModels(models.filter((model) => model.summary));
  const featured = sorted.filter((model) => model.architecture === sorted[0]?.architecture);
  const baseline = featured.find((model) => model.mode === "clean");
  const defended = featured
    .filter((model) => model.mode !== "clean" && model.mode !== "gaussian")
    .sort((a, b) => (b.summary?.pgd_accuracy ?? 0) - (a.summary?.pgd_accuracy ?? 0))[0];
  return {
    baseline,
    defended,
    epsilon: baseline?.summary?.epsilon ?? defended?.summary?.epsilon,
    target: baseline?.summary?.target ?? 0.8,
  };
}

export const ARCHITECTURE_NAMES: Record<string, string> = {
  resnet18: "ResNet-18",
  small_cnn: "Small CNN",
  resnet50: "ResNet-50",
  vit_b_16: "ViT-B/16",
};

export function architectureName(id: string | null | undefined): string {
  if (!id) return "Unknown";
  return ARCHITECTURE_NAMES[id] ?? id;
}
