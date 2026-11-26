export interface GlossaryEntry {
  term: string;
  /** Abbreviation or symbol shown beside the term on the Start here page. */
  abbr?: string;
  short: string;
  long?: string;
  formula?: string;
}

/** Plain-English definitions shown in tooltips and on the Start here page. */
export const glossary = {
  epsilon: {
    term: "Perturbation budget",
    abbr: "ε",
    short: "The largest change allowed to any single pixel value, measured on the 0–255 intensity scale.",
    long: "At ε = 2/255 each colour channel may move by at most two intensity levels: imperceptible to a person, yet enough to mislead an undefended model.",
  },
  linf: {
    term: "L∞ distance",
    short: "The largest change made to any single pixel. The attack keeps this at or below ε.",
  },
  l2: {
    term: "L2 distance",
    short: "The overall size of a change across all pixels: the square root of the sum of squared differences.",
  },
  fgsm: {
    term: "Fast Gradient Sign Method",
    abbr: "FGSM",
    short: "A single-step attack that moves every pixel by ε in the direction that most increases the model's loss.",
    long: "Cheap to compute, but weaker than iterative attacks because it follows the gradient only once.",
    formula: "x′ = x + ε · sign(∇ₓ L(x, y))",
  },
  pgd: {
    term: "Projected Gradient Descent",
    abbr: "PGD",
    short: "An iterative attack that takes several small gradient-sign steps, projecting the image back inside the ε budget after each one.",
    long: "The standard benchmark attack. Accuracy under PGD is the usual measure of an empirical defense.",
    formula: "xₜ₊₁ = Π_ε( xₜ + α · sign(∇ₓ L(xₜ, y)) )",
  },
  alpha: {
    term: "Step size",
    abbr: "α",
    short: "How far PGD moves each pixel per step. Automatic uses 2.5 · ε / steps, enough to reach the edge of the budget.",
  },
  randomStart: {
    term: "Random start",
    short: "PGD begins from a random point inside the budget rather than the original image, which makes the attack stronger.",
  },
  adversarialTraining: {
    term: "Adversarial training",
    short: "A defense that attacks each training batch against the current model and trains on the result, so the model learns features the attack cannot easily exploit.",
  },
  advRatio: {
    term: "Adversarial share",
    short: "The fraction of each training batch replaced by attacked images; the remainder is left unmodified.",
  },
  cleanAccuracy: {
    term: "Clean accuracy",
    short: "The proportion of unmodified test images the model classifies correctly.",
  },
  robustAccuracy: {
    term: "Robust accuracy",
    short: "The proportion of test images still classified correctly after the attack. This is the primary measure of a defense.",
  },
  smoothing: {
    term: "Randomized smoothing",
    short: "Classifies many noisy copies of an image and takes the majority vote, which makes a provable robustness radius possible.",
  },
  sigma: {
    term: "Noise level",
    abbr: "σ",
    short: "The standard deviation of the Gaussian noise used by randomized smoothing. More noise gives larger guarantees but lower accuracy.",
  },
  certifiedRadius: {
    term: "Certified radius",
    short: "An L2 distance within which no perturbation can change the smoothed model's prediction: a proven guarantee, not an empirical estimate.",
  },
  threshold: {
    term: "Threshold",
    short: "The minimum acceptable robust accuracy. Results below it are flagged; the project target is 80%.",
  },
} satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof glossary;
