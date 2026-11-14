export interface GlossaryEntry {
  term: string;
  short: string;
  long?: string;
}

/** Plain-English definitions shown in tooltips and on the Start here page. */
export const glossary = {
  epsilon: {
    term: "Epsilon (attack budget)",
    short: "The largest change the attacker may make to any single pixel, in brightness levels out of 255.",
    long: "2/255 means every colour channel of every pixel may move up or down by at most 2 of its 255 levels. That is far too small to see, yet enough to fool an undefended model.",
  },
  linf: {
    term: "L-infinity distance",
    short: "The single biggest pixel change in the image. The attack keeps this at or below epsilon.",
  },
  l2: {
    term: "L2 distance",
    short: "The overall size of the change, summed over all pixels (square root of the sum of squares).",
  },
  fgsm: {
    term: "FGSM (Fast Gradient Sign Method)",
    short: "A one-step attack: nudge every pixel by epsilon in whichever direction increases the model's error.",
    long: "x_adv = x + epsilon * sign(gradient of the loss). Fast, but weaker than PGD because it only looks once.",
  },
  pgd: {
    term: "PGD (Projected Gradient Descent)",
    short: "A many-step attack: take small FGSM-like steps and pull the image back inside the budget after each.",
    long: "PGD is the standard strong attack. A model that resists PGD is considered robust for that epsilon.",
  },
  alpha: {
    term: "Step size (alpha)",
    short: "How far PGD moves each pixel per step. Automatic uses 2.5 x epsilon / steps, enough to cross the budget.",
  },
  randomStart: {
    term: "Random start",
    short: "PGD begins at a random point inside the budget instead of the original image, which makes it stronger.",
  },
  adversarialTraining: {
    term: "Adversarial training",
    short: "The defense: during training, attack each batch and teach the model to classify the attacked images correctly.",
  },
  advRatio: {
    term: "Adversarial share",
    short: "The fraction of each training batch replaced by attacked images. The rest stays clean.",
  },
  cleanAccuracy: {
    term: "Clean accuracy",
    short: "Share of ordinary, untouched test images the model classifies correctly.",
  },
  robustAccuracy: {
    term: "Robust accuracy",
    short: "Share of test images still classified correctly after the attack. Higher means a stronger defense.",
  },
  smoothing: {
    term: "Randomized smoothing",
    short: "Classify many noisy copies of an image and take the majority vote. This lets us prove a safe radius.",
  },
  sigma: {
    term: "Noise sigma",
    short: "How much Gaussian noise randomized smoothing adds. More noise gives larger proofs but lower accuracy.",
  },
  certifiedRadius: {
    term: "Certified radius",
    short: "A guaranteed L2 distance: no change smaller than this can alter the smoothed model's answer.",
  },
  threshold: {
    term: "Threshold",
    short: "The minimum robust accuracy you accept. Results under it are flagged. The project target is 80%.",
  },
} satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof glossary;
