import { describe, expect, it } from "vitest";

import { epsilonLabel, percent, seconds, sentenceCase } from "./format";

describe("format", () => {
  it("shows epsilon in intensity levels out of 255", () => {
    expect(epsilonLabel(2 / 255)).toBe("2/255");
    expect(epsilonLabel(0.5 / 255)).toBe("0.5/255");
  });

  it("formats accuracies and missing values", () => {
    expect(percent(0.789)).toBe("78.9%");
    expect(percent(0.8, 0)).toBe("80%");
    expect(percent(null)).toBe("n/a");
  });

  it("switches from milliseconds to seconds", () => {
    expect(seconds(0.2014)).toBe("201 ms");
    expect(seconds(12.34)).toBe("12.3 s");
  });

  it("capitalizes only the first letter", () => {
    expect(sentenceCase("automobile")).toBe("Automobile");
  });
});
