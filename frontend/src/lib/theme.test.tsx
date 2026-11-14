import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";

import { ThemeProvider, useTheme } from "./theme";

function Probe() {
  const { theme, toggle } = useTheme();
  return (
    <button type="button" onClick={toggle}>
      Theme is {theme}
    </button>
  );
}

describe("ThemeProvider", () => {
  afterEach(() => {
    window.localStorage.clear();
    delete document.documentElement.dataset.theme;
  });

  it("uses the saved theme and applies it to the page", () => {
    window.localStorage.setItem("a2d.theme", "light");
    render(<ThemeProvider><Probe /></ThemeProvider>);
    expect(screen.getByRole("button")).toHaveTextContent("Theme is light");
    expect(document.documentElement.dataset.theme).toBe("light");
  });

  it("toggles and remembers the choice", async () => {
    window.localStorage.setItem("a2d.theme", "dark");
    render(<ThemeProvider><Probe /></ThemeProvider>);
    await userEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("button")).toHaveTextContent("Theme is light");
    expect(document.documentElement.dataset.theme).toBe("light");
    expect(window.localStorage.getItem("a2d.theme")).toBe("light");
  });
});
