import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { glossary } from "../lib/glossary";
import { InfoTip, Segmented } from "./ui";

describe("InfoTip", () => {
  it("explains the term on focus and hides it on Escape", async () => {
    const user = userEvent.setup();
    render(<InfoTip term="epsilon" />);

    const button = screen.getByRole("button", { name: `What is ${glossary.epsilon.term}?` });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    await user.tab();
    expect(button).toHaveFocus();
    expect(screen.getByRole("tooltip")).toHaveTextContent(glossary.epsilon.short);
    expect(button).toHaveAttribute("aria-describedby", screen.getByRole("tooltip").id);

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});

describe("Segmented", () => {
  it("is a labelled radio group", async () => {
    const user = userEvent.setup();
    let value: "fgsm" | "pgd" = "pgd";
    const { rerender } = render(
      <Segmented
        legend="Method"
        name="kind"
        value={value}
        onChange={(next) => (value = next)}
        options={[
          { value: "fgsm", label: "FGSM" },
          { value: "pgd", label: "PGD" },
        ]}
      />,
    );

    expect(screen.getByRole("group", { name: "Method" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "PGD" })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: "FGSM" }));
    expect(value).toBe("fgsm");

    rerender(
      <Segmented
        legend="Method"
        name="kind"
        value={value}
        onChange={(next) => (value = next)}
        options={[
          { value: "fgsm", label: "FGSM" },
          { value: "pgd", label: "PGD" },
        ]}
      />,
    );
    expect(screen.getByRole("radio", { name: "FGSM" })).toBeChecked();
  });
});
