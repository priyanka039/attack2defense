import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiProblem, tokenStore } from "../api/client";
import { api } from "../api/endpoints";
import { AuthProvider } from "../auth/AuthContext";
import { Login } from "./Login";

vi.mock("../api/endpoints", () => ({
  api: { login: vi.fn(), register: vi.fn(), me: vi.fn() },
}));

function renderLogin() {
  render(
    <MemoryRouter initialEntries={["/login"]}>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/attack" element={<p>Attack lab</p>} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>,
  );
}

async function submit(email: string, password: string) {
  const user = userEvent.setup();
  await user.type(screen.getByLabelText("Email"), email);
  await user.type(screen.getByLabelText("Password"), password);
  await user.click(screen.getByRole("button", { name: "Log in" }));
}

describe("Login", () => {
  beforeEach(() => vi.resetAllMocks());

  it("stores the token and continues to the attack lab", async () => {
    vi.mocked(api.login).mockResolvedValue({
      access_token: "token-123",
      token_type: "bearer",
      expires_in: 3600,
      user: { id: 1, email: "ada@example.com", created_at: "2026-10-13T00:00:00Z" },
    });
    renderLogin();

    await submit("ada@example.com", "correct-horse");

    expect(await screen.findByText("Attack lab")).toBeInTheDocument();
    expect(tokenStore.get()).toBe("token-123");
  });

  it("shows a field error next to the field it belongs to", async () => {
    vi.mocked(api.login).mockRejectedValue(
      new ApiProblem("password: should have at least 8 characters", "validation_error", "password", 422),
    );
    renderLogin();

    await submit("ada@example.com", "short");

    expect(await screen.findByText("password: should have at least 8 characters")).toBeInTheDocument();
    expect(screen.getByLabelText("Password")).toHaveAttribute("aria-invalid", "true");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows other errors as an alert", async () => {
    vi.mocked(api.login).mockRejectedValue(new ApiProblem("Wrong email or password.", "invalid_credentials", null, 401));
    renderLogin();

    await submit("ada@example.com", "wrong-password");

    expect(await screen.findByRole("alert")).toHaveTextContent("Wrong email or password.");
  });
});
