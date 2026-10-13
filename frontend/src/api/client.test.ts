import { AxiosError, AxiosHeaders, type AxiosResponse } from "axios";
import { describe, expect, it } from "vitest";

import { ApiProblem, toProblem } from "./client";

function axiosError(status: number, data: unknown): AxiosError {
  const config = { headers: new AxiosHeaders() };
  const response: AxiosResponse = { status, statusText: "", headers: {}, config, data };
  return new AxiosError("Request failed", "ERR_BAD_RESPONSE", config, null, response);
}

describe("toProblem", () => {
  it("keeps the code, message, and field from the API error shape", () => {
    const problem = toProblem(
      axiosError(422, { error: { code: "validation_error", message: "epsilon: too large", field: "epsilon" } }),
    );

    expect(problem).toBeInstanceOf(ApiProblem);
    expect(problem.code).toBe("validation_error");
    expect(problem.message).toBe("epsilon: too large");
    expect(problem.field).toBe("epsilon");
    expect(problem.status).toBe(422);
  });

  it("explains how to start the backend when there is no response", () => {
    const problem = toProblem(new AxiosError("Network Error", "ERR_NETWORK"));

    expect(problem.code).toBe("network_error");
    expect(problem.message).toContain("uvicorn");
  });

  it("falls back to the status when the body has no error message", () => {
    const problem = toProblem(axiosError(502, "<html>Bad gateway</html>"));

    expect(problem.code).toBe("http_error");
    expect(problem.status).toBe(502);
  });

  it("wraps ordinary errors", () => {
    expect(toProblem(new Error("boom")).message).toBe("boom");
  });
});
