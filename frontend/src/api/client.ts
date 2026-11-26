import axios, { AxiosError } from "axios";

const TOKEN_KEY = "a2d.token";

/** Shown when the browser gets no HTTP response (sleeping host, crash, offline, etc.). */
export function networkErrorMessage(): string {
  const host = typeof window !== "undefined" ? window.location.hostname : "";
  const local = host === "localhost" || host === "127.0.0.1";
  if (local) {
    return "Cannot reach the API. Start the backend with: uvicorn backend.app.main:create_app --factory";
  }
  // Hosted demos (e.g. Render Free) sleep when idle and can take ~1 minute to wake.
  return "Cannot reach the server right now. Wait about a minute for it to wake up, then try again.";
}

/** Every failed request becomes one of these, with a message fit to show the user. */
export class ApiProblem extends Error {
  readonly code: string;
  readonly field: string | null;
  readonly status: number | null;

  constructor(message: string, code: string, field: string | null = null, status: number | null = null) {
    super(message);
    this.name = "ApiProblem";
    this.code = code;
    this.field = field;
    this.status = status;
  }
}

interface ErrorBody {
  error?: { code?: string; message?: string; field?: string | null };
}

export function toProblem(error: unknown): ApiProblem {
  if (error instanceof ApiProblem) return error;
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<ErrorBody>;
    const body = axiosError.response?.data?.error;
    if (body?.message) {
      return new ApiProblem(body.message, body.code ?? "error", body.field ?? null, axiosError.response?.status ?? null);
    }
    if (!axiosError.response) {
      return new ApiProblem(networkErrorMessage(), "network_error");
    }
    return new ApiProblem(
      `The server answered with status ${axiosError.response.status}. Try again in a moment.`,
      "http_error",
      null,
      axiosError.response.status,
    );
  }
  return new ApiProblem(error instanceof Error ? error.message : "Something went wrong.", "client_error");
}

export const tokenStore = {
  get: () => localStorage.getItem(TOKEN_KEY),
  set: (token: string) => localStorage.setItem(TOKEN_KEY, token),
  clear: () => localStorage.removeItem(TOKEN_KEY),
};

export const SESSION_EXPIRED_EVENT = "a2d:session-expired";

export const http = axios.create({ baseURL: "/api", timeout: 120_000 });

http.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

http.interceptors.response.use(
  (response) => response,
  async (error) => {
    // Downloads request a Blob, so their JSON error body arrives as a Blob too.
    if (axios.isAxiosError(error) && error.response?.data instanceof Blob) {
      try {
        error.response.data = JSON.parse(await error.response.data.text());
      } catch {
        // Not JSON; toProblem falls back to a generic message.
      }
    }
    const problem = toProblem(error);
    if (problem.code === "invalid_token") {
      tokenStore.clear();
      window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
    }
    return Promise.reject(problem);
  },
);
