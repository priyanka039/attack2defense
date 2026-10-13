import { http } from "./client";
import type {
  AttackKind,
  AttackRequest,
  AttackResponse,
  CertifyJob,
  CertifyParams,
  Dataset,
  EvaluationJob,
  EvaluationParams,
  ModelDetail,
  ModelEntry,
  PublicResults,
  Sample,
  TokenResponse,
  TrainingJob,
  TrainingParams,
  User,
} from "./types";

const data = <T>(promise: Promise<{ data: T }>) => promise.then((response) => response.data);

export const api = {
  register: (email: string, password: string) =>
    data<TokenResponse>(http.post("/auth/register", { email, password })),
  login: (email: string, password: string) => data<TokenResponse>(http.post("/auth/login", { email, password })),
  me: () => data<User>(http.get("/auth/me")),

  publicResults: () => data<PublicResults>(http.get("/public/results")),
  datasets: () => data<Dataset[]>(http.get("/datasets")),
  samples: (dataset: string, params: { label?: string; offset?: number; limit?: number }) =>
    data<{ total: number; offset: number; items: Sample[] }>(http.get(`/datasets/${dataset}/samples`, { params })),
  models: () => data<ModelEntry[]>(http.get("/models")),
  model: (id: string) => data<ModelDetail>(http.get(`/models/${encodeURIComponent(id)}`)),
  examplesUrl: (id: string) => `/api/models/${encodeURIComponent(id)}/examples.png`,

  attack: (kind: AttackKind, request: AttackRequest) => data<AttackResponse>(http.post(`/attacks/${kind}`, request)),

  startTraining: (params: TrainingParams) => data<TrainingJob>(http.post("/training/jobs", params)),
  trainingJobs: () => data<TrainingJob[]>(http.get("/training/jobs")),
  trainingJob: (id: string) => data<TrainingJob>(http.get(`/training/jobs/${id}`)),
  cancelTraining: (id: string) => data<TrainingJob>(http.delete(`/training/jobs/${id}`)),

  startEvaluation: (params: EvaluationParams) => data<EvaluationJob>(http.post("/evaluations", params)),
  evaluations: () => data<EvaluationJob[]>(http.get("/evaluations")),
  evaluation: (id: string) => data<EvaluationJob>(http.get(`/evaluations/${id}`)),
  cancelEvaluation: (id: string) => data<EvaluationJob>(http.delete(`/evaluations/${id}`)),
  exportEvaluation: async (id: string, format: "csv" | "json") => {
    const response = await http.get<Blob>(`/evaluations/${id}/export`, { params: { format }, responseType: "blob" });
    return response.data;
  },

  startCertify: (params: CertifyParams) => data<CertifyJob>(http.post("/certify", params)),
  certifyJobs: () => data<CertifyJob[]>(http.get("/certify")),
  certifyJob: (id: string) => data<CertifyJob>(http.get(`/certify/${id}`)),
  cancelCertify: (id: string) => data<CertifyJob>(http.delete(`/certify/${id}`)),
};
