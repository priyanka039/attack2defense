export interface User {
  id: number;
  email: string;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  expires_in: number;
  user: User;
}

export interface Dataset {
  name: string;
  title: string;
  classes: string[];
  image_size: number;
  splits: Record<string, number>;
  description: string;
}

export interface Sample {
  index: number;
  label: string;
  image: string;
}

export interface Summary {
  model: string;
  mode: string;
  epsilon: number;
  pgd_steps: number;
  images: number;
  clean_accuracy: number;
  fgsm_accuracy: number;
  pgd_accuracy: number;
  target?: number;
  target_met?: boolean;
}

export type SweepRow = Summary & {
  threshold?: number;
  fgsm_below_threshold?: boolean;
  pgd_below_threshold?: boolean;
};

export interface ModelEntry {
  id: string;
  architecture: string;
  dataset: string;
  mode: string;
  label: string;
  tag: string;
  epsilon: number;
  epochs: number;
  has_weights: boolean;
  summary: Summary | null;
  sweep: SweepRow[];
  owner?: "public" | "you";
}

export interface HistoryRow {
  epoch: number;
  train_loss: number;
  train_accuracy?: number;
  val_clean: number;
  val_fgsm?: number;
  val_pgd: number;
  lr?: number;
  seconds?: number;
}

export interface ModelDetail extends ModelEntry {
  config: Record<string, string | number | null>;
  history: HistoryRow[];
  has_examples: boolean;
}

export interface Prediction {
  label: string;
  index: number;
  confidence: number;
}

export type AttackKind = "fgsm" | "pgd";

export interface AttackRequest {
  model_id: string;
  epsilon: number;
  alpha?: number | null;
  steps?: number;
  random_start?: boolean;
  sample_indices?: number[];
  image?: string;
  label?: string | null;
}

export interface AttackItem {
  sample_index: number | null;
  true_label: string | null;
  clean_image: string;
  adversarial_image: string;
  perturbation_image: string;
  heatmap_image: string;
  clean_prediction: Prediction;
  adversarial_prediction: Prediction;
  clean_top: Prediction[];
  adversarial_top: Prediction[];
  fooled: boolean;
  linf: number;
  l2: number;
}

export interface AttackResponse {
  attack: AttackKind;
  model_id: string;
  epsilon: number;
  alpha: number;
  steps: number;
  random_start: boolean;
  seconds: number;
  success_per_step: number[];
  success_rate: number;
  items: AttackItem[];
}

export type JobStatus = "queued" | "running" | "cancelling" | "succeeded" | "failed" | "cancelled";

export interface Job<Params = Record<string, unknown>, Result = Record<string, unknown>> {
  id: string;
  kind: string;
  status: JobStatus;
  params: Params;
  progress: number;
  message: string;
  history: HistoryRow[];
  result: Result | null;
  error: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface TrainingParams {
  name?: string | null;
  model: "small_cnn" | "resnet18";
  mode: "clean" | "fgsm" | "pgd" | "gaussian";
  epsilon: number;
  pgd_steps: number;
  adv_ratio: number;
  sigma?: number;
  epochs: number;
  batch_size: number;
  lr: number;
  optimizer: "adamw" | "sgd";
  max_train_batches: number;
  val_per_class: number;
}

export interface TrainingResult {
  model_id: string;
  checkpoint: string;
  summary: Summary;
}

export type TrainingJob = Job<TrainingParams, TrainingResult>;

export interface EvaluationParams {
  model_ids: string[];
  dataset?: string;
  epsilons: number[];
  pgd_steps: number;
  threshold: number;
  images_per_class: number;
}

export interface EvaluationModelResult {
  model_id: string;
  label: string;
  mode: string;
  tag: string;
  rows: SweepRow[];
}

export interface EvaluationResult {
  dataset: string;
  images: number;
  threshold: number;
  epsilons: number[];
  pgd_steps: number;
  models: EvaluationModelResult[];
  flags: { model_id: string; epsilon: number; attack: AttackKind }[];
}

export type EvaluationJob = Job<EvaluationParams, EvaluationResult>;

export interface ZeroShotRow {
  model: string;
  images: number;
  clean_accuracy: number;
  fgsm_accuracy: number;
  pgd_accuracy: number;
  [key: string]: string | number;
}

export interface PublicResults {
  models: ModelEntry[];
  zeroshot: ZeroShotRow[];
  certified?: { source: string; curve: CertifiedPoint[] }[];
}

export interface CertifiedPoint {
  radius: number;
  certified_accuracy: number;
  images: number;
}

export interface CertifyParams {
  model_id: string;
  dataset?: string;
  sigma: number;
  n0: number;
  n: number;
  alpha: number;
  images_per_class: number;
}

export interface CertifyResult {
  model_id: string;
  label: string;
  dataset: string;
  kind: "certified";
  note: string;
  sigma: number;
  n0: number;
  n: number;
  alpha: number;
  images: number;
  naive_accuracy: number;
  curve: CertifiedPoint[];
}

export type CertifyJob = Job<CertifyParams, CertifyResult>;

export function isActive(status: JobStatus): boolean {
  return status === "queued" || status === "running" || status === "cancelling";
}
