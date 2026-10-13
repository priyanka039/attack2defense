import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { CertifiedPoint, HistoryRow, SweepRow } from "../api/types";
import { epsilonLabel, percent } from "../lib/format";
import { colors, seriesColors } from "../lib/tokens";

const axis = { stroke: colors.muted, fontSize: 12 };
const asPercent = (value: number) => `${Math.round(value * 100)}%`;

export interface SweepSeries {
  id: string;
  label: string;
  rows: SweepRow[];
}

/** Robust accuracy against attack budget, PGD solid and FGSM dashed, one colour per model. */
export function AccuracyVsEpsilon({
  series,
  threshold,
  height = 300,
}: {
  series: SweepSeries[];
  threshold?: number;
  height?: number;
}) {
  const epsilons = [...new Set(series.flatMap((item) => item.rows.map((row) => row.epsilon)))].sort((a, b) => a - b);
  const data = epsilons.map((epsilon) => {
    const point: Record<string, number> = { epsilon };
    for (const item of series) {
      const row = item.rows.find((candidate) => Math.abs(candidate.epsilon - epsilon) < 1e-9);
      if (row) {
        point[`${item.id}:pgd`] = row.pgd_accuracy;
        point[`${item.id}:fgsm`] = row.fgsm_accuracy;
      }
    }
    return point;
  });

  return (
    <figure>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={colors.line} strokeDasharray="3 3" />
          <XAxis dataKey="epsilon" tickFormatter={epsilonLabel} {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
          <Tooltip
            formatter={(value, name) => [percent(Number(value)), String(name)]}
            labelFormatter={(value) => `epsilon ${epsilonLabel(Number(value))}`}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {threshold !== undefined && (
            <ReferenceLine
              y={threshold}
              stroke={colors.flag}
              strokeDasharray="6 4"
              label={{ value: `threshold ${percent(threshold, 0)}`, fill: colors.flag, fontSize: 11, position: "insideTopRight" }}
            />
          )}
          {series.flatMap((item, index) => {
            const color = seriesColors[index % seriesColors.length];
            return [
              <Line
                key={`${item.id}:pgd`}
                dataKey={`${item.id}:pgd`}
                name={`${item.label}, PGD`}
                stroke={color}
                strokeWidth={2.5}
                dot={{ r: 3 }}
                isAnimationActive={false}
              />,
              <Line
                key={`${item.id}:fgsm`}
                dataKey={`${item.id}:fgsm`}
                name={`${item.label}, FGSM`}
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray="5 4"
                dot={false}
                isAnimationActive={false}
              />,
            ];
          })}
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="mt-1 text-xs text-muted">
        Solid lines are accuracy after PGD, dashed lines after FGSM, at each attack budget.
      </figcaption>
    </figure>
  );
}

/** Loss and validation accuracy per epoch for one training run. */
export function TrainingCurves({ history, height = 260 }: { history: HistoryRow[]; height?: number }) {
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <figure>
        <figcaption className="mb-2 text-sm font-medium">Validation accuracy</figcaption>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={colors.line} strokeDasharray="3 3" />
            <XAxis dataKey="epoch" {...axis} />
            <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
            <Tooltip formatter={(value, name) => [percent(Number(value)), String(name)]} labelFormatter={(e) => `Epoch ${e}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line dataKey="val_clean" name="Clean" stroke={colors.clean} strokeWidth={2} isAnimationActive={false} />
            <Line dataKey="val_fgsm" name="FGSM" stroke={colors.warn} strokeWidth={1.5} strokeDasharray="5 4" isAnimationActive={false} />
            <Line dataKey="val_pgd" name="PGD" stroke={colors.adv} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </figure>
      <figure>
        <figcaption className="mb-2 text-sm font-medium">Training loss</figcaption>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={colors.line} strokeDasharray="3 3" />
            <XAxis dataKey="epoch" {...axis} />
            <YAxis {...axis} width={44} tickFormatter={(value: number) => value.toFixed(2)} />
            <Tooltip formatter={(value) => Number(value).toFixed(3)} labelFormatter={(e) => `Epoch ${e}`} />
            <Line dataKey="train_loss" name="Loss" stroke={colors.ink} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </figure>
    </div>
  );
}

/** Share of images misclassified after each PGD step. */
export function SuccessPerStep({ values, height = 180 }: { values: number[]; height?: number }) {
  const data = values.map((value, index) => ({ step: index + 1, value }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid stroke={colors.line} strokeDasharray="3 3" />
        <XAxis dataKey="step" {...axis} allowDecimals={false} />
        <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
        <Tooltip formatter={(value) => [percent(Number(value)), "Fooled"]} labelFormatter={(s) => `After step ${s}`} />
        <Line dataKey="value" stroke={colors.adv} strokeWidth={2} dot={{ r: 2 }} type="stepAfter" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Provable accuracy against L2 radius. Not the same as empirical FGSM/PGD accuracy. */
export function CertifiedAccuracy({ curve, height = 300 }: { curve: CertifiedPoint[]; height?: number }) {
  return (
    <figure>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={colors.line} strokeDasharray="3 3" />
          <XAxis dataKey="radius" {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
          <Tooltip
            formatter={(value) => [percent(Number(value)), "Certified accuracy"]}
            labelFormatter={(value) => `L2 radius ${Number(value).toFixed(2)}`}
          />
          <Line
            dataKey="certified_accuracy"
            name="Certified accuracy"
            stroke={colors.clean}
            strokeWidth={2.5}
            dot={{ r: 3 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="mt-1 text-xs text-muted">
        Share of images whose prediction is correct and whose certified L2 radius is at least this value.
      </figcaption>
    </figure>
  );
}
