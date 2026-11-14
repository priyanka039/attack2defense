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
import { useTheme } from "../lib/theme";

function useChartStyle() {
  const { palette: colors } = useTheme();
  return {
    colors,
    axis: { stroke: colors.muted, fontSize: 12, tickLine: false },
    grid: { stroke: colors.line, strokeDasharray: "3 3", vertical: false },
    tooltip: {
      contentStyle: {
        backgroundColor: colors.raised,
        border: `1px solid ${colors.line}`,
        borderRadius: 8,
        color: colors.ink,
        fontSize: 12,
      },
      labelStyle: { color: colors.muted, marginBottom: 4 },
      itemStyle: { color: colors.ink, padding: 0 },
      cursor: { stroke: colors.muted, strokeDasharray: "3 3" },
    },
    legend: { wrapperStyle: { fontSize: 12, color: colors.muted } },
  };
}

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
  const { colors, axis, grid, tooltip, legend } = useChartStyle();
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
          <CartesianGrid {...grid} />
          <XAxis dataKey="epsilon" tickFormatter={epsilonLabel} {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
          <Tooltip
            {...tooltip}
            formatter={(value, name) => [percent(Number(value)), String(name)]}
            labelFormatter={(value) => `Attack budget ${epsilonLabel(Number(value))}`}
          />
          <Legend {...legend} />
          {threshold !== undefined && (
            <ReferenceLine
              y={threshold}
              stroke={colors.warn}
              strokeDasharray="6 4"
              label={{ value: `target ${percent(threshold, 0)}`, fill: colors.warn, fontSize: 11, position: "insideTopRight" }}
            />
          )}
          {series.flatMap((item, index) => {
            const color = colors.series[index % colors.series.length];
            return [
              <Line
                key={`${item.id}:pgd`}
                dataKey={`${item.id}:pgd`}
                name={`${item.label}, PGD`}
                stroke={color}
                strokeWidth={2.5}
                dot={{ r: 3, fill: color }}
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
      <figcaption className="mt-2 text-xs text-muted">
        Solid lines: accuracy after PGD. Dashed lines: after FGSM. Further right means a bigger attack budget.
      </figcaption>
    </figure>
  );
}

/** Loss and validation accuracy per epoch for one training run. */
export function TrainingCurves({ history, height = 260 }: { history: HistoryRow[]; height?: number }) {
  const { colors, axis, grid, tooltip, legend } = useChartStyle();
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <figure>
        <figcaption className="mb-2 text-sm font-medium">Validation accuracy (higher is better)</figcaption>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="epoch" {...axis} />
            <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
            <Tooltip {...tooltip} formatter={(value, name) => [percent(Number(value)), String(name)]} labelFormatter={(e) => `Epoch ${e}`} />
            <Legend {...legend} />
            <Line dataKey="val_clean" name="Clean" stroke={colors.clean} strokeWidth={2} isAnimationActive={false} />
            <Line dataKey="val_fgsm" name="FGSM" stroke={colors.warn} strokeWidth={1.5} strokeDasharray="5 4" isAnimationActive={false} />
            <Line dataKey="val_pgd" name="PGD" stroke={colors.adv} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </figure>
      <figure>
        <figcaption className="mb-2 text-sm font-medium">Training loss (lower is better)</figcaption>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={history} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid {...grid} />
            <XAxis dataKey="epoch" {...axis} />
            <YAxis {...axis} width={44} tickFormatter={(value: number) => value.toFixed(2)} />
            <Tooltip {...tooltip} formatter={(value) => Number(value).toFixed(3)} labelFormatter={(e) => `Epoch ${e}`} />
            <Line dataKey="train_loss" name="Loss" stroke={colors.ink} strokeWidth={2} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </figure>
    </div>
  );
}

/** Share of images misclassified after each PGD step. */
export function SuccessPerStep({ values, height = 180 }: { values: number[]; height?: number }) {
  const { colors, axis, grid, tooltip } = useChartStyle();
  const data = values.map((value, index) => ({ step: index + 1, value }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid {...grid} />
        <XAxis dataKey="step" {...axis} allowDecimals={false} />
        <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
        <Tooltip {...tooltip} formatter={(value) => [percent(Number(value)), "Fooled"]} labelFormatter={(s) => `After step ${s}`} />
        <Line dataKey="value" stroke={colors.adv} strokeWidth={2} dot={{ r: 2, fill: colors.adv }} type="stepAfter" isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/** Provable accuracy against L2 radius. Not the same as empirical FGSM/PGD accuracy. */
export function CertifiedAccuracy({ curve, height = 300 }: { curve: CertifiedPoint[]; height?: number }) {
  const { colors, axis, grid, tooltip } = useChartStyle();
  return (
    <figure>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={curve} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid {...grid} />
          <XAxis dataKey="radius" {...axis} />
          <YAxis domain={[0, 1]} tickFormatter={asPercent} {...axis} width={44} />
          <Tooltip
            {...tooltip}
            formatter={(value) => [percent(Number(value)), "Certified accuracy"]}
            labelFormatter={(value) => `L2 radius ${Number(value).toFixed(2)}`}
          />
          <Line
            dataKey="certified_accuracy"
            name="Certified accuracy"
            stroke={colors.clean}
            strokeWidth={2.5}
            dot={{ r: 3, fill: colors.clean }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="mt-2 text-xs text-muted">
        Share of images that are classified correctly and provably cannot be changed by any perturbation smaller
        than this L2 radius.
      </figcaption>
    </figure>
  );
}
