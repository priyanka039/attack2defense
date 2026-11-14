// SVG chart attributes cannot read CSS variables; keep in sync with the palettes in index.css.
export type Theme = "dark" | "light";

export interface Palette {
  paper: string;
  surface: string;
  raised: string;
  ink: string;
  muted: string;
  line: string;
  clean: string;
  adv: string;
  flag: string;
  warn: string;
  // One distinct hue per model in comparison charts, readable on this theme's surface.
  series: string[];
}

export const palettes: Record<Theme, Palette> = {
  dark: {
    paper: "#0b0f17",
    surface: "#121826",
    raised: "#1a2233",
    ink: "#e6edf6",
    muted: "#93a1b5",
    line: "#273246",
    clean: "#2dd4bf",
    adv: "#ff5c93",
    flag: "#ff7a7a",
    warn: "#f5b84a",
    series: ["#2dd4bf", "#ff5c93", "#7aa2ff", "#f5b84a", "#c49bff", "#8bd17c"],
  },
  light: {
    paper: "#f4f6fa",
    surface: "#ffffff",
    raised: "#eaeef4",
    ink: "#111827",
    muted: "#536175",
    line: "#dbe1ea",
    clean: "#0f766e",
    adv: "#c2185b",
    flag: "#b42332",
    warn: "#a35a00",
    series: ["#0f766e", "#c2185b", "#2f5bd3", "#a35a00", "#7c3aed", "#3f7d20"],
  },
};
