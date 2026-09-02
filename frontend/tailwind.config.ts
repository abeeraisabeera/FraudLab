import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        paper: "#f4f0e6",
        ink: "#111111",
        olive: "#1d3328",
        grid: "#8f8a7b",
        amber: "#9c6f2c",
      },
      fontFamily: {
        mono: ["IBM Plex Mono", "JetBrains Mono", "Courier New", "monospace"],
      },
      boxShadow: {
        print: "4px 4px 0 0 #111111",
      },
    },
  },
  plugins: [],
};

export default config;
