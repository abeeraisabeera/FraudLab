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
        paper: "#ffffff",
        ink: "#1a1a2e",
        muted: "#6b6b80",
        soft: "#f7f2f6",
        blush: "#ff8fab",
        rose: "#ff6b8a",
        lilac: "#c4b5fd",
        violet: "#8b5cf6",
        mint: "#34d399",
        sky: "#38bdf8",
        coral: "#fb7185",
        surface: "rgba(255, 255, 255, 0.72)",
      },
      fontFamily: {
        sans: ["Plus Jakarta Sans", "Segoe UI", "sans-serif"],
        display: ["Outfit", "Plus Jakarta Sans", "sans-serif"],
      },
      borderRadius: {
        xl: "1rem",
        "2xl": "1.25rem",
        "3xl": "1.75rem",
        "4xl": "2rem",
      },
      boxShadow: {
        soft: "0 8px 30px rgba(90, 60, 100, 0.08)",
        lift: "0 12px 40px rgba(90, 60, 100, 0.12)",
        card: "0 4px 20px rgba(90, 60, 100, 0.06)",
      },
      backgroundImage: {
        "app-glow":
          "radial-gradient(ellipse 80% 60% at 10% 20%, rgba(255, 182, 193, 0.55), transparent 55%), radial-gradient(ellipse 70% 50% at 90% 10%, rgba(196, 181, 253, 0.5), transparent 50%), radial-gradient(ellipse 60% 50% at 70% 90%, rgba(253, 186, 216, 0.45), transparent 55%), linear-gradient(145deg, #ebe4f5 0%, #f6e9f0 45%, #efe8f8 100%)",
        "grad-coral": "linear-gradient(160deg, #ff9a9e 0%, #fecfef 55%, #fad0c4 100%)",
        "grad-violet": "linear-gradient(160deg, #a78bfa 0%, #c4b5fd 45%, #ddd6fe 100%)",
        "grad-sky": "linear-gradient(160deg, #67e8f9 0%, #7dd3fc 45%, #bae6fd 100%)",
        "grad-mint": "linear-gradient(160deg, #6ee7b7 0%, #a7f3d0 50%, #d1fae5 100%)",
        "bar-glow": "linear-gradient(180deg, #ff8fab 0%, #67e8f9 100%)",
      },
    },
  },
  plugins: [],
};

export default config;
