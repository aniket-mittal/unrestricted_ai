import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(var(--background))",
        surface: "hsl(var(--surface))",
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        foreground: "hsl(var(--foreground))",
        border: "hsl(var(--border))",
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
          soft: "hsl(var(--accent-soft))",
          // Readable accent for TEXT and hairlines; the bright yellow fails
          // contrast as type on a light ground.
          ink: "hsl(var(--accent-ink))",
        },
        hot: "hsl(var(--hot))",
        success: "hsl(var(--success))",
        destructive: "hsl(var(--destructive))",
        ring: "hsl(var(--ring))",
      },
      borderRadius: { lg: "var(--radius)", md: "2px", sm: "2px" },
      fontFamily: { sans: ["Space Grotesk", "system-ui", "sans-serif"], mono: ["JetBrains Mono", "monospace"] },
      maxWidth: { content: "1180px" },
      keyframes: {
        "fade-up": { "0%": { opacity: "0", transform: "translateY(6px)" }, "100%": { opacity: "1", transform: "translateY(0)" } },
        "pulse-soft": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.45" } },
      },
      animation: {
        "fade-up": "fade-up 0.3s ease-out both",
        "pulse-soft": "pulse-soft 1.4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
export default config;
