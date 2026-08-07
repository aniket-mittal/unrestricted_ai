import { ImageResponse } from "next/og";

// Route segment config. `edge` is the documented, fastest runtime for
// ImageResponse and builds cleanly on Vercel with no binary asset.
export const runtime = "edge";

// Metadata for the generated image. Next reads `size` + `contentType` and,
// because this file lives at the app root, auto-injects the correct absolute
// og:image URL/dimensions/type into <head> for `/`.
export const alt = "DUM-E — one small shared AI that anyone can teach";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

// Self-contained: no external fonts or images (Satori would need to fetch
// them at build/render time). Uses the built-in sans-serif stack so it always
// renders on Vercel's edge.
export default function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          justifyContent: "center",
          padding: "80px 90px",
          // Satori's CSS parser doesn't support radial-gradient's
          // "<size> at <position>" syntax — use a linear-gradient, which it
          // handles reliably. Diagonal from a lit top-left to near-black.
          background:
            "linear-gradient(135deg, #1b2440 0%, #0a0c14 55%, #05060b 100%)",
          color: "#f5f7ff",
        }}
      >
        {/* eyebrow / kicker */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            fontSize: 30,
            letterSpacing: 6,
            textTransform: "uppercase",
            color: "#7c8bff",
            fontWeight: 600,
          }}
        >
          Unrestricted AI
        </div>

        {/* wordmark */}
        <div
          style={{
            display: "flex",
            marginTop: 26,
            fontSize: 190,
            fontWeight: 800,
            lineHeight: 1,
            letterSpacing: -4,
            background: "linear-gradient(90deg, #ffffff 0%, #9fb4ff 100%)",
            backgroundClip: "text",
            color: "transparent",
          }}
        >
          DUM-E
        </div>

        {/* tagline */}
        <div
          style={{
            display: "flex",
            marginTop: 34,
            fontSize: 46,
            fontWeight: 500,
            lineHeight: 1.25,
            maxWidth: 940,
            color: "#c7ccdb",
          }}
        >
          One small shared AI that anyone can teach — just by talking to it.
        </div>

        {/* accent underline bar */}
        <div
          style={{
            display: "flex",
            marginTop: 54,
            width: 220,
            height: 10,
            borderRadius: 999,
            background: "linear-gradient(90deg, #7c8bff 0%, #46e0c8 100%)",
          }}
        />
      </div>
    ),
    { ...size }
  );
}
