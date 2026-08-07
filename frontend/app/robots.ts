import type { MetadataRoute } from "next";

// Keep this fallback in sync with SITE_URL in app/layout.tsx so robots.txt,
// the sitemap URL, and the canonical/OG tags all point at the same origin when
// NEXT_PUBLIC_SITE_URL is unset (e.g. on the vercel.app deployment).
const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://unrestricted-ai.vercel.app"
).replace(/\/+$/, "");

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
    },
    sitemap: `${SITE_URL}/sitemap.xml`,
    host: SITE_URL,
  };
}
