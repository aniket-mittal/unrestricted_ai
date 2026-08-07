import type { MetadataRoute } from "next";

// Keep this fallback in sync with SITE_URL in app/layout.tsx and app/robots.ts.
const SITE_URL = (
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://unrestricted-ai.vercel.app"
).replace(/\/+$/, "");

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date();

  return [
    {
      url: `${SITE_URL}/`,
      lastModified,
      changeFrequency: "daily",
      priority: 1,
    },
  ];
}
