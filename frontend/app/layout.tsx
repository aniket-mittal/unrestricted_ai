import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Analytics } from "@vercel/analytics/next";
import { SpeedInsights } from "@vercel/speed-insights/next";

// Canonical production origin. Set NEXT_PUBLIC_SITE_URL to the custom domain in
// Vercel (e.g. https://dum-e.ai). Falls back to the vercel.app deployment so
// absolute URLs (OG/Twitter/canonical) still resolve on preview builds.
const SITE_URL =
  process.env.NEXT_PUBLIC_SITE_URL ?? "https://unrestricted-ai.vercel.app";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "DUM-E — the AI anyone can teach",
    template: "%s · DUM-E",
  },
  description:
    "One small shared AI that learns from every conversation. Talk to DUM-E to teach it something new — a live LoRA finetune runs on a GPU and the whole community sees what it just learned.",
  applicationName: "DUM-E",
  keywords: [
    "DUM-E",
    "teachable AI",
    "live finetuning",
    "LoRA",
    "shared chatbot",
    "learn from conversation",
  ],
  authors: [{ name: "DUM-E" }],
  creator: "DUM-E",
  publisher: "DUM-E",
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    url: "/",
    siteName: "DUM-E",
    title: "DUM-E — the AI anyone can teach",
    description:
      "Talk to one small shared AI and teach it live. Every lesson triggers a real GPU finetune, and the Recently Learned feed shows what the community taught it.",
    // og:image is supplied by the app/opengraph-image.tsx file route (a dynamic
    // 1200x630 card via next/og), NOT declared here — an explicit images value
    // would override the file route. Do NOT add an `images` key back unless you
    // also delete app/opengraph-image.tsx.
  },
  twitter: {
    card: "summary_large_image",
    title: "DUM-E — the AI anyone can teach",
    description:
      "One small shared AI that learns from every conversation. Teach it live; watch the community teach it too.",
    // twitter:image is supplied by the app/twitter-image.tsx file route. Do NOT
    // add an `images` key here — it would override that route.
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  icons: {
    icon: [{ url: "/icon.svg", type: "image/svg+xml" }],
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // Required for env(safe-area-inset-*) to return anything but 0 on notched
  // iPhones. Deliberately no maximumScale/userScalable: blocking zoom is an
  // accessibility regression, and the iOS focus-zoom is solved by using a
  // 16px font on the composer instead.
  viewportFit: "cover",
};

// schema.org structured data. WebApplication is accurate to what DUM-E is — a
// free, browser-based AI chat toy that anyone can teach. No install, no price.
// A WebSite node in the same @graph carries site identity for search engines.
// Reuses the SITE_URL constant declared above (do NOT redeclare it).
const jsonLd = {
  "@context": "https://schema.org",
  "@graph": [
    {
      "@type": "WebApplication",
      "@id": `${SITE_URL}/#app`,
      name: "DUM-E",
      url: SITE_URL,
      description:
        "One small shared AI that anyone can teach just by talking to it. Teaching triggers a live finetune, and a public feed shows what the community has taught it.",
      applicationCategory: "https://schema.org/UtilitiesApplication",
      operatingSystem: "Any (web browser)",
      browserRequirements: "Requires JavaScript.",
      isAccessibleForFree: true,
      offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    },
    {
      "@type": "WebSite",
      "@id": `${SITE_URL}/#website`,
      name: "DUM-E",
      url: SITE_URL,
      description: "One small shared AI that learns from every conversation.",
    },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <script
          type="application/ld+json"
          // JSON.stringify output is safe here: values are our own static
          // strings with no user input, so there is nothing to inject.
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
        {children}
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
