import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DUM-E",
  description: "One small shared AI that learns from every conversation.",
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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
