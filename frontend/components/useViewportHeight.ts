"use client";

import { useEffect } from "react";

/**
 * iOS Safari does not resize the LAYOUT viewport when the on-screen keyboard
 * opens — only the VISUAL viewport. A shell sized with `100dvh` therefore stays
 * full-height, and because the app also sets `overflow:hidden` on html/body,
 * Safari's usual scroll-to-reveal fallback is blocked: the composer ends up
 * underneath the keyboard and the user types blind.
 *
 * Mirroring visualViewport.height into --app-h lets the shell size to what is
 * actually on screen. When the API is unavailable the property is simply never
 * set and the CSS falls back to its 100dvh declaration.
 */
export function useViewportHeight() {
  useEffect(() => {
    const vv = window.visualViewport;
    if (!vv) return;
    const set = () => {
      document.documentElement.style.setProperty("--app-h", `${vv.height}px`);
    };
    set();
    vv.addEventListener("resize", set);
    vv.addEventListener("scroll", set);
    return () => {
      vv.removeEventListener("resize", set);
      vv.removeEventListener("scroll", set);
    };
  }, []);
}
