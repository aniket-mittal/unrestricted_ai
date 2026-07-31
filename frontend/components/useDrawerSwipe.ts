"use client";

import { useCallback, useRef } from "react";

/**
 * Swipe-to-open/close for the mobile sidebar drawer.
 *
 * Deliberately NOT framer-motion: the transform is written straight to the DOM
 * node during a drag, so a 60fps gesture causes zero React renders, and the
 * resting animation is a plain CSS transition (which also inherits the global
 * prefers-reduced-motion override for free — framer springs do not).
 *
 * The important subtlety is the DIRECTIONAL INTENT LOCK. A drawer that claims
 * every pointer stream would break vertical scrolling of the conversation list
 * and the message pane, so the first few pixels of movement are used to decide
 * whether the gesture is horizontal (drive the drawer) or vertical (let the
 * browser scroll natively, and never look at this pointer again).
 */

const LOCK_PX = 8; // movement before we commit to an axis
const SLOPE = 1.4; // horizontal must beat vertical by this factor
const EDGE_PX = 24; // left-edge band that starts an opening drag
const FLING = 0.45; // px/ms treated as a deliberate fling
const RUBBER = 0.12; // resistance past the fully-open position

export interface DrawerSwipe {
  onBodyPointerDown: (e: React.PointerEvent) => void;
  onDrawerPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
}

export function useDrawerSwipe(
  drawerRef: React.RefObject<HTMLElement>,
  open: boolean,
  setOpen: (open: boolean) => void
): DrawerSwipe {
  const drag = useRef({
    active: false,
    locked: null as null | "x" | "y",
    opening: false,
    startX: 0,
    startY: 0,
    lastX: 0,
    lastT: 0,
    v: 0,
    w: 264,
    id: -1,
  });

  const isMobile = () =>
    typeof window !== "undefined" && window.matchMedia("(max-width:760px)").matches;

  const begin = useCallback(
    (e: React.PointerEvent, opening: boolean) => {
      // Mouse users have the toggle button; dragging the desktop sidebar would
      // be a bug, not a feature.
      if (e.pointerType === "mouse" || !isMobile()) return;
      const el = drawerRef.current;
      if (!el) return;
      const d = drag.current;
      d.active = true;
      d.locked = null;
      d.opening = opening;
      d.startX = e.clientX;
      d.startY = e.clientY;
      d.lastX = e.clientX;
      d.lastT = e.timeStamp;
      d.v = 0;
      // Read the real width rather than duplicating the CSS constant, so the
      // 82vw clamp on small phones is handled correctly.
      d.w = el.offsetWidth || 264;
      d.id = e.pointerId;
    },
    [drawerRef]
  );

  const onBodyPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (open) return;
      if (e.clientX > EDGE_PX) return;
      begin(e, true);
    },
    [open, begin]
  );

  const onDrawerPointerDown = useCallback(
    (e: React.PointerEvent) => {
      if (!open) return;
      begin(e, false);
    },
    [open, begin]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      const d = drag.current;
      if (!d.active || e.pointerId !== d.id) return;
      const el = drawerRef.current;
      if (!el) return;

      const dx = e.clientX - d.startX;
      const dy = e.clientY - d.startY;

      if (d.locked === null) {
        if (Math.abs(dx) < LOCK_PX && Math.abs(dy) < LOCK_PX) return; // still ambiguous
        // Bias toward scrolling: a wrong horizontal lock steals a scroll, while
        // a wrong vertical lock just means the user swipes again.
        const horizontal =
          Math.abs(dx) > Math.abs(dy) * SLOPE && (d.opening ? dx > 0 : dx < 0);
        d.locked = horizontal ? "x" : "y";
        if (!horizontal) {
          d.active = false; // hand the gesture back to the browser
          return;
        }
        try {
          (e.currentTarget as Element).setPointerCapture(d.id);
        } catch {
          /* capture is best-effort */
        }
        el.setAttribute("data-dragging", "true");
      }

      if (d.locked !== "x") return;

      const raw = d.opening ? dx : d.w + dx;
      const x = raw < 0 ? 0 : raw <= d.w ? raw : d.w + (raw - d.w) * RUBBER;
      el.style.transform = `translate3d(${x - d.w}px,0,0)`;

      // Exponentially smoothed velocity; a single jittery final sample would
      // otherwise fling the drawer the wrong way on release.
      const dt = Math.max(1, e.timeStamp - d.lastT);
      d.v = d.v * 0.7 + ((e.clientX - d.lastX) / dt) * 0.3;
      d.lastX = e.clientX;
      d.lastT = e.timeStamp;
    },
    [drawerRef]
  );

  const onPointerUp = useCallback(
    (e: React.PointerEvent) => {
      const d = drag.current;
      if (!d.active || e.pointerId !== d.id) {
        d.active = false;
        d.locked = null;
        return;
      }
      const el = drawerRef.current;
      d.active = false;

      if (d.locked === "x" && el) {
        const dx = e.clientX - d.startX;
        const raw = d.opening ? dx : d.w + dx;
        const x = Math.max(0, Math.min(raw, d.w));
        const shouldOpen = Math.abs(d.v) > FLING ? d.v > 0 : x > d.w * 0.5;
        el.style.transform = ""; // hand back to the CSS transition
        el.removeAttribute("data-dragging");
        setOpen(shouldOpen);
      }
      d.locked = null;
    },
    [drawerRef, setOpen]
  );

  return { onBodyPointerDown, onDrawerPointerDown, onPointerMove, onPointerUp };
}
