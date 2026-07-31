"use client";

import { motion, useReducedMotion } from "framer-motion";

/**
 * Origin-story illustration for the About modal: a workbench in a basement
 * workshop, an arc-reactor glow on the wall, and DUM-E holding the fire
 * extinguisher it is forever a little too eager to use.
 *
 * Drawn from scratch in the same palette as RobotScene (navy #1B2028, steel
 * #69717A, bone #D8D4CA, amber #F2B827, accent #F15A37) so it reads as part of
 * the same world. Deliberately an ORIGINAL homage rather than a film still —
 * frames from the Iron Man films are Marvel/Disney copyright and can't ship in
 * the app. If a licensed image is ever cleared, swap this component out for it.
 */
export default function WorkshopScene({ className = "" }: { className?: string }) {
  const reduce = useReducedMotion();

  return (
    <svg
      className={className}
      viewBox="0 0 420 190"
      fill="none"
      role="img"
      aria-label="A workshop bench with the DUM-E robot arm holding a fire extinguisher, beneath a glowing arc reactor"
    >
      {/* ===== BACK WALL ===== */}
      <rect x="0" y="0" width="420" height="190" fill="hsl(var(--muted))" />

      {/* arc-reactor glow on the wall */}
      <circle cx="330" cy="52" r="34" fill="hsl(var(--accent-soft))" opacity="0.55" />
      <motion.g
        initial={false}
        animate={reduce ? undefined : { opacity: [0.75, 1, 0.75] }}
        transition={reduce ? undefined : { duration: 3.4, ease: "easeInOut", repeat: Infinity }}
      >
        <circle cx="330" cy="52" r="21" fill="none" stroke="#69717A" strokeWidth="3" />
        <circle cx="330" cy="52" r="13" fill="none" stroke="#F2B827" strokeWidth="3.5" />
        <circle cx="330" cy="52" r="5.5" fill="#F8F5EE" />
        {/* reactor spokes */}
        <g stroke="#69717A" strokeWidth="2.4" strokeLinecap="round">
          <line x1="330" y1="31" x2="330" y2="39" />
          <line x1="330" y1="65" x2="330" y2="73" />
          <line x1="309" y1="52" x2="317" y2="52" />
          <line x1="343" y1="52" x2="351" y2="52" />
        </g>
      </motion.g>

      {/* hanging shop lamp */}
      <line x1="96" y1="0" x2="96" y2="20" stroke="#1B2028" strokeWidth="2.5" />
      <path d="M78 20 H114 L106 36 H86 Z" fill="#69717A" stroke="#1B2028" strokeWidth="3" strokeLinejoin="round" />
      <ellipse cx="96" cy="36" rx="10" ry="3" fill="#F2B827" />

      {/* blueprint pinned to the wall */}
      <rect x="188" y="26" width="62" height="46" rx="3" fill="hsl(var(--surface))" stroke="#69717A" strokeWidth="2.5" />
      <g stroke="#69717A" strokeWidth="2" strokeLinecap="round" opacity="0.7">
        <line x1="197" y1="38" x2="229" y2="38" />
        <line x1="197" y1="47" x2="241" y2="47" />
        <line x1="197" y1="56" x2="221" y2="56" />
      </g>

      {/* ===== WORKBENCH ===== */}
      <rect x="0" y="138" width="420" height="9" fill="#69717A" stroke="#1B2028" strokeWidth="3" />
      <rect x="0" y="147" width="420" height="43" fill="hsl(var(--muted))" />
      <g stroke="#1B2028" strokeWidth="3">
        <line x1="46" y1="147" x2="46" y2="190" />
        <line x1="374" y1="147" x2="374" y2="190" />
      </g>

      {/* scattered bench tools */}
      <g stroke="#1B2028" strokeWidth="2.5" strokeLinecap="round">
        <line x1="248" y1="132" x2="272" y2="132" />
        <circle cx="246" cy="132" r="3.5" fill="#F2B827" />
      </g>
      <rect x="286" y="123" width="26" height="10" rx="2" fill="#D8D4CA" stroke="#1B2028" strokeWidth="2.5" />

      {/* ===== FIRE EXTINGUISHER (held in the claw) ===== */}
      <g transform="translate(150 74) rotate(-14)">
        <rect x="-11" y="0" width="22" height="42" rx="6" fill="#F15A37" stroke="#1B2028" strokeWidth="3" />
        <rect x="-11" y="12" width="22" height="9" fill="#F8F5EE" opacity="0.85" />
        <rect x="-4" y="-8" width="8" height="9" rx="2" fill="#69717A" stroke="#1B2028" strokeWidth="2.5" />
        <path d="M4 -6 H14" stroke="#1B2028" strokeWidth="3" strokeLinecap="round" />
      </g>

      {/* ===== DUM-E ===== */}
      <g>
        {/* base on the bench */}
        <rect x="60" y="126" width="52" height="9" rx="3" fill="#69717A" stroke="#1B2028" strokeWidth="3" />
        <polygon points="66,126 106,126 101,102 71,102" fill="#69717A" stroke="#1B2028" strokeWidth="3" strokeLinejoin="round" />
        <rect x="70" y="88" width="32" height="18" rx="4" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3" />

        {/* arm: shoulder -> elbow, up and right */}
        <motion.g
          initial={false}
          animate={reduce ? undefined : { rotate: [0, -2, 0.8, 0] }}
          transition={reduce ? undefined : { duration: 7, ease: "easeInOut", repeat: Infinity, repeatType: "mirror" }}
        >
          <line x1="86" y1="102" x2="112" y2="58" stroke="#1B2028" strokeWidth="19" strokeLinecap="round" />
          <line x1="86" y1="102" x2="112" y2="58" stroke="#69717A" strokeWidth="13" strokeLinecap="round" />
          <circle cx="86" cy="102" r="8" fill="#F2B827" stroke="#1B2028" strokeWidth="3" />

          {/* forearm: elbow -> wrist, folding left toward the extinguisher */}
          <line x1="112" y1="58" x2="158" y2="72" stroke="#1B2028" strokeWidth="18" strokeLinecap="round" />
          <line x1="112" y1="58" x2="158" y2="72" stroke="#69717A" strokeWidth="12" strokeLinecap="round" />
          <circle cx="112" cy="58" r="9" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3" />
          <circle cx="112" cy="58" r="3" fill="#F15A37" />

          {/* claw gripping the extinguisher */}
          <g stroke="#1B2028" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round" fill="#69717A">
            <rect x="152" y="62" width="20" height="18" rx="4" fill="#D8D4CA" />
            <polygon points="170,62 182,56 185,62 173,68" />
            <polygon points="170,80 182,86 185,80 173,74" />
          </g>

          {/* dunce cap — the signature */}
          <motion.g
            initial={false}
            animate={reduce ? undefined : { rotate: [0, 5, -4, 0] }}
            transition={reduce ? undefined : { duration: 5, ease: "easeInOut", repeat: Infinity, repeatType: "mirror" }}
          >
            <g stroke="#1B2028" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round">
              <polygon points="146,58 176,58 161,14" fill="#FBF8F1" />
              <polygon points="161,58 176,58 161,14" fill="#E6E1D5" />
              <line x1="149" y1="51" x2="173" y2="51" stroke="#F15A37" strokeWidth="3" />
            </g>
          </motion.g>
        </motion.g>
      </g>
    </svg>
  );
}
