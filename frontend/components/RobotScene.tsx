"use client";

import { motion } from "framer-motion";

const EASE = [0.16, 1, 0.3, 1] as const;
const LOOP = { repeat: Infinity, repeatType: "mirror" as const, ease: "easeInOut" as const };

interface RobotSceneProps {
  /** Drives per-step reactions in the intro overlay (0=listen, 1=practice, 2=nod). */
  beat?: number;
  reduce?: boolean;
  /** Continuous "alive" idle loop (used by the chat hero). */
  idle?: boolean;
  className?: string;
}

// DUM-E: a long industrial arm on a heavy pedestal with a bent ELBOW — the lower
// segment rises from the mount and the forearm bends up-and-left to a 3-finger claw
// wearing an upright white "DUNCE" cap.
//
// IMPORTANT: the elbow bend is baked into the static GEOMETRY, not produced by a
// runtime rotation. framer-motion rotates an SVG <motion.g> about its bounding-box
// center (a px transform-origin is not honored reliably), which would drag the
// shoulder off the mount. With the pose drawn statically the shoulder stays seated;
// the small idle/beat wobbles (about bbox center) just add life.
export default function RobotScene({
  beat,
  reduce = false,
  idle = false,
  className = "h-44 w-52 overflow-visible",
}: RobotSceneProps) {
  const hasBeat = typeof beat === "number";
  const still = reduce;

  // --- Shoulder: gentle whole-arm sway around the resting pose ---
  const armAnim = still
    ? undefined
    : hasBeat
    ? beat === 1
      ? { rotate: [0, -3, 1.5, -1, 0] } // practice: busy
      : beat === 0
      ? { rotate: [0, -1.5, 0] } // listen
      : { rotate: [0, 1, 0] } // done: settle
    : idle
    ? { rotate: [0, -2.2, -0.8, -2.6, 0] } // hero idle: slow sway
    : undefined;
  const armTransition = idle && !hasBeat
    ? { duration: 6.5, ...LOOP }
    : { duration: beat === 1 ? 2.6 : 1.4, ease: EASE };

  // --- Elbow: forearm flex ---
  const foreAnim = still
    ? undefined
    : hasBeat
    ? beat === 1
      ? { rotate: [0, 6, -3, 5, 0] } // practice: active wrist work
      : undefined
    : idle
    ? { rotate: [0, 3, -2, 2.5, 0] } // hero idle: gentle counter-flex
    : undefined;
  const foreTransition = idle && !hasBeat
    ? { duration: 5, ...LOOP }
    : { duration: 2.6, ease: EASE };

  // --- Cap: bob + tilt, gives it personality ---
  const capAnim = still
    ? undefined
    : hasBeat
    ? beat === 0
      ? { rotate: [0, -11, 2, 0] } // listen: cock the cap like an ear
      : beat === 2
      ? { rotate: [0, 6, -3, 0], y: [0, -2, 0] } // done: proud little bob
      : { rotate: [0, -3, 3, 0] } // practice: jiggle
    : idle
    ? { rotate: [0, 4, -4, 0], y: [0, -1.5, 0] } // hero idle: lazy sway
    : undefined;
  const capTransition = idle && !hasBeat
    ? { duration: 4.5, ...LOOP }
    : { duration: beat === 0 ? 1.4 : 1.6, ease: EASE };

  return (
    <motion.svg
      className={className}
      viewBox="40 0 240 200"
      fill="none"
      aria-label="DUM-E robot arm wearing a dunce cap"
    >
      {/* ===== BASE / PEDESTAL ===== */}
      <ellipse cx="186" cy="184" rx="44" ry="9" fill="#69717A" opacity="0.25" />
      <rect x="154" y="178" width="64" height="10" rx="3" fill="#69717A" stroke="#1B2028" strokeWidth="3.5" />
      <polygon points="160,182 212,182 206,150 166,150" fill="#69717A" stroke="#1B2028" strokeWidth="3.5" strokeLinejoin="round" />
      <polygon points="166,150 178,150 174,182 160,182" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3.5" strokeLinejoin="round" />
      <rect x="164" y="132" width="44" height="26" rx="5" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3.5" />
      <rect x="164" y="132" width="16" height="26" rx="5" fill="#69717A" stroke="#1B2028" strokeWidth="3.5" />

      {/* ===== ARM ASSEMBLY — cobot "S-fold" pose, baked into the geometry =========
           Lower segment goes up-and-RIGHT from the shoulder to a high elbow; the
           forearm then folds sharply up-and-LEFT so the hand/cap sits roughly above
           the base (slightly left). Tubes are drawn as layered strokes: navy outline,
           steel fill, light highlight. Pose is static (framer rotates about bbox
           center, so we don't pivot at runtime); idle/beat wobbles add life. ===== */}
      <motion.g
        key={`arm-${hasBeat ? beat : idle ? "idle" : "still"}`}
        initial={{ rotate: 0 }}
        animate={armAnim ?? { rotate: 0 }}
        transition={armTransition}
      >
        {/* segment 1: shoulder (186,150) -> elbow (214,86), up-and-right */}
        <line x1="186" y1="150" x2="214" y2="86" stroke="#1B2028" strokeWidth="25" strokeLinecap="round" />
        <line x1="186" y1="150" x2="214" y2="86" stroke="#69717A" strokeWidth="18" strokeLinecap="round" />
        <line x1="181.9" y1="148.2" x2="209.9" y2="84.2" stroke="#D8D4CA" strokeWidth="5" strokeLinecap="round" />

        {/* shoulder pivot — seated on the mount */}
        <circle cx="186" cy="150" r="11" fill="#F2B827" stroke="#1B2028" strokeWidth="3.5" />
        <circle cx="186" cy="150" r="3.5" fill="#1B2028" />

        {/* ===== FOREARM ASSEMBLY ===== */}
        <motion.g
          key={`fore-${hasBeat ? beat : idle ? "idle" : "still"}`}
          initial={{ rotate: 0 }}
          animate={foreAnim ?? { rotate: 0 }}
          transition={foreTransition}
        >
          {/* segment 2: elbow (214,86) -> wrist (150,58), sharp fold up-and-left */}
          <line x1="214" y1="86" x2="150" y2="58" stroke="#1B2028" strokeWidth="24" strokeLinecap="round" />
          <line x1="214" y1="86" x2="150" y2="58" stroke="#69717A" strokeWidth="17" strokeLinecap="round" />
          <line x1="215.8" y1="81.9" x2="151.8" y2="53.9" stroke="#D8D4CA" strokeWidth="5" strokeLinecap="round" />

          {/* DUM-E label band on the forearm (mid-segment ~ (182,72)) */}
          <rect x="171" y="67" width="22" height="9" rx="2" transform="rotate(-24 182 72)" fill="#1B2028" />
          <rect x="173.5" y="69" width="17" height="2" rx="1" transform="rotate(-24 182 72)" fill="#F8F5EE" opacity="0.85" />

          {/* elbow pivot (214,86) */}
          <circle cx="214" cy="86" r="12" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3.5" />
          <circle cx="214" cy="86" r="4" fill="#F15A37" />

          {/* ===== HAND: wrist + claw + cap, placed at the wrist (150,58) =====
               Built in the original wrist-local space then translated (+81,-18)
               from the old wrist (69,76) to the new one. */}
          <g transform="translate(81 -18)">
            {/* wrist pivot */}
            <circle cx="69" cy="76" r="11" fill="#F2B827" stroke="#1B2028" strokeWidth="3.5" />
            <circle cx="69" cy="76" r="3.5" fill="#1B2028" />

            {/* gripper head + 3-finger claw */}
            <rect x="52" y="58" width="26" height="22" rx="5" transform="rotate(-12 65 69)" fill="#D8D4CA" stroke="#1B2028" strokeWidth="3.5" />
            <g stroke="#1B2028" strokeWidth="3.5" strokeLinejoin="round" strokeLinecap="round" fill="#69717A">
              <polygon points="56,60 44,52 41,57 53,64" />
              <polygon points="58,66 46,70 49,76 60,71" />
              <polygon points="62,63 53,55 57,51 65,59" />
            </g>

            {/* ===== DUNCE CAP — upright cone, DUNCE reading vertically ===== */}
            <motion.g
              key={`cap-${hasBeat ? beat : idle ? "idle" : "still"}`}
              initial={{ rotate: 0, y: 0 }}
              animate={capAnim}
              transition={capTransition}
            >
              <g stroke="#1B2028" strokeWidth="3.5" strokeLinejoin="round" strokeLinecap="round">
                <polygon points="46,60 82,60 64,6" fill="#FBF8F1" />
                <polygon points="64,60 82,60 64,6" fill="#E6E1D5" />
                <line x1="50" y1="52" x2="78" y2="52" stroke="#F15A37" strokeWidth="3.2" />
              </g>
              <text x="65" y="38" fill="#1B2028" fontFamily="'Arial Narrow', monospace" fontWeight="700" fontSize="7" letterSpacing="0.3" textAnchor="middle" transform="rotate(90 65 38)">DUNCE</text>
            </motion.g>
          </g>
        </motion.g>
      </motion.g>
    </motion.svg>
  );
}
