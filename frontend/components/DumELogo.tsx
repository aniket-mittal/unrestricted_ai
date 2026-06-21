export default function DumELogo({ className = "h-11 w-11" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 48 48"
      fill="none"
      aria-label="DUM-E: an industrial robot arm wearing a dunce cap, reaching with a claw"
    >
      {/* wheeled base */}
      <path
        d="M14 41.5 H32"
        stroke="currentColor"
        strokeWidth="4.6"
        strokeLinecap="round"
      />
      <circle cx="17.5" cy="43.8" r="1.6" fill="currentColor" />
      <circle cx="28.5" cy="43.8" r="1.6" fill="currentColor" />

      {/* shoulder column from base up to the elbow joint */}
      <path
        d="M26 41.5 V31.5"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />

      {/* amber hydraulic elbow joint — the one warm accent */}
      <circle
        cx="26"
        cy="29"
        r="4.6"
        fill="#F2B827"
        stroke="currentColor"
        strokeWidth="2.4"
      />

      {/* forearm segment angling up-left from elbow toward the wrist/head */}
      <path
        d="M23.5 25.5 L15 16"
        stroke="#69717A"
        strokeWidth="6.6"
        strokeLinecap="round"
      />

      {/* wrist / head block at the end of the forearm */}
      <circle
        cx="14"
        cy="15"
        r="4.4"
        fill="#69717A"
        stroke="currentColor"
        strokeWidth="2.4"
      />

      {/* three-fingered claw reaching out to the upper-left */}
      <path
        d="M11.4 12 L6.5 8.5 M10 14 L4.5 12.5 M11 16.5 L6 20"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* dunce cap perched jauntily on the head — the signature identifier */}
      <path
        d="M14 9 L19 1.5 L23.5 7.5 Z"
        fill="#F8F5EE"
        stroke="currentColor"
        strokeWidth="2.4"
        strokeLinejoin="round"
        transform="rotate(-12 18 5)"
      />
    </svg>
  );
}
