// Twitter/X share card. Reuses the Open Graph image's renderer so both unfurl
// identically — one source of truth for the design, no duplicated JSX.
//
// `runtime` must be a literal in THIS file: Next statically analyses the route
// segment config and can't follow it through a re-export (it would silently
// fall back to the Node runtime and warn). The alt/size/contentType metadata
// and the image renderer itself are safe to re-export.
export const runtime = "edge";
export { alt, size, contentType } from "./opengraph-image";
export { default } from "./opengraph-image";
