/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    // Proxy API + WS to the FastAPI backend so the browser is same-origin.
    const backend = process.env.BACKEND_URL || "http://127.0.0.1:8000";
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};
export default nextConfig;
