import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const STATION = "http://127.0.0.1:8788";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  // Relative asset URLs: the bundle is served off disk by the station, and
  // it should not care what path it is mounted under.
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
  server: {
    port: 5181,
    // The dev server is cross-origin from the station, so it proxies every
    // path the console asks for. `/mission` covers events, next, tasking,
    // clear, ownship and prior.json; `/missions` covers the run list.
    proxy: {
      "/missions": STATION,
      "/mission": STATION,
      "/reasoner": STATION,
      "/replay": STATION,
      "/location": STATION,
      "/tiles": STATION,
      "/area": STATION,
      "/run": STATION,
    },
  },
});
