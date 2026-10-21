import { fileURLToPath } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL(".", import.meta.url)) },
  },
  // Relative asset URLs: baselines.human.study_server serves the bundle off disk.
  base: "./",
  build: { outDir: "dist", emptyOutDir: true },
});
