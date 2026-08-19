import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { copyFileSync, mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectDirectory = path.dirname(fileURLToPath(import.meta.url));
const runtimePublicFiles = [
  "assets/inter.woff2",
  "assets/space-grotesk.woff2",
  "assets/brain/destrieux-fsaverage5.u8",
  "assets/brain/destrieux-fsaverage5.json",
  "assets/brain/fsaverage5-half.bin",
  "assets/brain/fsaverage5-half.json",
];

function copyVerifiedPublicAssets() {
  return {
    name: "copy-verified-public-assets",
    closeBundle() {
      for (const relativePath of runtimePublicFiles) {
        const source = path.join(projectDirectory, "public", relativePath);
        const destination = path.join(projectDirectory, "dist/client", relativePath);
        mkdirSync(path.dirname(destination), { recursive: true });
        copyFileSync(source, destination);
      }
    },
  };
}

export default defineConfig({
  build: {
    outDir: "dist/client",
    // Copy only the audited, runtime-referenced public assets below.
    copyPublicDir: false,
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: ["terminal.local"],
    watch: {
      ignored: ["**/backend/.venv/**", "**/backend/.runtime/**", "**/backend/__pycache__/**"],
    },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [react(), copyVerifiedPublicAssets()],
});
