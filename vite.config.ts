import { defineConfig } from 'vite';

// Multi-threaded WASM needs a cross-origin isolated page. Locally these headers
// do it; on GitHub Pages public/coi-sw.js adds them instead.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  // Relative asset paths, so the build works under https://<user>.github.io/<repo>/.
  base: './',
  server: { headers: isolation },
  preview: { headers: isolation },
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  build: { target: 'es2022' },
});
