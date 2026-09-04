import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';

// Results come back from devices nobody here controls, pasted as JSON. Without a build stamp
// there is no way to tell a stale page from a current one, and we already lost a round to that:
// an iPhone report was read as a failure of the new diagnostics when it was simply the previous
// deploy still being served.
const buildId = (process.env.GITHUB_SHA
  ?? (() => { try { return execSync('git rev-parse HEAD').toString(); } catch { return 'dev'; } })()
).trim().slice(0, 7);
const buildAt = new Date().toISOString();

// The bench page is the site. The engine is the library it drives.
// `base` is set for GitHub Pages, which serves from /<repo>/.
export default defineConfig({
  base: process.env.PAGES_BASE ?? '/',
  define: {
    __BUILD_ID__: JSON.stringify(buildId),
    __BUILD_AT__: JSON.stringify(buildAt),
  },
  build: { target: 'es2022', outDir: 'dist' },
});
