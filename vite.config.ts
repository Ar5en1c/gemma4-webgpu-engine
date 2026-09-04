import { defineConfig } from 'vite';

// The bench page is the site. The engine is the library it drives.
// `base` is set for GitHub Pages, which serves from /<repo>/.
export default defineConfig({
  base: process.env.PAGES_BASE ?? '/',
  build: { target: 'es2022', outDir: 'dist' },
});
