import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Deployed to GitHub Pages under /icu-capacity-forecasting/, so assets must be
// requested from that subpath rather than the domain root.
export default defineConfig({
  base: '/icu-capacity-forecasting/',
  plugins: [react()],
})
