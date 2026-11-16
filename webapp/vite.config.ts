import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the gateway runs on :8020 and every API call is under /api, so one proxy
// rule covers it. Production: `npm run build` and the gateway serves dist/ from
// the same origin, so no proxy and no CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:8020' },
  },
})
