import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  base: './',
  resolve: { alias: { '@': '/src' } },
  server: {
    host: true,
    port: 5175,              // <- your dev port
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://localhost:4000', // <- your Node API port
        changeOrigin: true,
        secure: false,
      },
    },
  },
  preview: {
    port: 4175,
    proxy: {
      '/api': {
        target: 'http://localhost:4000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
});