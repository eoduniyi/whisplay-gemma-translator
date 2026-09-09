/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import net from "net";

// Helper to check if a port is free on a given interface
function checkInterface(port, host) {
  return new Promise((resolve) => {
    const server = net.createServer();
    server.unref();
    server.once("error", () => resolve(false));
    try {
      server.listen(port, host, () => {
        server.close(() => resolve(true));
      });
    } catch {
      resolve(false);
    }
  });
}

// Ensure the port is free across IPv4, IPv6 localhost, and wildcard 0.0.0.0
// to prevent macOS dual-stack collision where localhost (::1) is occupied by
// another dev server while 0.0.0.0:port appears free.
async function isPortAvailable(port) {
  const hosts = ["127.0.0.1", "::1", "0.0.0.0"];
  for (const host of hosts) {
    const free = await checkInterface(port, host);
    if (!free) return false;
  }
  return true;
}

async function findAvailablePort(startPort = 5173, maxScan = 100) {
  for (let p = startPort; p < startPort + maxScan; p++) {
    if (await isPortAvailable(p)) {
      return p;
    }
  }
  return startPort;
}

export default defineConfig(async () => {
  const preferredPort = parseInt(
    process.env.PORT || process.env.TRANSLATOR_WEB_PORT || process.env.VITE_PORT || "5173",
    10,
  );
  const webPort = await findAvailablePort(preferredPort);
  const backendPort = process.env.TRANSLATOR_PORT || "3000";

  return {
    plugins: [svelte()],
    server: {
      host: "0.0.0.0",
      port: webPort,
      strictPort: false,
      proxy: {
        "/api": {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
        },
        "/proxy": {
          target: `http://localhost:${backendPort}`,
          changeOrigin: true,
        },
      },
    },
  };
});
