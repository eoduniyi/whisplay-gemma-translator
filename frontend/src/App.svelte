<script>
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

import TranslatorApp from "./TranslatorApp.svelte"
import SettingsOverlay from "./components/SettingsOverlay.svelte"
import { testConnectionAPI } from "./utils/api"

// App shell: owns the global config (keyboardMode and themeColor persist in
// localStorage), the settings overlay, and applies the theme by overriding
// the --bg-black CSS variable.
let config = $state({
  endpointUrl: "http://localhost:9379/v1",
  modelName: "gemma4-e2b",
  apiKey: "",
  keyboardMode: localStorage.getItem("keyboardMode") || "landscape",
  useProxy: true,
  enableTts: true,
  visualizerBars: 16,
  systemPrompt: "Translator mode",
  themeColor: localStorage.getItem("themeColor") || "#ffa500",
  lane1Language: localStorage.getItem("lane1Language") || "zh",
  lane2Language: localStorage.getItem("lane2Language") || "en",
});

let isSettingsOpen = $state(false);

async function testConnection() {
  try {
    await testConnectionAPI(
      config.endpointUrl,
      config.useProxy,
      config.apiKey,
    );
  } catch (err) {
    // Ignored: API test failure
  }
}

$effect(() => {
  testConnection();
});

$effect(() => {
  localStorage.setItem("keyboardMode", config.keyboardMode);
});

$effect(() => {
  if (config.themeColor) {
    document.documentElement.style.setProperty('--bg-black', config.themeColor);
    localStorage.setItem("themeColor", config.themeColor);
  }
});

$effect(() => {
  localStorage.setItem("lane1Language", config.lane1Language);
  localStorage.setItem("lane2Language", config.lane2Language);

  fetch("/api/languages", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      languages: [config.lane1Language, config.lane2Language],
    }),
  }).catch((err) => {
    console.debug("Language preparation unavailable:", err);
  });
});
</script>

<div class="app-container">
  <div class="config-wrap-right" style="opacity: 1;">
    <button
      class="config-toggle-btn"
      onclick={() => (isSettingsOpen = true)}
      title="Settings"
    >
      ⚙️
    </button>
  </div>

  <div style="height: 100%;">
    <TranslatorApp bind:config />
  </div>

  <SettingsOverlay
    isActive={isSettingsOpen}
    onClose={() => (isSettingsOpen = false)}
    bind:config
    onTestConnection={testConnection}
  />
</div>
