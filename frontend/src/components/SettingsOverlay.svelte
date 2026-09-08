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

import { playBlip } from "../utils/audio-blip"

let {
  isActive = false,
  onClose,
  config = $bindable(),
  onTestConnection,
} = $props();

const LANGUAGE_OPTIONS = [
  { code: "zh", name: "Chinese" },
  { code: "en", name: "English" },
  { code: "ar", name: "Arabic" },
  { code: "es", name: "Spanish" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
];

const THEME_COLORS = [
  { name: "RED", value: "#ff4444" },
  { name: "WHITE", value: "#ffffff" },
  { name: "YELLOW", value: "#ffeb3b" },
  { name: "BLUE", value: "#2196f3" },
  { name: "GREEN", value: "#4caf50" },
  { name: "ORANGE", value: "#ffa500" },
];

let systemVolume = $state(null);
let languageStatus = $state({});

async function refreshLanguageStatus() {
  try {
    const res = await fetch('/api/languages', { cache: 'no-store' });
    const data = await res.json();
    const nextStatus = {};
    for (const language of data.languages || []) {
      nextStatus[language.code] = language.status;
    }
    languageStatus = nextStatus;
  } catch (e) {
    console.error("Failed to fetch languages", e);
  }
}

$effect(() => {
  if (isActive) {
    fetch('/api/volume')
      .then((res) => res.json())
      .then((data) => {
        if (data.volume !== undefined && data.volume !== null) {
          systemVolume = data.volume;
        }
      })
      .catch((e) => console.error("Failed to fetch volume", e));
    refreshLanguageStatus();

    const timer = setInterval(refreshLanguageStatus, 300);
    return () => clearInterval(timer);
  }
});

function handleChange(key, value) {
  config[key] = value;
}

async function prepareLanguages(languages) {
  try {
    const res = await fetch('/api/languages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ languages }),
    });
    const data = await res.json();
    const nextStatus = {};
    for (const language of data.languages || []) {
      nextStatus[language.code] = language.status;
    }
    languageStatus = nextStatus;
    setTimeout(refreshLanguageStatus, 1200);
  } catch (e) {
    console.error("Failed to prepare languages", e);
  }
}

function handleLanguageChange(key, value) {
  if (key === "lane1Language" && value === config.lane2Language) {
    config.lane2Language = config.lane1Language;
  }
  if (key === "lane2Language" && value === config.lane1Language) {
    config.lane1Language = config.lane2Language;
  }
  config[key] = value;
  prepareLanguages([config.lane1Language, config.lane2Language]);
}

function languageNameFor(code) {
  return LANGUAGE_OPTIONS.find((language) => language.code === code)?.name || code;
}

async function handleVolumeChange(action) {
  try {
    const res = await fetch('/api/volume', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action }),
    });
    const data = await res.json();
    if (data.volume !== undefined && data.volume !== null) {
      systemVolume = data.volume;
    }
    playBlip("ping");
  } catch (e) {
    console.error('Failed to change volume:', e);
  }
}

async function handleExitApp() {
  try {
    await fetch('/api/kiosk/exit', { method: 'POST' });
  } catch (e) {
    console.error('Failed to exit app:', e);
  }
}
</script>

<!-- svelte-ignore a11y_label_has_associated_control -->
{#if isActive}
  <div
    class="settings-overlay active"
    style="display: flex;"
  >
    <header class="overlay-header">
      <h2>Developer Settings</h2>
      <button class="overlay-close-btn" onclick={onClose}>
        ✕
      </button>
    </header>
    <div class="overlay-body">
      <div class="form-group" style="margin-bottom: 10px;">
        <label>System Volume</label>
        <div style="display: flex; gap: 8px; align-items: center;">
          <button
            class="overlay-btn"
            style="flex: 1;"
            onclick={() => handleVolumeChange('down')}
          >
            -
          </button>
          <span style="min-width: 40px; text-align: center; font-weight: bold;">
            {systemVolume !== null ? `${systemVolume}%` : "--"}
          </span>
          <button
            class="overlay-btn"
            style="flex: 1;"
            onclick={() => handleVolumeChange('up')}
          >
            +
          </button>
        </div>
      </div>

      <div class="form-group">
        <label>Languages</label>
        <div class="language-select-row">
          <select
            value={config.lane1Language}
            onchange={(e) => handleLanguageChange("lane1Language", e.target.value)}
          >
            {#each LANGUAGE_OPTIONS as language (language.code)}
              <option value={language.code}>
                1 · {language.name}
              </option>
            {/each}
          </select>
          <select
            value={config.lane2Language}
            onchange={(e) => handleLanguageChange("lane2Language", e.target.value)}
          >
            {#each LANGUAGE_OPTIONS as language (language.code)}
              <option value={language.code}>
                2 · {language.name}
              </option>
            {/each}
          </select>
        </div>
        <div class="language-progress-list">
          {#each [config.lane1Language, config.lane2Language] as code (code)}
            {@const current = languageStatus[code] || {}}
            {@const status = current.status || "not-installed"}
            {@const progress = Number.isFinite(current.progress) ? current.progress : 0}
            {@const stage = current.stage || status}
            <div class="language-progress-item">
              <div class="language-progress-meta">
                <span>{code.toUpperCase()} · {languageNameFor(code)}</span>
                <span>{status === "ready" ? "100%" : `${progress}%`}</span>
              </div>
              <div class="language-progress-track {status}">
                <div
                  class="language-progress-fill"
                  style="width: {Math.max(0, Math.min(100, progress))}%;"
                ></div>
              </div>
              <div class="language-progress-stage">
                {stage}
              </div>
            </div>
          {/each}
        </div>
      </div>

      <div class="form-group">
        <label>Theme Color</label>
        <div style="display: flex; gap: 10px; margin-top: 5px;">
          {#each THEME_COLORS as c (c.name)}
            <button
              onclick={() => handleChange("themeColor", c.value)}
              title={c.name}
              style="width: 30px; height: 30px; border-radius: 50%; background-color: {c.value}; border: {config.themeColor === c.value ? '2px solid #000' : '2px solid transparent'}; box-shadow: {config.themeColor === c.value ? '0 0 0 2px #fff' : 'none'}; cursor: pointer; padding: 0;"
            ></button>
          {/each}
        </div>
      </div>

      <div class="form-group">
        <label>API Endpoint</label>
        <div class="input-inline">
          <input
            type="text"
            bind:value={config.endpointUrl}
          />
          <button class="overlay-btn btn-sm" onclick={onTestConnection}>
            Test
          </button>
        </div>
      </div>

      <div class="form-group">
        <label>Model Name</label>
        <input
          type="text"
          bind:value={config.modelName}
          list="model-suggestions"
        />
        <datalist id="model-suggestions">
          <option value="gemma4-e2b"></option>
          <option value="gemma-4-2b"></option>
        </datalist>
      </div>

      <div class="form-group">
        <label>API Key</label>
        <input
          type="password"
          placeholder="Optional api key"
          bind:value={config.apiKey}
        />
      </div>

      <div class="form-group">
        <label>Keyboard Mode</label>
        <select bind:value={config.keyboardMode}>
          <option value="landscape">
            Landscape — active person (Space / Z / ← →)
          </option>
          <option value="vertical">
            Vertical — two-hand (Z / X / ← → / − +)
          </option>
        </select>
      </div>

      <div class="form-row-checkboxes">
        <label class="checkbox-container">
          <input
            type="checkbox"
            bind:checked={config.enableTts}
          />
          <span class="checkbox-label">Enable Speech Output</span>
        </label>
      </div>

      <div class="form-group">
        <label>Visualizer</label>
        <div class="slider-row">
          <div class="slider-group">
            <label>
              Bars: <span>{config.visualizerBars}</span>
            </label>
            <input
              type="range"
              min="8"
              max="128"
              step="8"
              bind:value={config.visualizerBars}
            />
          </div>
        </div>
      </div>

      <button class="overlay-btn" onclick={handleExitApp}>
        Exit App
      </button>
    </div>
  </div>
{/if}
