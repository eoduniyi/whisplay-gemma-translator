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

let {
  isActive = false,
  onClose,
  transcriptionSource = "",
  transcriptionText = "",
  translationTarget = "",
  translationText = "",
  metaText = "",
  timing = null,
  placeholderText = "Select languages, push to talk",
} = $props();

let hasData = $derived(isActive || transcriptionSource !== "");

function formatTimingValue(value) {
  if (typeof value === "number") return `${value.toFixed(2)}s`;
  return value || "--";
}
</script>

<div
  class="response-drawer {isActive ? 'active' : ''}"
  id="response-drawer"
>
  <div class="app-title">Gemma Translator</div>
  <button
    type="button"
    class="drawer-handle"
    onclick={onClose}
    style="display: none;"
    aria-label="Close drawer"
  ></button>
  <div class="chat-panel">
    {#if !hasData}
      <div class="initial-placeholder">
        {placeholderText}
      </div>
    {:else}
      <div class="chat-row chat-row-left">
        <div class="chat-bubble bubble-left">
          <div class="bubble-label">
            {transcriptionSource || "LANG 1"}
          </div>
          <div class="bubble-text">
            {transcriptionText || "— listening —"}
          </div>
        </div>
      </div>
      <div class="chat-row chat-row-right">
        <div class="chat-bubble bubble-right">
          <div class="bubble-label">
            {translationTarget || "LANG 2"}
          </div>
          <div class="bubble-text">
            {translationText || "— waiting —"}
          </div>
        </div>
      </div>
      <div class="chat-meta">
        {#if timing}
          <div class="timing-item {timing.stt === 'loading' ? 'loading' : ''}">
            <span class="timing-label">STT</span>
            <span class="timing-value">
              {#if timing.stt === 'loading'}
                <span class="timing-dots" aria-label="loading">
                  <span></span><span></span><span></span>
                </span>
              {:else}
                {formatTimingValue(timing.stt)}
              {/if}
            </span>
          </div>

          <div class="timing-item {timing.translate === 'loading' ? 'loading' : ''} {typeof timing.translate === 'number' && timing.translate < 0.02 ? 'neu-fast' : ''}">
            <span class="timing-label">{typeof timing.translate === 'number' && timing.translate < 0.02 ? 'Neu Fast' : 'LLM 1st'}</span>
            <span class="timing-value">
              {#if timing.translate === 'loading'}
                <span class="timing-dots" aria-label="loading">
                  <span></span><span></span><span></span>
                </span>
              {:else if typeof timing.translate === 'number' && timing.translate < 0.02}
                &lt; 1ms
              {:else}
                {formatTimingValue(timing.translate)}
              {/if}
            </span>
          </div>

          <div class="timing-item {timing.tts === 'loading' ? 'loading' : ''}">
            <span class="timing-label">TTS</span>
            <span class="timing-value">
              {#if timing.tts === 'loading'}
                <span class="timing-dots" aria-label="loading">
                  <span></span><span></span><span></span>
                </span>
              {:else}
                {formatTimingValue(timing.tts)}
              {/if}
            </span>
          </div>

          {#if metaText}
            <span class="timing-total">{metaText}</span>
          {/if}
        {:else if metaText}
          {metaText}
        {/if}
      </div>
    {/if}
  </div>
</div>
