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
  laneId,
  laneLabel,
  languages,
  currentIndex,
  isRecording = false,
  isActivePerson = false,
  onRotate,
  onSelect,
  onPressStart,
  onPressEnd,
} = $props();

let drumEl = $state(null);
let longPressTimer = null;
let touchRecording = false;

$effect(() => {
  if (drumEl) {
    drumEl.style.transform = `translateY(-${currentIndex * 24}px)`;
  }
});

function handlePrev(e) {
  e.preventDefault();
  if (!isRecording) onRotate?.(-1);
}

function handleNext(e) {
  e.preventDefault();
  if (!isRecording) onRotate?.(1);
}

function clearLongPressTimer() {
  if (longPressTimer) {
    clearTimeout(longPressTimer);
    longPressTimer = null;
  }
}

function handlePressDown(e) {
  if (e.pointerType === "mouse" && e.button !== 0) return;
  e.preventDefault();
  e.currentTarget?.setPointerCapture?.(e.pointerId);
  onSelect?.();
  clearLongPressTimer();
  touchRecording = false;
  longPressTimer = setTimeout(() => {
    touchRecording = true;
    onPressStart?.();
  }, 320);
}

function handlePressUp(e) {
  e.preventDefault();
  clearLongPressTimer();
  if (touchRecording) {
    touchRecording = false;
    onPressEnd?.();
  }
}

function handlePressCancel() {
  clearLongPressTimer();
  if (touchRecording) {
    touchRecording = false;
    onPressEnd?.();
  }
}
</script>

<!-- svelte-ignore a11y_no_noninteractive_element_to_interactive_role a11y_no_static_element_interactions -->
<section
  class="language-lane lane-{laneId === 1 ? 'one' : 'two'} {isRecording ? 'recording' : ''} {isActivePerson ? 'selected' : ''}"
  id="lane-{laneId}"
  role="button"
  tabindex="0"
  aria-label="{languages[currentIndex]?.name || ''} push to talk"
  onpointerdown={handlePressDown}
  onpointerup={handlePressUp}
  onpointercancel={handlePressCancel}
  onlostpointercapture={handlePressCancel}
  oncontextmenu={(e) => e.preventDefault()}
>
  <div class="lane-header">
    <span class="lane-label">{laneLabel}</span>
  </div>
  <div class="revolver-stage">
    <button
      class="rotator-arrow"
      title="Previous language"
      onclick={handlePrev}
    >
      ◀
    </button>
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="revolver-viewport"
      oncontextmenu={(e) => e.preventDefault()}
    >
      <div class="revolver-drum" bind:this={drumEl}>
        {#each languages as l (l.code)}
          <div class="revolver-item">
            {l.name.split(" ")[0].toUpperCase()}
          </div>
        {/each}
      </div>
    </div>
    <button
      class="rotator-arrow"
      title="Next language"
      onclick={handleNext}
    >
      ▶
    </button>
  </div>
  <span class="language-name" style="display: none;">
    {languages[currentIndex]?.name || ''}
  </span>
  <div class="corner-brackets"></div>
</section>
