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
  activePerson = 1,
  isRecording = false,
  analyser = null,
  barsCount = 16,
} = $props();

let canvas1 = $state(null);
let canvas2 = $state(null);
let animationId = null;

function drawStaticWaveform(ctx, canvas, isActive = false) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 1;
  ctx.strokeStyle = isActive ? "#000000" : "rgba(0,0,0,0.3)";
  ctx.beginPath();
  ctx.moveTo(0, canvas.height - 1);
  ctx.lineTo(canvas.width, canvas.height - 1);
  ctx.stroke();

  const numBars = barsCount || 32;
  const barWidth = canvas.width / numBars - 1;
  let x = 0;

  for (let i = 0; i < numBars; i++) {
    // Active lane displays elevated VU ticks so the user sees microphone readiness
    const barHeight = isActive ? ((i % 4 === 0) ? 5 : 2) : 1;
    ctx.fillStyle = isActive ? "#000000" : "rgba(0,0,0,0.25)";
    ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
  }
}

$effect(() => {
  if (!canvas1 || !canvas2) return;
  const ctx1 = canvas1.getContext("2d");
  const ctx2 = canvas2.getContext("2d");

  if (!isRecording || !analyser) {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    drawStaticWaveform(ctx1, canvas1, activePerson === 1);
    drawStaticWaveform(ctx2, canvas2, activePerson === 2);
    return;
  }

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);

  function draw() {
    animationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(dataArray);

    const activeCtx = activePerson === 1 ? ctx1 : ctx2;
    const activeCanvas = activePerson === 1 ? canvas1 : canvas2;
    const inactiveCtx = activePerson === 1 ? ctx2 : ctx1;
    const inactiveCanvas = activePerson === 1 ? canvas2 : canvas1;

    drawStaticWaveform(inactiveCtx, inactiveCanvas);

    activeCtx.clearRect(0, 0, activeCanvas.width, activeCanvas.height);
    activeCtx.strokeStyle = "#000000";
    activeCtx.lineWidth = 1;
    activeCtx.beginPath();
    activeCtx.moveTo(0, activeCanvas.height);
    activeCtx.lineTo(activeCanvas.width, activeCanvas.height);
    activeCtx.stroke();

    const numBars = barsCount || 32;
    const barWidth = activeCanvas.width / numBars - 1;
    let x = 0;

    // Ignore the top quarter of the spectrum (mostly empty for speech).
    const maxBin = Math.floor(bufferLength * 0.75);
    const binsPerBar = Math.floor(maxBin / numBars);

    for (let i = 0; i < numBars; i++) {
      let sum = 0;
      for (let j = 0; j < binsPerBar; j++) {
        sum += dataArray[i * binsPerBar + j];
      }
      let average = sum / binsPerBar;
      let barHeight = (average / 255.0) * activeCanvas.height;

      activeCtx.fillStyle = "#000000";
      activeCtx.fillRect(
        x,
        activeCanvas.height - barHeight,
        barWidth,
        barHeight,
      );
      x += barWidth + 1;
    }
  }

  draw();

  return () => {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
  };
});
</script>

<section class="visualizer-divider" style="display: flex;">
  <div class="canvas-wrap lane-one" style="width: 50%;">
    <canvas
      bind:this={canvas1}
      width="240"
      height="28"
      style="display: block; width: 100%;"
    ></canvas>
  </div>
  <div
    class="canvas-wrap lane-two"
    style="width: 50%; position: relative;"
  >
    <canvas
      bind:this={canvas2}
      width="240"
      height="28"
      style="display: block; width: 100%;"
    ></canvas>
  </div>
</section>
