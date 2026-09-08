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

import { getMergedSamples, resample, blobToBase64 } from "./audioHelpers"

// Inline audio worklet for real-time, low-latency audio capture off the main thread.
const WORKLET_CODE = `
class WhisperAudioProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0].length > 0) {
      // Must clone channel data since Web Audio reuses the Float32Array buffer each quantum
      this.port.postMessage(new Float32Array(input[0]));
    }
    return true;
  }
}
registerProcessor('whisper-audio-processor', WhisperAudioProcessor);
`;

// Mic capture controller for Svelte 5: records raw Float32 PCM via Web Audio,
// resamples to 16 kHz mono, and returns it base64-encoded — the exact payload
// format expected by POST /api/stt (backend/server.py).
export class AudioRecorder {
  isRecording = $state(false)
  micError = $state(null)
  analyser = $state(null)

  #audioContext = null
  #source = null
  #workletNode = null
  #scriptProcessor = null
  #stream = null
  #recordedSamples = []

  async startRecording() {
    if (this.isRecording) return false
    this.micError = null
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      })

      this.#stream = stream

      const AudioContext = window.AudioContext || window.webkitAudioContext
      this.#audioContext = new AudioContext()
      if (this.#audioContext.state === "suspended") {
        await this.#audioContext.resume()
      }
      const source = this.#audioContext.createMediaStreamSource(stream)
      this.#source = source

      // Small FFT — the analyser only feeds the low-res bar visualizer.
      const analyser = this.#audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      this.analyser = analyser

      this.#recordedSamples = []

      // Prefer AudioWorkletNode (dedicated real-time audio thread);
      // fallback to ScriptProcessorNode on legacy runtimes.
      let useWorklet = false
      if (this.#audioContext.audioWorklet && typeof AudioWorkletNode !== "undefined") {
        try {
          const blob = new Blob([WORKLET_CODE], { type: "application/javascript" })
          const workletUrl = URL.createObjectURL(blob)
          await this.#audioContext.audioWorklet.addModule(workletUrl)
          URL.revokeObjectURL(workletUrl)

          const workletNode = new AudioWorkletNode(
            this.#audioContext,
            "whisper-audio-processor",
          )
          workletNode.port.onmessage = (e) => {
            this.#recordedSamples.push(e.data)
          }

          source.connect(workletNode)
          workletNode.connect(this.#audioContext.destination)
          this.#workletNode = workletNode
          useWorklet = true
        } catch (workletErr) {
          console.warn("AudioWorklet init failed; using ScriptProcessor fallback:", workletErr)
        }
      }

      if (!useWorklet) {
        const scriptProcessor = this.#audioContext.createScriptProcessor(4096, 1, 1)
        this.#scriptProcessor = scriptProcessor
        scriptProcessor.onaudioprocess = (e) => {
          const inputData = e.inputBuffer.getChannelData(0)
          this.#recordedSamples.push(new Float32Array(inputData))
        }
        source.connect(scriptProcessor)
        scriptProcessor.connect(this.#audioContext.destination)
      }

      this.isRecording = true
      return true
    } catch (err) {
      console.error("Error accessing microphone:", err)
      const msg = err.message || "Microphone access failed (HTTPS required for remote devices)"
      this.micError = msg
      return false
    }
  }

  async stopRecording() {
    if (!this.isRecording) return null

    this.isRecording = false
    if (this.#stream) {
      this.#stream.getTracks().forEach((track) => track.stop())
      this.#stream = null
    }

    if (this.#workletNode) {
      this.#workletNode.disconnect()
      this.#workletNode.port.onmessage = null
      this.#workletNode = null
    }

    if (this.#scriptProcessor) {
      this.#scriptProcessor.disconnect()
      this.#scriptProcessor.onaudioprocess = null
      this.#scriptProcessor = null
    }
    if (this.#source) {
      this.#source.disconnect()
      this.#source = null
    }
    if (this.analyser) {
      this.analyser.disconnect()
    }

    const actualSampleRate = this.#audioContext?.sampleRate || 16000
    if (this.#audioContext && this.#audioContext.state !== "closed") {
      await this.#audioContext.close()
      this.#audioContext = null
    }

    if (this.#recordedSamples.length === 0) {
      console.warn("No audio samples recorded")
      return null
    }

    const mergedSamples = getMergedSamples(this.#recordedSamples)
    this.#recordedSamples = []
    const targetSampleRate = 16000 // Moonshine STT expects 16 kHz mono
    const resampledSamples = resample(
      mergedSamples,
      actualSampleRate,
      targetSampleRate,
    )
    const rawBlob = new Blob([resampledSamples.buffer], {
      type: "application/octet-stream",
    })

    try {
      const base64Data = await blobToBase64(rawBlob)
      return { rawBlob, base64Data }
    } catch (err) {
      console.error("Base64 encoding failed:", err)
      return null
    }
  }

  destroy() {
    if (this.#audioContext && this.#audioContext.state !== "closed") {
      this.#audioContext.close()
    }
  }
}
