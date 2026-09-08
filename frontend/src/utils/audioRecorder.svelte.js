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

// Mic capture controller for Svelte 5: records raw Float32 PCM via Web Audio,
// resamples to 16 kHz mono, and returns it base64-encoded — the exact payload
// format expected by POST /api/stt (backend/server.py).
export class AudioRecorder {
  isRecording = $state(false)
  micError = $state(null)
  analyser = $state(null)

  #audioContext = null
  #source = null
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

      const scriptProcessor = this.#audioContext.createScriptProcessor(4096, 1, 1)
      this.#scriptProcessor = scriptProcessor
      this.#recordedSamples = []

      scriptProcessor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0)
        this.#recordedSamples.push(new Float32Array(inputData))
      }

      source.connect(scriptProcessor)
      scriptProcessor.connect(this.#audioContext.destination)

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
