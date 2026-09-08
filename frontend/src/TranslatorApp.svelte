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

import LanguageLane from "./components/LanguageLane.svelte"
import ResponseDrawer from "./components/ResponseDrawer.svelte"
import Visualizer from "./components/Visualizer.svelte"
import { AudioRecorder } from "./utils/audioRecorder.svelte"
import {
  transcribeAudio,
  translateTextStream,
  extractSpeechChunks,
} from "./utils/api"
import { playBlip } from "./utils/audio-blip"

let { config = $bindable() } = $props();

const AVAILABLE_LANGUAGES = [
  { code: "zh", name: "Chinese", voice: "tts", ttsLang: "zh" },
  { code: "en", name: "English", voice: "tts", ttsLang: "en" },
  { code: "ar", name: "Arabic", voice: "tts", ttsLang: "ar" },
  { code: "es", name: "Spanish", voice: "tts", ttsLang: "es" },
  { code: "ja", name: "Japanese", voice: "tts", ttsLang: "ja" },
  { code: "ko", name: "Korean", voice: "tts", ttsLang: "ko" },
];

function languageIndexFor(code, fallbackIndex) {
  const index = AVAILABLE_LANGUAGES.findIndex((language) => language.code === code);
  return index >= 0 ? index : fallbackIndex;
}

let isDrawerOpen = $state(false);
let activePerson = $state(1);

let transcriptionData = $state({
  source: "",
  text: "— listening —",
});

let translationData = $state({
  target: "",
  text: "— waiting —",
});

let metaText = $state("");
let timing = $state(null);

let onlineAudioPlayer = null;
let speechSession = null;

let lang1Index = $state(languageIndexFor(config.lane1Language, 0));
let lang2Index = $state(languageIndexFor(config.lane2Language, 1));
let activeLaneRecording = $state(null); // 1 or 2

let hardwareButtonPressed = false;

const recorder = new AudioRecorder();

$effect(() => {
  if (recorder.micError) {
    isDrawerOpen = true;
    transcriptionData = { source: "Microphone", text: "Access Failed" };
    translationData = {
      target: "Error",
      text: `${recorder.micError} (HTTPS is required when accessing from remote devices)`,
    };
  }
});

$effect(() => {
  lang1Index = languageIndexFor(config.lane1Language, 0);
});

$effect(() => {
  lang2Index = languageIndexFor(config.lane2Language, 1);
});

$effect(() => {
  const preventContextMenu = (e) => e.preventDefault();
  window.addEventListener("contextmenu", preventContextMenu);
  return () => window.removeEventListener("contextmenu", preventContextMenu);
});

function stopSpeaking() {
  if (speechSession) {
    speechSession.cancelled = true;
    speechSession.controllers.forEach((controller) => controller.abort());
    speechSession = null;
  }
  if (onlineAudioPlayer) {
    onlineAudioPlayer.pause();
    if (onlineAudioPlayer.dataset.objectUrl) {
      URL.revokeObjectURL(onlineAudioPlayer.dataset.objectUrl);
    }
    onlineAudioPlayer = null;
  }
}

function createSpeechSession(targetLang, onSynthesisReady) {
  stopSpeaking();
  const session = {
    cancelled: false,
    controllers: new Set(),
    playChain: Promise.resolve(),
    synthesisMs: 0,
  };
  speechSession = session;

  session.enqueue = (text) => {
    if (!text.trim() || session.cancelled) return;
    const controller = new AbortController();
    session.controllers.add(controller);
    const startedAt = performance.now();
    // Start synthesis immediately. Playback is chained separately so later
    // sentences can be synthesized while the first sentence is speaking.
    const synthesis = fetch(
      `/api/tts?text=${encodeURIComponent(text)}&lang=${encodeURIComponent(targetLang)}`,
      { cache: "no-store", signal: controller.signal },
    ).then(async (response) => {
      session.controllers.delete(controller);
      if (!response.ok) throw new Error(`TTS failed: ${response.status}`);
      const blob = await response.blob();
      session.synthesisMs += performance.now() - startedAt;
      onSynthesisReady?.(session.synthesisMs);
      return blob;
    });

    session.playChain = session.playChain.then(async () => {
      const blob = await synthesis;
      if (session.cancelled) return;
      const objectUrl = URL.createObjectURL(blob);
      const player = new Audio(objectUrl);
      player.volume = 1.0;
      player.dataset.objectUrl = objectUrl;
      onlineAudioPlayer = player;
      await new Promise((resolve, reject) => {
        player.onended = resolve;
        player.onerror = () => reject(new Error("TTS playback failed"));
        player.play().catch(reject);
      }).finally(() => {
        URL.revokeObjectURL(objectUrl);
        if (onlineAudioPlayer === player) onlineAudioPlayer = null;
      });
    });
  };
  return session;
}

function handleRotateLanguage(lane, direction) {
  if (recorder.isRecording) return;
  const N = AVAILABLE_LANGUAGES.length;
  playBlip("language");

  if (lane === 1) {
    let ni = (lang1Index + direction + N) % N;
    if (ni === lang2Index) ni = (ni + direction + N) % N;
    lang1Index = ni;
    config.lane1Language = AVAILABLE_LANGUAGES[ni].code;
  } else {
    let ni = (lang2Index + direction + N) % N;
    if (ni === lang1Index) ni = (ni + direction + N) % N;
    lang2Index = ni;
    config.lane2Language = AVAILABLE_LANGUAGES[ni].code;
  }
}

async function handleRecordStart(lane) {
  stopSpeaking();

  if (activePerson !== lane) {
    playBlip("speaker");
    activePerson = lane;
  }

  const ok = await recorder.startRecording();
  if (ok) {
    activeLaneRecording = lane;
    playBlip("ping");
  } else {
    activeLaneRecording = null;
  }
}

async function handleRecordStop() {
  const recordedLane = activeLaneRecording;
  activeLaneRecording = null;
  const audioData = await recorder.stopRecording();

  if (audioData && recordedLane) {
    processTranslation(recordedLane, audioData.base64Data);
  }
}

async function processTranslation(lane, base64Data) {
  isDrawerOpen = true;

  const lane1Language = AVAILABLE_LANGUAGES[languageIndexFor(config.lane1Language, lang1Index)];
  const lane2Language = AVAILABLE_LANGUAGES[languageIndexFor(config.lane2Language, lang2Index)];
  const src = lane === 1 ? lane1Language : lane2Language;
  const dst = lane === 1 ? lane2Language : lane1Language;

  transcriptionData = {
    source: `${src.name} (Source)`,
    text: "Analyzing voice input...",
  };
  translationData = {
    target: `${dst.name} (Translation)`,
    text: "Translating...",
  };
  metaText = "";
  timing = {
    stt: "loading",
    translate: null,
    tts: config.enableTts ? null : "off",
  };

  try {
    // 1. Transcription
    transcriptionData.text = "Listening...";
    const sttStart = performance.now();
    const transcribedText = await transcribeAudio(base64Data, src.code);
    const sttSeconds = (performance.now() - sttStart) / 1000;
    timing = { ...timing, stt: sttSeconds, translate: "loading" };
    transcriptionData.text = transcribedText;

    if (!transcribedText.trim()) {
      translationData.text = "(No speech detected)";
      timing = { ...timing, translate: null, tts: null };
      return;
    }

    // 2. Translation (Streaming with fast speech chunking)
    let sentenceBuffer = "";
    const currentSpeechSession = config.enableTts
      ? createSpeechSession(dst.ttsLang, (ttsMs) => {
          timing = { ...timing, tts: ttsMs / 1000 };
        })
      : null;

    const llmPrompt = `Source language: ${src.name}\nTarget language: ${dst.name}\nText:\n${transcribedText}`;
    const result = await translateTextStream(
      llmPrompt,
      {
        ...config,
        modelName: config.modelName,
        systemPrompt: config.systemPrompt,
      },
      (delta, fullText) => {
        translationData.text = fullText;
        sentenceBuffer += delta;
        const parsed = extractSpeechChunks(sentenceBuffer);
        sentenceBuffer = parsed.remaining;
        parsed.chunks.forEach((chunk) => currentSpeechSession?.enqueue(chunk));
      },
    );

    const finalChunks = extractSpeechChunks(sentenceBuffer, true).chunks;
    finalChunks.forEach((chunk) => currentSpeechSession?.enqueue(chunk));

    translationData.text = result.translation;
    timing = {
      ...timing,
      translate: (result.firstTokenMs ?? result.durationMs) / 1000,
      tts: config.enableTts ? (timing.tts === null ? "loading" : timing.tts) : "off",
    };

    if (currentSpeechSession) {
      currentSpeechSession.playChain.catch((err) => {
        console.error(err);
        timing = { ...timing, tts: "error" };
      });
    }
  } catch (err) {
    console.error(err);
    transcriptionData.text =
      transcriptionData.text === "Listening..."
        ? "(Transcription failed)"
        : transcriptionData.text;
    translationData.text = `Error: ${err.message}`;
    if (timing) {
      timing = {
        stt: timing.stt === "loading" ? "error" : timing.stt,
        translate: timing.translate === "loading" ? "error" : timing.translate,
        tts: timing.tts === "loading" ? "error" : timing.tts,
      };
    }
  }
}

// Push-to-talk keyboard control
$effect(() => {
  const handleKeyDown = (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    const key = e.key.toLowerCase();

    if (config.keyboardMode === "landscape") {
      if (key === " " || e.key === "Spacebar") {
        e.preventDefault();
        if (!recorder.isRecording) {
          playBlip("speaker");
          activePerson = activePerson === 1 ? 2 : 1;
        }
      } else if (key === "z") {
        e.preventDefault();
        if (!e.repeat && !recorder.isRecording) handleRecordStart(activePerson);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handleRotateLanguage(activePerson, -1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        handleRotateLanguage(activePerson, 1);
      }
    } else {
      if (key === "z") {
        e.preventDefault();
        if (!e.repeat && !recorder.isRecording) handleRecordStart(1);
      } else if (key === "x") {
        e.preventDefault();
        if (!e.repeat && !recorder.isRecording) handleRecordStart(2);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        handleRotateLanguage(1, -1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        handleRotateLanguage(1, 1);
      } else if (key === "-" || key === "_") {
        e.preventDefault();
        handleRotateLanguage(2, -1);
      } else if (key === "+" || key === "=") {
        e.preventDefault();
        handleRotateLanguage(2, 1);
      }
    }
  };

  const handleKeyUp = (e) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    const key = e.key.toLowerCase();

    if (config.keyboardMode === "landscape") {
      if (key === "z" && recorder.isRecording) handleRecordStop();
    } else {
      if (key === "z" && activeLaneRecording === 1) handleRecordStop();
      if (key === "x" && activeLaneRecording === 2) handleRecordStop();
    }
  };

  window.addEventListener("keydown", handleKeyDown);
  window.addEventListener("keyup", handleKeyUp);
  return () => {
    window.removeEventListener("keydown", handleKeyDown);
    window.removeEventListener("keyup", handleKeyUp);
  };
});

// Whisplay hardware button polling
$effect(() => {
  let cancelled = false;
  let unavailableCount = 0;

  const pollHardware = async () => {
    try {
      const res = await fetch("/api/hardware", { cache: "no-store" });
      if (!res.ok) throw new Error(`hardware status ${res.status}`);
      const state = await res.json();
      if (!state.enabled || state.stale) return;

      unavailableCount = 0;
      const pressed = Boolean(state.buttonPressed);
      if (pressed !== hardwareButtonPressed) {
        hardwareButtonPressed = pressed;
        if (pressed) {
          handleRecordStart(activePerson);
        } else {
          handleRecordStop();
        }
      }
    } catch (err) {
      unavailableCount += 1;
      if (unavailableCount === 1) {
        console.debug("Hardware controls unavailable:", err);
      }
    } finally {
      if (!cancelled) window.setTimeout(pollHardware, unavailableCount > 10 ? 1000 : 80);
    }
  };

  pollHardware();
  return () => {
    cancelled = true;
  };
});
</script>

<div class="translator-envelope">
  <ResponseDrawer
    isActive={isDrawerOpen}
    onClose={() => (isDrawerOpen = false)}
    transcriptionSource={transcriptionData.source}
    transcriptionText={transcriptionData.text}
    translationTarget={translationData.target}
    translationText={translationData.text}
    metaText={metaText}
    timing={timing}
  />

  <main class="translator-workspace">
    <Visualizer
      activePerson={activePerson}
      isRecording={recorder.isRecording}
      analyser={recorder.analyser}
      barsCount={parseInt(config.visualizerBars, 10)}
    />

    <div class="languages-container">
      <LanguageLane
        laneId={1}
        laneLabel="1"
        languages={AVAILABLE_LANGUAGES}
        currentIndex={lang1Index}
        isRecording={activeLaneRecording === 1}
        isActivePerson={activePerson === 1}
        onRotate={(dir) => handleRotateLanguage(1, dir)}
        onSelect={() => (activePerson = 1)}
        onPressStart={() => handleRecordStart(1)}
        onPressEnd={handleRecordStop}
      />
      <LanguageLane
        laneId={2}
        laneLabel="2"
        languages={AVAILABLE_LANGUAGES}
        currentIndex={lang2Index}
        isRecording={activeLaneRecording === 2}
        isActivePerson={activePerson === 2}
        onRotate={(dir) => handleRotateLanguage(2, dir)}
        onSelect={() => (activePerson = 2)}
        onPressStart={() => handleRecordStart(2)}
        onPressEnd={handleRecordStop}
      />
    </div>
  </main>
</div>
