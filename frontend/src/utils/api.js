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

// API client for the Python backend (backend/server.py). LLM calls are
// routed through the backend's /proxy by default so the kiosk stays
// same-origin (no CORS) and works fully offline.

// Normalize a user-entered endpoint into an OpenAI-compatible ".../v1" base.
export function getNormalizedBaseUrl(endpointUrl) {
  let url = endpointUrl.trim();
  if (!url) return "http://localhost:9379/v1";
  url = url.replace(/\/+$/, "");
  if (!url.endsWith("/v1")) {
    url += "/v1";
  }
  return url;
}

// Cheap connectivity probe: GET {base}/v1/models.
export async function testConnectionAPI(endpointUrl, useProxy, apiKey) {
  const baseUrl = getNormalizedBaseUrl(endpointUrl);
  const targetUrl = `${baseUrl}/models`;

  const headers = {};
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`;
  }

  const fetchUrl = useProxy ? `/proxy?url=${encodeURIComponent(targetUrl)}` : targetUrl;

  const response = await fetch(fetchUrl, { method: "GET", headers });
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  return true;
}

// POST base64 Float32 PCM (16 kHz mono) to the local Moonshine STT.
async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}

export async function transcribeAudio(base64Data, sourceLangCode) {
  const response = await fetchWithTimeout(
    "/api/stt",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        audio_base64: base64Data,
        language: sourceLangCode,
      }),
    },
    30000,
  );

  if (!response.ok) {
    throw new Error(`STT failed: ${response.status}`);
  }

  const sttData = await response.json();
  return sttData.text || "";
}

function generatePayloadJSON(transcribedText, model, systemPrompt, stream = false) {
  const messages = [];
  if (systemPrompt && systemPrompt.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() });
  }
  messages.push({
    role: "user",
    content: transcribedText,
  });

  return JSON.stringify({
    model: model || "gemma4-e2b",
    messages,
    stream,
    temperature: 0,
    top_k: 1,
  });
}

// Stream an OpenAI-compatible chat completion. `onDelta` is called as soon as
// each SSE token reaches the browser, which lets the UI and sentence/TTS queue
// advance without waiting for the full translation.
export async function translateTextStream(transcribedText, config, onDelta) {
  const { endpointUrl, useProxy, apiKey, modelName, systemPrompt } = config;
  const baseUrl = getNormalizedBaseUrl(endpointUrl);
  const targetUrl = `${baseUrl}/chat/completions`;
  const payload = generatePayloadJSON(transcribedText, modelName, systemPrompt, true);
  const headers = { "Content-Type": "application/json" };
  if (apiKey && apiKey.trim() !== "") {
    headers.Authorization = `Bearer ${apiKey.trim()}`;
  }
  const fetchUrl = useProxy ? `/proxy?url=${encodeURIComponent(targetUrl)}` : targetUrl;
  const startedAt = performance.now();
  const response = await fetch(fetchUrl, { method: "POST", headers, body: payload });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API ${response.status}: ${errorText || response.statusText}`);
  }
  if (!response.body) throw new Error("Streaming response body is unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let translation = "";
  let firstTokenMs = null;

  const consumeEvent = (rawEvent) => {
    const data = rawEvent
      .split("\n")
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (!data || data === "[DONE]") return;
    const event = JSON.parse(data);
    const delta = event.choices?.[0]?.delta?.content || "";
    if (!delta) return;
    if (firstTokenMs === null) firstTokenMs = performance.now() - startedAt;
    translation += delta;
    onDelta?.(delta, translation);
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    for (const event of events) consumeEvent(event);
    if (done) break;
  }
  if (buffer.trim()) consumeEvent(buffer);
  return {
    translation: translation.trim(),
    firstTokenMs,
    durationMs: performance.now() - startedAt,
  };
}

// Chat-completions request. The system prompt demands a bare
// {"translation": ...} JSON object; we still tolerate ``` fences and fall
// back to the raw reply text if parsing fails.
export async function translateText(transcribedText, config) {
  const { endpointUrl, useProxy, apiKey, modelName, systemPrompt } = config;
  const baseUrl = getNormalizedBaseUrl(endpointUrl);
  const targetUrl = `${baseUrl}/chat/completions`;
  const payload = generatePayloadJSON(transcribedText, modelName, systemPrompt);

  const headers = { "Content-Type": "application/json" };
  if (apiKey && apiKey.trim() !== "") {
    headers["Authorization"] = `Bearer ${apiKey.trim()}`;
  }

  const fetchUrl = useProxy ? `/proxy?url=${encodeURIComponent(targetUrl)}` : targetUrl;
  const startRequestTime = Date.now();

  const response = await fetchWithTimeout(
    fetchUrl,
    {
      method: "POST",
      headers,
      body: payload,
    },
    60000,
  );
  const requestDuration = ((Date.now() - startRequestTime) / 1000).toFixed(2);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`API ${response.status}: ${errorText || response.statusText}`);
  }

  const data = await response.json();
  let modelResponse = "";
  if (data.choices && data.choices[0] && data.choices[0].message) {
    modelResponse = data.choices[0].message.content || "";
  } else {
    modelResponse = JSON.stringify(data, null, 2);
  }

  let translationVal = "";
  try {
    let cleanJson = modelResponse.trim();
    if (cleanJson.startsWith("```json")) {
      cleanJson = cleanJson.slice(7);
    }
    if (cleanJson.startsWith("```")) {
      cleanJson = cleanJson.slice(3);
    }
    if (cleanJson.endsWith("```")) {
      cleanJson = cleanJson.slice(0, -3);
    }
    cleanJson = cleanJson.trim();

    const parsed = JSON.parse(cleanJson);
    translationVal = parsed.translation || "";
  } catch (e) {
    translationVal = modelResponse;
  }

  return {
    translation: translationVal,
    duration: requestDuration,
    tokens: data.usage?.total_tokens || 0,
  };
}

// Word-safe chunking so each /api/tts request stays under ~`limit` chars.
export function splitTextIntoSpeechChunks(text, limit = 180) {
  const words = text.split(/\s+/);
  const chunks = [];
  let currentChunk = "";
  for (const word of words) {
    if ((currentChunk + " " + word).trim().length <= limit) {
      currentChunk = (currentChunk + " " + word).trim();
    } else {
      if (currentChunk) chunks.push(currentChunk);
      currentChunk = word;
    }
  }
  if (currentChunk) chunks.push(currentChunk);
  return chunks;
}

// Pull complete, speakable clauses out of a streaming token buffer. This is
// based on whisplay-ai-chatbot's StreamResponser behavior: punctuation emits a
// chunk early, short clauses are merged, and an overlong unpunctuated passage
// is split at a safe whitespace boundary.
export function extractSpeechChunks(text, final = false, limit = 90, clauseSplitThreshold = 60) {
  const chunks = [];
  let start = 0;
  let scan = 0;
  while (scan < text.length) {
    const char = text[scan];
    const previous = text[scan - 1] || "";
    const next = text[scan + 1] || "";
    const decimalPoint = char === "." && /\d/.test(previous) && /\d/.test(next);
    const strongBoundary = /[。！？!?\n]/.test(char) || (char === "." && !decimalPoint);
    // Keep short sentences intact. Clause punctuation is only a useful early
    // TTS boundary once the current sentence has grown large enough.
    const softBoundary = /[，,：:；;]/.test(char) && scan - start + 1 >= clauseSplitThreshold;
    if (strongBoundary || softBoundary) {
      const candidate = text.slice(start, scan + 1).trim();
      if (candidate) chunks.push(candidate);
      start = scan + 1;
    } else if (scan - start + 1 >= limit) {
      const windowText = text.slice(start, scan + 1);
      const whitespace = Math.max(windowText.lastIndexOf(" "), windowText.lastIndexOf("\n"));
      if (whitespace >= Math.floor(limit * 0.55)) {
        const candidate = windowText.slice(0, whitespace).trim();
        if (candidate) chunks.push(candidate);
        start += whitespace + 1;
        scan = start;
        continue;
      }
    }
    scan += 1;
  }
  const remaining = text.slice(start).trimStart();
  if (final && remaining.trim()) {
    chunks.push(remaining.trim());
    return { chunks, remaining: "" };
  }
  return { chunks, remaining };
}
