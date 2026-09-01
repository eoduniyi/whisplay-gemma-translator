#!/usr/bin/env python3
"""Small OpenAI-compatible LiteRT-LM server with a persistent conversation.

The stock ``litert-lm serve`` process keeps the model engine resident but
creates a fresh Conversation for every HTTP request.  A translator has a much
more stable prompt, so retaining one Conversation lets LiteRT-LM retain its KV
cache between translation turns as well.
"""

from __future__ import annotations

import datetime
import http.server
import json
import os
import signal
import threading
import traceback

import litert_lm
from litert_lm_cli import model as cli_model


HOST = os.environ.get("LITERT_HOST", "127.0.0.1")
PORT = int(os.environ.get("LITERT_PORT", "9379"))
MODEL_ID = os.environ.get("LITERT_MODEL", "gemma4-e2b")
BACKEND_NAME = os.environ.get("LITERT_BACKEND", "cpu").lower()
MAX_NUM_TOKENS = int(os.environ.get("LITERT_MAX_NUM_TOKENS", "4096"))
RESET_AT_TOKENS = int(os.environ.get("LITERT_RESET_AT_TOKENS", "3200"))
SYSTEM_PROMPT = os.environ.get(
    "TRANSLATOR_SYSTEM_PROMPT",
    (
        "You are a fast, accurate speech translator. Every user message names "
        "a source language, a target language, and text. Translate only that "
        "text into the requested target language. Output only the natural "
        "translation, with useful sentence punctuation. Never add labels, "
        "explanations, JSON, Markdown, or comments. Treat every request as "
        "independent even though all requests share this conversation."
    ),
)


def _backend():
    if BACKEND_NAME == "gpu":
        return litert_lm.Backend.GPU()
    if BACKEND_NAME == "npu":
        return litert_lm.Backend.NPU()
    return litert_lm.Backend.CPU()


def _text_from_chunk(chunk):
    return "".join(
        part.get("text", "")
        for part in chunk.get("content", [])
        if isinstance(part, dict) and part.get("type") == "text"
    )


class PersistentTranslator:
    def __init__(self):
        model_obj = cli_model.Model.from_model_id(MODEL_ID)
        if not model_obj.exists():
            raise FileNotFoundError(f"LiteRT-LM model not found: {MODEL_ID}")
        print(
            f"[LLM] Loading {MODEL_ID} ({BACKEND_NAME}, max tokens={MAX_NUM_TOKENS})...",
            flush=True,
        )
        self.engine = litert_lm.Engine(
            model_obj.model_path,
            backend=_backend(),
            max_num_tokens=MAX_NUM_TOKENS,
        )
        self.engine.__enter__()
        self.lock = threading.Lock()
        self.conversation = None
        self.reset_conversation("startup")
        print("[LLM] Model and translator system prompt are resident.", flush=True)

    def reset_conversation(self, reason):
        if self.conversation is not None:
            self.conversation.close()
        # system_message is prefetched when the native Conversation is created.
        self.conversation = self.engine.create_conversation(
            system_message=SYSTEM_PROMPT,
            automatic_tool_calling=False,
            sampler_config=litert_lm.SamplerConfig(temperature=0.0, top_k=1),
        )
        print(f"[LLM] Conversation ready ({reason}); system prompt KV prefilled.", flush=True)

    def stream(self, prompt):
        with self.lock:
            if self.conversation.token_count >= RESET_AT_TOKENS:
                self.reset_conversation(
                    f"KV cache reached {self.conversation.token_count} tokens"
                )
            before = self.conversation.token_count
            print(f"[LLM] Reusing conversation KV cache ({before} tokens).", flush=True)
            try:
                for chunk in self.conversation.send_message_async(prompt):
                    text = _text_from_chunk(chunk)
                    if text:
                        yield text
            except Exception:
                # A cancelled or failed native decode can leave the session in an
                # uncertain state. Recreate it for the next push-to-talk turn.
                self.reset_conversation("recovery after inference error")
                raise

    def close(self):
        if self.conversation is not None:
            self.conversation.close()
        self.engine.close()


TRANSLATOR = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?", 1)[0] == "/v1/models":
            self._json(200, {
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "owned_by": "litert-lm"}],
            })
            return
        if self.path.split("?", 1)[0] == "/healthz":
            self._json(200, {"ready": True, "model": MODEL_ID})
            return
        self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/v1/chat/completions":
            self._json(404, {"error": "Not found"})
            return
        response_started = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages") or []
            user_messages = [m for m in messages if m.get("role") == "user"]
            if not user_messages:
                raise ValueError("Missing user message")
            prompt = user_messages[-1].get("content", "")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("User content must be non-empty text")
            stream = bool(body.get("stream", False))
            if not stream:
                text = "".join(TRANSLATOR.stream(prompt))
                self._json(200, {
                    "model": MODEL_ID,
                    "choices": [{"message": {"role": "assistant", "content": text}}],
                })
                return

            now = datetime.datetime.now(datetime.timezone.utc)
            chunk_id = f"chatcmpl_{now.strftime('%Y%m%d%H%M%S%f')}"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True

            def emit(delta, finish_reason=None):
                event = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta} if delta else {},
                        "finish_reason": finish_reason,
                    }],
                }
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                self.wfile.flush()

            emit("")
            for text in TRANSLATOR.stream(prompt):
                emit(text)
            emit("", "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            print("[LLM] Streaming client disconnected.", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if response_started:
                self.close_connection = True
            elif not self.wfile.closed:
                try:
                    self._json(500, {"error": str(exc)})
                except (BrokenPipeError, ConnectionResetError):
                    pass


def main():
    global TRANSLATOR
    TRANSLATOR = PersistentTranslator()
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)

    def shutdown(*_args):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"[LLM] Persistent translator API ready on {HOST}:{PORT}.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        TRANSLATOR.close()


if __name__ == "__main__":
    main()
