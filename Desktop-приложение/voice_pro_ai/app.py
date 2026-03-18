from __future__ import annotations

import base64
from collections import deque
import ctypes
import io
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
import zipfile
from datetime import date, datetime
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import keyboard
import numpy as np
import pyperclip
import requests
import sounddevice as sd
from PIL import Image
from pystray import Icon, Menu, MenuItem

from voice_pro_ai import updater
from voice_pro_ai.version import APP_VERSION

try:
    import winsound
except Exception:  # pragma: no cover
    winsound = None

try:
    import winreg
except Exception:  # pragma: no cover
    winreg = None

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None
try:
    import ctranslate2
except Exception:  # pragma: no cover
    ctranslate2 = None
try:
    from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel as vosk_set_log_level
    VOSK_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    KaldiRecognizer = None
    VoskModel = None
    vosk_set_log_level = None
    VOSK_IMPORT_ERROR = str(sys.exc_info()[1] or "")

try:
    import sherpa_onnx
    SHERPA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    sherpa_onnx = None
    SHERPA_IMPORT_ERROR = str(sys.exc_info()[1] or "")

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    import soundcard as sc
except Exception:  # pragma: no cover
    sc = None

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover
    Llama = None
try:
    from gpt4all import GPT4All
except Exception:  # pragma: no cover
    GPT4All = None
try:
    from ctransformers import AutoModelForCausalLM
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None

try:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TQDM_DISABLE"] = "1"
    from huggingface_hub import HfApi
    from huggingface_hub.utils import disable_progress_bars
    try:
        disable_progress_bars()
    except Exception:
        pass
except Exception:  # pragma: no cover
    HfApi = None


APP_NAME = "Voice PRO AI"
APP_RELEASE_BADGE = "Beta"  # Set to "" to hide the badge.
APP_USER_MODEL_ID = "VoicePROAI.Desktop.v2"
AUTOSTART_VALUE_NAME = "VoiceProAI"
AUTOSTART_TASK_NAME = "VoicePROAI_Autostart"
EMBEDDED_GITHUB_REPO = os.getenv("VOICEPRO_GITHUB_REPO", "dencikpapinka-png/ai-voice-pro-ai").strip()
DEFAULT_UPDATE_FEED_URL = (
    f"https://github.com/{EMBEDDED_GITHUB_REPO}/releases/latest/download/update.json"
    if "/" in EMBEDDED_GITHUB_REPO
    else ""
)
DEFAULT_UPDATE_CHANNEL = "stable"
CONFIG_DIR = Path.home() / "AppData" / "Roaming" / "VoicePROAI"
CONFIG_PATH = CONFIG_DIR / "config.json"
LOG_PATH = CONFIG_DIR / "app.log"
STATS_PATH = CONFIG_DIR / "stats.json"
HISTORY_PATH = CONFIG_DIR / "history.json"
AUTOREPLACE_PATH = CONFIG_DIR / "autoreplace.json"
MUTEX_NAME = "VoiceProAI_SingleInstance"
IPC_COMMAND_PATH = CONFIG_DIR / "ipc_command.txt"
MEDIA_EXPORT_DIR = CONFIG_DIR / "exports"
MEETINGS_DIR = CONFIG_DIR / "meetings"


@dataclass
class AppConfig:
    openrouter_api_key: str = ""
    model: str = "google/gemini-2.5-flash"
    transcription_backend: str = "api"  # api | local
    local_backend_engine: str = "whisper"  # whisper | vosk | sherpa
    local_whisper_model: str = "small"  # small | medium | large-v3
    local_vosk_model: str = "vosk-model-small-ru-0.22"
    local_sherpa_model: str = "voicepro-gigaam-rnnt"
    meeting_summary_backend: str = "api"  # api | local_gguf | local_ollama
    meeting_local_summary_model: str = "qwen2.5-3b-instruct-q4_k_m"
    meeting_system_audio_mode: str = "auto"  # auto | manual
    meeting_system_audio_device_id: str = ""  # sc::<id> | si::<index> | wa::<index>
    meeting_ollama_model: str = "qwen2.5:7b-instruct-q4_K_M"
    meeting_ollama_url: str = "http://127.0.0.1:11434"
    ptt_key: str = "f8"
    activation_mode: str = "hold"  # hold | toggle
    auto_paste: bool = True
    copy_result_to_clipboard: bool = False
    restore_clipboard: bool = True
    history_enabled: bool = True
    autoreplace_enabled: bool = True
    punctuate_text: bool = True
    capitalize_sentences: bool = True
    terminal_period: bool = True
    convert_numbers: bool = False
    normalize_quotes_dashes: bool = True
    clean_text: bool = False
    clean_text_mode: str = "soft"  # off | soft | strong
    max_text_quality: bool = False
    whisper_medium_turbo: bool = False
    local_fast_refine: bool = True
    show_overlay_widget: bool = True
    show_live_preview_text: bool = True
    show_mic_quality_indicator: bool = True
    play_recording_sounds: bool = True
    play_notification_sounds: bool = True
    autostart_windows: bool = False
    auto_download_updates: bool = False
    update_feed_url: str = DEFAULT_UPDATE_FEED_URL
    update_channel: str = DEFAULT_UPDATE_CHANNEL
    run_as_admin: bool = False
    hardware_acceleration: bool = True
    notify_on_no_sound: bool = True
    input_device: str = ""
    sample_rate: int = 24000


class ConfigStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppConfig:
        if not self.path.exists():
            cfg = AppConfig()
            self.save(cfg)
            return cfg

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Config JSON root must be an object")
            # Backward compatibility: migrate old boolean clean_text to mode.
            if "clean_text_mode" not in raw and "clean_text" in raw:
                raw["clean_text_mode"] = "soft" if bool(raw.get("clean_text")) else "off"
            allowed = {f.name for f in fields(AppConfig)}
            filtered = {k: v for k, v in raw.items() if k in allowed}
            cfg = AppConfig(**filtered)
            # Persist normalized config to remove stale keys from disk.
            self.save(cfg)
            return cfg
        except Exception:
            logging.exception("Failed to read config, using defaults")
            return AppConfig()

    def save(self, cfg: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class AppStats:
    total_words: int = 0
    total_audio_seconds: float = 0.0
    today_words: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    today_input_tokens: int = 0
    today_output_tokens: int = 0
    today_iso_date: str = ""


class StatsStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> AppStats:
        today = date.today().isoformat()
        if not self.path.exists():
            stats = AppStats(today_iso_date=today)
            self.save(stats)
            return stats
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("Stats JSON root must be an object")
            allowed = {f.name for f in fields(AppStats)}
            filtered = {k: v for k, v in raw.items() if k in allowed}
            stats = AppStats(**filtered)
        except Exception:
            logging.exception("Failed to read stats, using defaults")
            stats = AppStats(today_iso_date=today)

        if stats.today_iso_date != today:
            stats.today_iso_date = today
            stats.today_words = 0
            stats.today_input_tokens = 0
            stats.today_output_tokens = 0
            self.save(stats)
        return stats

    def save(self, stats: AppStats) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> AppStats:
        stats = AppStats(today_iso_date=date.today().isoformat())
        self.save(stats)
        return stats


class HistoryStore:
    def __init__(self, path: Path, max_items: int = 300) -> None:
        self.path = path
        self.max_items = max(50, int(max_items))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            self.save([])
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
            items: list[dict] = []
            for row in raw:
                if not isinstance(row, dict):
                    continue
                item_id = str(row.get("id", "")).strip()
                text = str(row.get("text", "")).strip()
                ts = str(row.get("ts", "")).strip()
                if not item_id or not text or not ts:
                    continue
                items.append({"id": item_id, "text": text, "ts": ts})
            return items[: self.max_items]
        except Exception:
            logging.exception("Failed to read history, using empty list")
            return []

    def save(self, items: list[dict]) -> None:
        clean: list[dict] = []
        for row in items[: self.max_items]:
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            ts = str(row.get("ts", "")).strip()
            if not item_id or not text or not ts:
                continue
            clean.append({"id": item_id, "text": text, "ts": ts})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, text: str, ts: Optional[str] = None) -> list[dict]:
        now_ts = ts or datetime.now().strftime("%d.%m.%Y, %H:%M")
        item = {
            "id": f"{int(time.time() * 1000)}-{abs(hash(text)) % 100000}",
            "text": str(text).strip(),
            "ts": now_ts,
        }
        items = self.load()
        items.insert(0, item)
        self.save(items)
        return self.load()

    def delete(self, item_id: str) -> list[dict]:
        target = str(item_id).strip()
        items = [x for x in self.load() if str(x.get("id", "")).strip() != target]
        self.save(items)
        return self.load()

    def clear(self) -> list[dict]:
        self.save([])
        return []


def default_autoreplace_rules() -> list[dict]:
    return [
        {
            "id": "rx-trim-space-before-punct",
            "enabled": True,
            "pattern": r"\s+([,.;:!?])",
            "replacement": r"\1",
            "mode": "RegExp",
            "flags": "u",
        },
        {
            "id": "rx-collapse-spaces",
            "enabled": True,
            "pattern": r"[ \t]{2,}",
            "replacement": " ",
            "mode": "RegExp",
            "flags": "u",
        },
        {
            "id": "rx-collapse-empty-lines",
            "enabled": True,
            "pattern": r"\n{3,}",
            "replacement": "\n\n",
            "mode": "RegExp",
            "flags": "u",
        },
    ]


class AutoReplaceStore:
    def __init__(self, path: Path, max_items: int = 200) -> None:
        self.path = path
        self.max_items = max(20, int(max_items))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            defaults = default_autoreplace_rules()
            self.save(defaults)
            return defaults
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("AutoReplace JSON root must be an array")
            out: list[dict] = []
            for row in raw:
                if not isinstance(row, dict):
                    continue
                item = {
                    "id": str(row.get("id", "")).strip() or f"rule-{int(time.time() * 1000)}",
                    "enabled": bool(row.get("enabled", True)),
                    "pattern": str(row.get("pattern", "")),
                    "replacement": str(row.get("replacement", "")),
                    "mode": "RegExp" if str(row.get("mode", "Text")).lower() in {"regexp", "regex", "rx"} else "Text",
                    "flags": str(row.get("flags", "u")),
                }
                out.append(item)
            return out[: self.max_items]
        except Exception:
            logging.exception("Failed to read autoreplace rules, using defaults")
            defaults = default_autoreplace_rules()
            self.save(defaults)
            return defaults

    def save(self, rules: list[dict]) -> None:
        clean: list[dict] = []
        for row in list(rules)[: self.max_items]:
            if not isinstance(row, dict):
                continue
            clean.append(
                {
                    "id": str(row.get("id", "")).strip() or f"rule-{int(time.time() * 1000)}",
                    "enabled": bool(row.get("enabled", True)),
                    "pattern": str(row.get("pattern", "")),
                    "replacement": str(row.get("replacement", "")),
                    "mode": "RegExp" if str(row.get("mode", "Text")).lower() in {"regexp", "regex", "rx"} else "Text",
                    "flags": str(row.get("flags", "u")),
                }
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self) -> list[dict]:
        defaults = default_autoreplace_rules()
        self.save(defaults)
        return defaults


class AudioRecorder:
    def __init__(self, config: AppConfig) -> None:
        self.sample_rate = config.sample_rate
        self.input_device = str(config.input_device or "").strip()
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._recording = False
        self._recorded_chunks: list[np.ndarray] = []
        self._record_start_monotonic = 0.0
        self._level_rms = 0.0
        self._level_peak = 0.0
        self._level_updated_monotonic = 0.0

    @staticmethod
    def list_input_devices() -> list[str]:
        try:
            devices = sd.query_devices()
        except Exception:
            logging.exception("Failed to query input devices")
            return []
        out: list[str] = []
        for d in devices:
            try:
                if int(d.get("max_input_channels", 0) or 0) > 0:
                    name = str(d.get("name", "")).strip()
                    if name:
                        out.append(name)
            except Exception:
                continue
        return out

    @staticmethod
    def resolve_device_index(device_name: str) -> Optional[int]:
        name = str(device_name or "").strip()
        if not name:
            return None
        try:
            devices = sd.query_devices()
        except Exception:
            logging.exception("Failed to query devices for resolve")
            return None
        for idx, d in enumerate(devices):
            try:
                if int(d.get("max_input_channels", 0) or 0) <= 0:
                    continue
                if str(d.get("name", "")).strip() == name:
                    return idx
            except Exception:
                continue
        return None

    def ensure_stream(self) -> None:
        if self._stream is not None:
            return

        def callback(indata: np.ndarray, frames: int, _time_info, status) -> None:
            if status:
                logging.warning("Audio callback status: %s", status)

            frame = np.copy(indata[:, 0])
            frame_abs = np.abs(frame)
            instant_peak = float(np.max(frame_abs)) if frame_abs.size else 0.0
            instant_rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0

            with self._lock:
                if self._recording:
                    self._recorded_chunks.append(frame)
                    if self._level_rms <= 0.0:
                        self._level_rms = instant_rms
                    else:
                        self._level_rms = (self._level_rms * 0.72) + (instant_rms * 0.28)
                    if self._level_peak <= 0.0:
                        self._level_peak = instant_peak
                    else:
                        self._level_peak = max(instant_peak, self._level_peak * 0.86)
                    self._level_updated_monotonic = time.monotonic()

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=256,
            latency="low",
            device=self.resolve_device_index(self.input_device),
            callback=callback,
        )
        self._stream.start()
        logging.info("Audio stream started")

    def start_recording(self) -> None:
        self.ensure_stream()
        with self._lock:
            # Strict push-to-talk: capture only after hotkey is pressed.
            self._recorded_chunks = []
            self._recording = True
            self._record_start_monotonic = time.monotonic()
            self._level_rms = 0.0
            self._level_peak = 0.0
            self._level_updated_monotonic = self._record_start_monotonic

    def stop_recording(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            self._recording = False
            chunks = self._recorded_chunks
            self._recorded_chunks = []
            duration_sec = max(0.0, time.monotonic() - self._record_start_monotonic)

        if not chunks:
            return None, duration_sec

        audio = np.concatenate(chunks, axis=0)
        return audio, duration_sec

    def get_recording_snapshot(self) -> tuple[Optional[np.ndarray], float]:
        with self._lock:
            if not self._recording:
                return None, 0.0
            chunks = list(self._recorded_chunks)
            duration_sec = max(0.0, time.monotonic() - self._record_start_monotonic)

        if not chunks:
            return None, duration_sec
        audio = np.concatenate(chunks, axis=0)
        return audio, duration_sec

    def get_signal_quality(self) -> tuple[int, str]:
        with self._lock:
            if not self._recording:
                return 0, "quiet"
            rms = float(self._level_rms)
            peak = float(self._level_peak)
            last_updated = float(self._level_updated_monotonic)

        if (time.monotonic() - last_updated) > 0.5:
            return 0, "quiet"

        level_norm = max(peak, rms * 3.4)
        level_pct = int(max(0, min(100, round(level_norm * 100))))

        if peak >= 0.92 or rms >= 0.45:
            state = "overload"
        elif peak < 0.018 and rms < 0.0045:
            state = "quiet"
        else:
            state = "ok"
        return level_pct, state

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logging.exception("Failed to close audio stream")
            self._stream = None

    def set_input_device(self, device_name: str) -> None:
        with self._lock:
            self.input_device = str(device_name or "").strip()
            was_recording = self._recording
            self._recording = False
            self._recorded_chunks = []
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logging.exception("Failed to close old stream on device change")
            self._stream = None
        # Warm up new device immediately.
        try:
            self.ensure_stream()
        except Exception:
            logging.exception("Failed to warm up stream after device change")
        if was_recording:
            # Safety: never auto-resume recording after device switch.
            self._record_start_monotonic = 0.0


class OpenRouterTranscriber:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @staticmethod
    def supports_live_preview() -> bool:
        return True

    def get_runtime_status(self) -> dict:
        key_ok = bool(str(self.config.openrouter_api_key or "").strip())
        return {
            "backend": "api",
            "model_label": str(self.config.model or "OpenRouter"),
            "ready": True,
            "state": "ready" if key_ok else "init",
            "message": "Готово" if key_ok else "Укажите API-ключ для распознавания через OpenRouter.",
        }

    def _clean_text_mode(self) -> str:
        raw = str(getattr(self.config, "clean_text_mode", "") or "").strip().lower()
        if raw in {"off", "soft", "strong"}:
            return raw
        # Fallback for old configs with only boolean.
        return "soft" if bool(getattr(self.config, "clean_text", False)) else "soft"

    def transcribe(self, wav_bytes: bytes, duration_sec: float) -> tuple[str, dict]:
        if not self.config.openrouter_api_key:
            raise RuntimeError("Укажите OpenRouter API-ключ в настройках")

        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        variants = [
            {
                "type": "input_audio",
                "input_audio": {"data": b64_audio, "format": "wav"},
            },
            {
                "type": "input_audio",
                "inputAudio": {"data": b64_audio, "format": "wav"},
            },
        ]

        last_error: Optional[Exception] = None
        punctuation_instr = (
            "Add natural sentence punctuation."
            if self.config.punctuate_text
            else "Do not force punctuation; keep as spoken."
        )
        capitalize_instr = (
            "Capitalize sentence starts."
            if self.config.capitalize_sentences
            else "Preserve casing from recognition output."
        )
        numbers_instr = (
            "Convert spoken numbers to digits where natural (for example 'две тысячи' -> '2000')."
            if self.config.convert_numbers
            else "Do not force conversion of spoken numbers to digits."
        )
        quotes_instr = (
            "If direct speech is clear, keep quotes and prefer Russian guillemets В«В». "
            "Normalize long dash to a standard hyphen when needed."
            if self.config.normalize_quotes_dashes
            else "Do not enforce quote/dash normalization."
        )
        clean_mode = self._clean_text_mode()
        if clean_mode == "strong":
            clean_text_instr = (
                "Strong clean text mode: remove discourse fillers, repetitions, and non-essential conversational glue, "
                "while preserving core meaning and intent."
            )
        elif clean_mode == "soft":
            clean_text_instr = (
                "Soft clean text mode: remove only obvious fillers and repeated fragments; preserve wording and structure."
            )
        else:
            clean_text_instr = "Keep wording close to recognition output except hesitation sounds."
        for part in variants:
            try:
                payload = {
                    "model": self.config.model,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 8000,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a high-accuracy speech-to-text transcriber. "
                                "Transcribe meaning faithfully, but remove disfluencies and filler sounds "
                                "(for example: 'э', 'эм', 'ммм', elongated hesitation sounds). "
                                "Do not summarize or paraphrase content words. "
                                f"{punctuation_instr} "
                                f"{capitalize_instr} "
                                f"{numbers_instr} "
                                f"{quotes_instr} "
                                f"{clean_text_instr} "
                                "Output plain text only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Transcribe this audio fully. "
                                        f"Audio duration is about {int(round(duration_sec))} seconds. "
                                        "Remove filler sounds and hesitation tokens, but keep normal words and conjunctions. "
                                        f"{punctuation_instr} "
                                        f"{capitalize_instr} "
                                        f"{numbers_instr} "
                                        f"{quotes_instr} "
                                        f"{clean_text_instr} "
                                        "Output plain text only."
                                    ),
                                },
                                part,
                            ],
                        },
                    ],
                }
                result = self._call_openrouter(payload)
                text = self._extract_text(result)
                if text:
                    return self._postprocess_text(text), self._extract_usage(result)
            except Exception as err:  # pragma: no cover
                last_error = err
                logging.exception("OpenRouter variant failed")

        if last_error:
            raise last_error
        raise RuntimeError("Empty transcription")

    def transcribe_live_preview(self, wav_bytes: bytes, duration_sec: float) -> str:
        if not self.config.openrouter_api_key:
            return ""

        b64_audio = base64.b64encode(wav_bytes).decode("ascii")
        variants = [
            {
                "type": "input_audio",
                "input_audio": {"data": b64_audio, "format": "wav"},
            },
            {
                "type": "input_audio",
                "inputAudio": {"data": b64_audio, "format": "wav"},
            },
        ]

        last_error: Optional[Exception] = None
        for part in variants:
            try:
                payload = {
                    "model": self.config.model,
                    "temperature": 0,
                    "top_p": 1,
                    "max_tokens": 420,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a high-accuracy real-time speech transcriber. "
                                "Transcribe verbatim, exactly as spoken, with no paraphrasing and no translation. "
                                "Primary language is Russian. If speech is Russian, output Russian Cyrillic text. "
                                "Do not invent words. If unsure, keep closest audible words only. "
                                "Output plain text only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Live transcript for overlay preview, verbatim. "
                                        f"Audio duration is about {int(round(duration_sec))} seconds. "
                                        "Return only the spoken words from this audio segment."
                                    ),
                                },
                                part,
                            ],
                        },
                    ],
                }
                result = self._call_openrouter(payload, timeout_sec=18)
                text = self._extract_text(result)
                if text:
                    return self._postprocess_live_preview(text)
            except Exception as err:
                last_error = err
                logging.exception("OpenRouter live preview variant failed")

        if last_error:
            logging.info("Live preview transcription skipped: %s", last_error)
        return ""

    @staticmethod
    def _extract_usage(data: dict) -> dict:
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return {"input_tokens": 0, "output_tokens": 0}
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
        try:
            input_tokens = max(0, int(input_tokens or 0))
        except Exception:
            input_tokens = 0
        try:
            output_tokens = max(0, int(output_tokens or 0))
        except Exception:
            output_tokens = 0
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    def _call_openrouter(self, payload: dict, timeout_sec: int = 90) -> dict:
        resp = requests.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost.voice-pro-ai",
                "X-Title": APP_NAME,
            },
            json=payload,
            timeout=timeout_sec,
        )

        data: dict
        try:
            data = resp.json()
        except Exception:
            raise RuntimeError(f"Invalid API response (HTTP {resp.status_code})")

        if not resp.ok:
            message = data.get("error", {}).get("message") or f"HTTP {resp.status_code}"
            raise RuntimeError(message)
        return data

    @staticmethod
    def _extract_text(data: dict) -> str:
        content = data.get("choices", [{}])[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join([x for x in parts if x]).strip()

        text = data.get("choices", [{}])[0].get("message", {}).get("text")
        return str(text).strip() if isinstance(text, str) else ""

    def _postprocess_text(self, text: str) -> str:
        started = time.perf_counter()
        src_text = str(text or "").strip()
        if not src_text:
            return ""

        normalized = OpenRouterTranscriber._normalize_for_denoise(src_text)
        normalized = OpenRouterTranscriber._remove_disfluencies(normalized)
        if not normalized:
            return ""

        normalized = OpenRouterTranscriber._fix_common_asr_typos(normalized)
        normalized = OpenRouterTranscriber._repair_common_word_endings(normalized)
        pre_denoise = normalized

        denoise_removed_tokens = 0
        denoise_repetition_collapses = 0
        denoise_guard_rollbacks = 0
        clean_mode = self._clean_text_mode()
        if clean_mode != "off":
            normalized, denoise_stats = OpenRouterTranscriber._denoise_text_balanced(
                normalized,
                strong=(clean_mode == "strong"),
            )
            denoise_removed_tokens = int(denoise_stats.get("removed_tokens", 0))
            denoise_repetition_collapses = int(denoise_stats.get("repetition_collapses", 0))
            if OpenRouterTranscriber._semantic_loss_exceeds_threshold(pre_denoise, normalized):
                denoise_guard_rollbacks += 1
                normalized = pre_denoise

        normalized = self._apply_surface_formatting(normalized)
        if OpenRouterTranscriber._semantic_loss_exceeds_threshold(pre_denoise, normalized):
            denoise_guard_rollbacks += 1
            normalized = self._apply_surface_formatting(pre_denoise)

        postprocess_ms = (time.perf_counter() - started) * 1000.0
        logging.info(
            "Postprocess metrics: denoise_removed_tokens=%s denoise_repetition_collapses=%s denoise_guard_rollbacks=%s postprocess_ms=%.1f",
            denoise_removed_tokens,
            denoise_repetition_collapses,
            denoise_guard_rollbacks,
            postprocess_ms,
        )
        return normalized.strip()

    def _apply_surface_formatting(self, text: str) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        if self.config.punctuate_text:
            lines = normalized.splitlines()
            out: list[str] = []
            for line in lines:
                s = line.strip()
                if not s:
                    out.append("")
                    continue
                if s.endswith((".", "!", "?", "…")):
                    out.append(s)
                    continue
                out.append(OpenRouterTranscriber._choose_terminal_punctuation(s))
            normalized = "\n".join(out).strip()
        elif self.config.terminal_period and normalized and not normalized.endswith((".", "!", "?", "…")):
            normalized = normalized + "."

        if self.config.capitalize_sentences:
            normalized = OpenRouterTranscriber._capitalize_sentence_starts(normalized)

        if self.config.normalize_quotes_dashes:
            normalized = OpenRouterTranscriber._insert_missing_quotes(normalized)
            normalized = OpenRouterTranscriber._normalize_quotes_to_guillemets(normalized)
            normalized = re.sub(r"[—–]", "-", normalized)
        normalized = OpenRouterTranscriber._normalize_russian_punctuation(normalized)
        if bool(getattr(self.config, "max_text_quality", False)):
            normalized = OpenRouterTranscriber._enhance_text_quality(normalized)
        return normalized.strip()

    @staticmethod
    def _postprocess_live_preview(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _append_terminal_punctuation(s: str) -> str:
        lowered = s.lower()
        if re.search(r"\b(кто|что|где|когда|почему|зачем|как|сколько|какой|какая|какие|which|what|why|how|when|where)\b", lowered):
            return s + "?"
        return s + "."

    @staticmethod
    def _choose_terminal_punctuation(s: str) -> str:
        text = str(s or "").strip()
        if not text:
            return ""
        if OpenRouterTranscriber._looks_like_question(text):
            return text + "?"
        if OpenRouterTranscriber._looks_like_exclamation(text):
            return text + "!"
        return text + "."

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        lowered = text.lower()
        compact = re.sub(r"\s+", " ", lowered).strip()
        if re.search(
            r"\b(кто|что|где|когда|почему|зачем|как|сколько|какой|какая|какие|чей|чья|чьи|"
            r"можно\s+ли|не\s+мог(ли|ла|ли\s+бы)?|разве|неужели)\b",
            lowered,
            flags=re.IGNORECASE,
        ):
            return True
        if re.match(
            r"(?iu)^(можно|можешь|сможешь|сможете|подскажи|скаж(и|ите)|расскажи|объясни|"
            r"уточни|покажи|поможешь|поможете|получится|будет)\b",
            compact,
        ):
            return True
        # For long phrases with filler words at the start, infer question from the tail.
        tail = compact[-140:] if len(compact) > 140 else compact
        if re.search(
            r"(?iu)\b(можно|можем|сможем|получится|будет|нужно|надо)\b.{0,28}\bли\b",
            tail,
        ):
            return True
        if re.search(
            r"(?iu)\b(можем|сможем|можно)\b.{0,56}\b(сделать|реализовать|добавить|поправить|исправить|ускорить|оптимизировать)\b",
            tail,
        ):
            return True
        if re.search(
            r"(?iu)\b(можешь|сможешь|подскаж(и|ите)|скаж(и|ите)|объясни|расскажи|покажи)\b",
            tail,
        ):
            return True
        if lowered.endswith(" ли") or re.search(r"\bли\b", lowered):
            return True
        return False

    @staticmethod
    def _looks_like_exclamation(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return False
        if re.search(r"\b(отлично|супер|класс|здорово|внимание|срочно)\b", lowered):
            return True
        return False

    @staticmethod
    def _fix_common_asr_typos(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Conservative fixes for frequent ASR artifacts in RU dictation.
        replacements = [
            (r"(?iu)\bвнесеет\b", "внесет"),
            (r"(?iu)\bфайли\b", "файл"),
            (r"(?iu)\bэксель\s+файл\b", "Excel-файл"),
            (r"(?iu)\bexcel\s+файл\b", "Excel-файл"),
        ]
        for pat, repl in replacements:
            s = re.sub(pat, repl, s)
        return s

    @staticmethod
    def _repair_common_word_endings(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        replacements = [
            (r"(?iu)\bзнаки\s+применания\b", "знаки препинания"),
            (r"(?iu)\bзнак[и]?\s+применания\b", "знаки препинания"),
            (r"(?iu)\bприпинания\b", "препинания"),
            (r"(?iu)\bсочентани[ея]\b", "сочетание"),
            (r"(?iu)\bнатсройк[аи]\b", "настройка"),
            (r"(?iu)\bвесри[яи]\b", "версия"),
            (r"(?iu)\bтранскребаци[яи]\b", "транскрибации"),
            (r"(?iu)\bтранскрбиру[её]т\b", "транскрибирует"),
            (r"(?iu)\bтранскрибуру[ею]т\b", "транскрибируют"),
            (r"(?iu)\bвстравит[ья]\b", "вставить"),
            (r"(?iu)\bпопров[ьт]\b", "поправь"),
        ]
        for pat, repl in replacements:
            s = re.sub(pat, repl, s)
        return s

    @staticmethod
    def _normalize_russian_punctuation(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Normalize punctuation spacing.
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([,.;:!?])([^\s\n])", r"\1 \2", s)
        # Common etiquette punctuation.
        s = re.sub(r"(?iu),?\s*пожалуйста\s*,?", ", пожалуйста,", s)
        s = re.sub(r"\s*,\s*,+", ", ", s)
        s = re.sub(r"(?iu)(^|[.!?]\s),\s*пожалуйста,", r"\1Пожалуйста,", s)
        s = re.sub(r"(?iu)^пожалуйста,\s*", "Пожалуйста, ", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _enhance_text_quality(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Stronger but still conservative text polishing.
        rules = [
            (r"(?iu)\bвсе\s+внесет\b", "всё внесёт"),
            (r"(?iu)\bвсе\s+внесе[тд]\b", "всё внесёт"),
            (r"(?iu)\bexcel\b", "Excel"),
            (r"(?iu)\bфайл[иы]\b", "файл"),
            (r"(?iu)\bтаблиц[уы]\b", "таблицу"),
            (r"(?iu)\bдля\s+меня\b", "для меня"),
        ]
        for pat, repl in rules:
            s = re.sub(pat, repl, s)
        # Normalize conjunction punctuation in dictation patterns.
        s = re.sub(r"(?iu),\s*то\s+есть\s*", ", то есть ", s)
        s = re.sub(r"(?iu)\bну\s+то\s+есть\b", "то есть", s)
        s = re.sub(r"(?iu)\bи\s+все\b", "и всё", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    @staticmethod
    def _remove_disfluencies(text: str) -> str:
        s = text
        # Remove only explicit hesitation sounds; keep conjunctions and normal short words.
        s = re.sub(r"(?iu)(^|[\s,;:()\[\]\"'«»—-])(э+|эм+|мм+|ммм+)(?=[\s,;:.!?()\[\]\"'«»—-]|$)", r"\1", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([,;:]){2,}", r"\1", s)
        return s.strip()

    @staticmethod
    def _cleanup_discourse_text(text: str, strong: bool = False) -> str:
        s = str(text or "").strip()
        if not s:
            return ""

        # Soft cleanup only: removes common discourse fillers at sentence boundaries
        # and obvious repeated fragments without changing core semantics.
        sentence_start_fillers = [
            r"знаешь",
            r"слушай",
            r"смотри",
            r"ну",
            r"короче",
            r"в общем",
        ]
        for token in sentence_start_fillers:
            s = re.sub(rf"(?iu)(^|(?<=[.!?]\s)){token}\s*,?\s*", r"\1", s)

        # Remove frequent filler bundles inside phrases.
        s = re.sub(r"(?iu)\b(да,\s*)?то\s+есть\b[, ]*", "", s)
        s = re.sub(r"(?iu)\bкак\s+будто\s+бы\b[, ]*", "", s)
        s = re.sub(r"(?iu)\bпо\s+идее\b[, ]*", "", s)
        s = re.sub(r"(?iu)\bну\s+то\s+есть\b[, ]*", "", s)

        if strong:
            # Strong mode: remove additional conversational glue.
            s = re.sub(r"(?iu)\b(в\s+общем(?:-то)?|короче(?:\s+говоря)?|в\s+принципе|на\s+самом\s+деле)\b[, ]*", "", s)
            s = re.sub(r"(?iu)\b(если\s+честно|честно\s+говоря|как\s+бы)\b[, ]*", "", s)
            s = re.sub(r"(?iu)\b(ну|вот)\b(?=\s+[а-яёa-z])", "", s)

        # Collapse exact repeated short fragments (2..8 words) separated by punctuation.
        s = re.sub(
            r"(?iu)\b((?:[а-яёa-z0-9-]+\s+){1,7}[а-яёa-z0-9-]+)\b(?:\s*[,;:.!?-]\s*|\s+)\1\b",
            r"\1",
            s,
        )

        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([,;:]){2,}", r"\1", s)
        s = re.sub(r"(?iu)(^|[.!?]\s)([,;:])", r"\1", s)
        return s.strip(" ,;:")

    @staticmethod
    def _normalize_for_denoise(text: str) -> str:
        s = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"[ \t]{2,}", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(re.findall(r"(?iu)\b[0-9a-zа-яё][0-9a-zа-яё_-]*\b", str(text or "")))

    @staticmethod
    def _apply_on_unprotected_text(text: str, fn) -> str:
        # Keep quoted/code fragments intact while cleaning surrounding discourse.
        parts = re.split(r"(«[^»]*»|\"[^\"]*\"|'[^']*'|`[^`]*`)", str(text or ""))
        out: list[str] = []
        for idx, part in enumerate(parts):
            if idx % 2 == 1:
                out.append(part)
            else:
                out.append(fn(part))
        return "".join(out)

    @staticmethod
    def _extract_semantic_signature(text: str) -> set[str]:
        s = str(text or "")
        sig: set[str] = set()
        for m in re.findall(r"(?iu)\b\d+(?:[.,]\d+)?\b", s):
            sig.add(f"num:{m.lower()}")
        for m in re.findall(r"(?iu)\b[\w\.-]+@[\w\.-]+\.\w+\b", s):
            sig.add(f"mail:{m.lower()}")
        for m in re.findall(r"(?iu)\bhttps?://[^\s]+", s):
            sig.add(f"url:{m.lower().rstrip('.,;:!?')}")
        for m in re.findall(r"(?iu)\b[a-z0-9_./-]{3,}\b", s):
            if any(ch.isdigit() for ch in m) or any(ch in m for ch in "._/-"):
                sig.add(f"tech:{m.lower()}")
        for m in re.findall(r"(?iu)(?:«[^»]+»|\"[^\"]+\"|'[^']+'|`[^`]+`)", s):
            core = m.strip("«»\"'` \t")
            if core:
                sig.add(f"q:{core.lower()}")
        for m in re.findall(r"(?iu)\b(?:f\d{1,2}|ctrl|alt|shift|win)\b", s):
            sig.add(f"cmd:{m.lower()}")
        return sig

    @staticmethod
    def _semantic_loss_exceeds_threshold(before: str, after: str) -> bool:
        sig_before = OpenRouterTranscriber._extract_semantic_signature(before)
        if not sig_before:
            return False
        sig_after = OpenRouterTranscriber._extract_semantic_signature(after)
        lost = sig_before - sig_after
        # Maximum allowed loss for balanced profile: 10% of significant tokens, but no more than 2.
        allowed_loss = min(2, max(0, int(len(sig_before) * 0.10)))
        return len(lost) > allowed_loss

    @staticmethod
    def _denoise_text_balanced(text: str, strong: bool = False) -> tuple[str, dict]:
        src = OpenRouterTranscriber._normalize_for_denoise(text)
        if not src:
            return "", {"removed_tokens": 0, "repetition_collapses": 0}
        before_words = OpenRouterTranscriber._word_count(src)

        repetition_collapses = 0

        def _clean_segment(segment: str) -> str:
            nonlocal repetition_collapses
            s = segment
            # Hesitations: always remove in balanced mode outside protected spans.
            s = re.sub(
                r"(?iu)(^|[\s,;:()\[\]«»\"'`-])(э+|эм+|мм+|ммм+|ну+)(?=[\s,;:.!?()\[\]«»\"'`-]|$)",
                r"\1",
                s,
            )
            # Discourse glue: remove on sentence boundaries and as repeated fillers.
            s = re.sub(r"(?iu)(^|(?<=[.!?]\s))(знаешь|слушай|смотри|короче|в общем)\s*,?\s*", r"\1", s)
            s = re.sub(r"(?iu)\b(то есть|как бы|по идее|на самом деле)\b(?:\s*,\s*|\s+)", " ", s)
            if strong:
                s = re.sub(r"(?iu)\b(в принципе|если честно|честно говоря|по сути)\b(?:\s*,\s*|\s+)", " ", s)
            # Collapse exact repeated short fragments (1..8 words) separated by punctuation/spaces.
            rep_pat = re.compile(
                r"(?iu)\b((?:[а-яёa-z0-9-]+\s+){0,7}[а-яёa-z0-9-]+)\b(?:\s*[,;:.!?-]\s*|\s+)\1\b"
            )
            while True:
                s, n = rep_pat.subn(r"\1", s)
                repetition_collapses += int(n)
                if n == 0:
                    break
            s = re.sub(r"[ \t]{2,}", " ", s)
            s = re.sub(r"\s+([,.;:!?])", r"\1", s)
            s = re.sub(r"([,;:]){2,}", r"\1", s)
            s = re.sub(r"(?iu)(^|[.!?]\s)([,;:])", r"\1", s)
            return s.strip(" ,;:")

        out = OpenRouterTranscriber._apply_on_unprotected_text(src, _clean_segment)
        out = OpenRouterTranscriber._normalize_for_denoise(out)
        after_words = OpenRouterTranscriber._word_count(out)
        removed_tokens = max(0, before_words - after_words)
        return out, {
            "removed_tokens": removed_tokens,
            "repetition_collapses": repetition_collapses,
        }

    @staticmethod
    def _capitalize_sentence_starts(text: str) -> str:
        if not text:
            return ""
        # Capitalize the first alphabetic character at line start or after sentence-ending punctuation.
        return re.sub(
            r"(^|(?<=[\.\!\?…]\s)|(?<=\n))([a-zа-яё])",
            lambda m: m.group(1) + m.group(2).upper(),
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    @staticmethod
    def _insert_missing_quotes(text: str) -> str:
        if not text:
            return ""

        quote_chars = "\"'«»“”„?"
        speech_verbs = (
            "сказал(?:а|и)?|спросил(?:а|и)?|ответил(?:а|и)?|написал(?:а|и)?|"
            "подумал(?:а|и)?|говорит|говорю|сказали|спросили|ответили|"
            "said|asked|answered|wrote|thought|says"
        )
        pattern = re.compile(
            rf"(?iu)^(.{{0,300}}?\b(?:{speech_verbs})\b\s*:\s*)(.+)$"
        )

        out: list[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or any(ch in line for ch in quote_chars):
                out.append(raw_line)
                continue

            m = pattern.match(line)
            if not m:
                out.append(raw_line)
                continue

            prefix = m.group(1)
            tail = m.group(2).strip()
            if not tail:
                out.append(raw_line)
                continue
            if tail[:1] in quote_chars:
                out.append(raw_line)
                continue

            out.append(f"{prefix}В«{tail}В»")

        return "\n".join(out)

    @staticmethod
    def _normalize_quotes_to_guillemets(text: str) -> str:
        if not text:
            return ""
        s = text
        # Normalize common paired quote styles to Russian guillemets.
        s = re.sub(r'"([^"\n]+)"', r'В«\1В»', s)
        s = re.sub(r"“([^”\n]+)”", r'«\1»', s)
        s = re.sub(r"„([^”\n]+)”", r'«\1»', s)
        return s


class LocalWhisperSmallTranscriber(OpenRouterTranscriber):
    MODEL_NAME = "small"
    ALLOWED_MODELS = {"small", "medium", "large-v3"}
    REPO_PREFIX = "Systran/faster-whisper-"
    REQUIRED_MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")
    _download_lock = threading.Lock()

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._model = None
        self._model_lock = threading.Lock()
        self._download_root = self.get_download_root()
        self._runtime_lock = threading.Lock()
        self._runtime_state = "init"
        self._runtime_message = "Модель не загружена."
        self._selected_model = self.normalize_model_name(str(getattr(self.config, "local_whisper_model", self.MODEL_NAME) or self.MODEL_NAME))
        self._model_label = f"Whisper {self._selected_model} (локально)"
        if self.is_model_present(self._selected_model):
            self._runtime_state = "ready"
            self._runtime_message = f"Локальная модель Whisper {self._selected_model} скачана и готова."
        else:
            self._runtime_state = "init"
            self._runtime_message = f"Модель Whisper {self._selected_model} не загружена. Скачайте её в настройках."
        self._cpu_threads = max(2, min(16, (os.cpu_count() or 4)))
        self._force_cpu_only = False

    def _is_medium_turbo(self) -> bool:
        return bool(self._selected_model == "medium" and bool(getattr(self.config, "whisper_medium_turbo", False)))

    def _build_model_label(self) -> str:
        if self._selected_model == "medium":
            profile = "Turbo" if self._is_medium_turbo() else "Стандарт"
            return f"Whisper medium ({profile}, локально)"
        return f"Whisper {self._selected_model} (локально)"

    def _runtime_device_cfg(self) -> tuple[str, str]:
        if self._force_cpu_only:
            return "cpu", "int8"
        use_hw = bool(getattr(self.config, "hardware_acceleration", True))
        if use_hw and ctranslate2 is not None:
            try:
                if int(ctranslate2.get_cuda_device_count()) > 0:
                    # Guard against machines with a driver but missing CUDA runtime DLLs.
                    try:
                        ctypes.WinDLL("cublas64_12.dll")
                    except Exception:
                        return "cpu", "int8"
                    if self._is_medium_turbo():
                        return "cuda", "int8_float16"
                    return "cuda", "float16"
            except Exception:
                pass
        return "cpu", "int8"

    @classmethod
    def get_download_root(cls) -> Path:
        return CONFIG_DIR / "models" / "faster-whisper"

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        model = str(model_name or "").strip().lower()
        return model if model in cls.ALLOWED_MODELS else cls.MODEL_NAME

    @classmethod
    def repo_id_for_model(cls, model_name: str) -> str:
        model = cls.normalize_model_name(model_name)
        return f"{cls.REPO_PREFIX}{model}"

    @classmethod
    def model_dir(cls, model_name: str) -> Path:
        model = cls.normalize_model_name(model_name)
        return cls.get_download_root() / model

    @classmethod
    def is_model_present(cls, model_name: str) -> bool:
        model_path = cls.model_dir(model_name)
        if not model_path.exists():
            return False
        for required in cls.REQUIRED_MODEL_FILES:
            if not (model_path / required).exists():
                return False
        return True

    @classmethod
    def model_cached_size_bytes(cls, model_name: str) -> int:
        root = cls.model_dir(model_name)
        if not root.exists():
            return 0
        total = 0
        for p in root.rglob("*"):
            try:
                if p.is_file():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        return total

    @classmethod
    def download_model_snapshot(cls, model_name: str, progress_cb=None) -> Path:
        model = cls.normalize_model_name(model_name)
        if HfApi is None:
            raise RuntimeError("Модуль huggingface_hub не найден в сборке приложения")
        target = cls.model_dir(model)
        target.mkdir(parents=True, exist_ok=True)
        repo_id = cls.repo_id_for_model(model)
        with cls._download_lock:
            api = HfApi()
            info = api.model_info(repo_id, files_metadata=False)
            siblings = list(getattr(info, "siblings", []) or [])
            # GUI-safe download path:
            # - no tqdm/progress console rendering
            # - no symlink mode (Windows admin rights not required)
            for sibling in siblings:
                rfilename = str(getattr(sibling, "rfilename", "") or "").strip()
                if not rfilename:
                    continue
                relative_name = rfilename.lstrip("/\\").replace("\\", "/")
                out_path = target / Path(relative_name)
                if out_path.exists() and out_path.is_file() and out_path.stat().st_size > 0:
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")
                url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(relative_name, safe='/')}"
                try:
                    with requests.get(url, stream=True, timeout=(20, 240)) as resp:
                        resp.raise_for_status()
                        total_bytes = int(resp.headers.get("Content-Length") or 0)
                        downloaded_bytes = 0
                        with open(tmp_path, "wb") as tmp:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    tmp.write(chunk)
                                    downloaded_bytes += len(chunk)
                                    if callable(progress_cb):
                                        try:
                                            progress_cb(
                                                state="downloading",
                                                downloaded=downloaded_bytes,
                                                total=total_bytes,
                                                message=f"Скачивание файла {relative_name}...",
                                            )
                                        except Exception:
                                            pass
                    os.replace(tmp_path, out_path)
                except Exception:
                    try:
                        if tmp_path.exists():
                            tmp_path.unlink()
                    except Exception:
                        pass
                    raise
        return target

    @classmethod
    def remove_cached_model(cls, model_name: str) -> int:
        model = cls.normalize_model_name(model_name)
        root = cls.model_dir(model)
        if not root.exists():
            return 0
        try:
            shutil.rmtree(root, ignore_errors=False)
            return 1
        except Exception:
            logging.exception("Failed to remove local model folder: %s", root)
            return 0

    @staticmethod
    def supports_live_preview() -> bool:
        # Local mode prioritizes stability and predictable latency.
        return False

    def _set_runtime(self, state: str, message: str) -> None:
        with self._runtime_lock:
            self._runtime_state = str(state or "init")
            self._runtime_message = str(message or "")

    def get_runtime_status(self) -> dict:
        with self._runtime_lock:
            state = str(self._runtime_state or "init")
            msg = str(self._runtime_message or "")
        downloaded = False
        try:
            downloaded = bool(self.is_model_present(self._selected_model))
        except Exception:
            downloaded = False
        loaded = self._model is not None
        if loaded:
            state = "ready"
            if not msg:
                msg = f"Локальная модель Whisper {self._selected_model} загружена."
        elif downloaded and state not in {"loading", "error"}:
            state = "ready"
            if not msg:
                msg = f"Локальная модель Whisper {self._selected_model} скачана и готова."
        elif not downloaded and state not in {"loading", "error"}:
            state = "init"
            if not msg:
                msg = f"Модель Whisper {self._selected_model} не загружена. Скачайте её в настройках."
        model_label = self._build_model_label()
        return {
            "backend": "local",
            "engine": "whisper",
            "model_label": model_label,
            "ready": state == "ready",
            "state": state,
            "message": msg,
            "model_loaded": loaded,
            "model_downloaded": downloaded,
        }

    def _ensure_model(self):
        if self._model is not None:
            self._set_runtime("ready", f"Локальная модель Whisper {self._selected_model} загружена.")
            return self._model
        if WhisperModel is None:
            self._set_runtime("error", "Не найден faster-whisper в сборке приложения.")
            raise RuntimeError(
                "Локальный режим недоступен: не установлен faster-whisper. "
                "Переустановите приложение с локальным движком."
            )
        with self._model_lock:
            if self._model is None:
                self._set_runtime("loading", f"Загрузка локальной модели Whisper {self._selected_model}...")
                try:
                    model_path = self.model_dir(self._selected_model)
                    if not self.is_model_present(self._selected_model):
                        self.download_model_snapshot(self._selected_model)
                    device, compute_type = self._runtime_device_cfg()
                    # Single-request desktop dictation has better latency with one worker.
                    # Multiple workers may increase contention and tail latency on some CPUs.
                    workers = 1
                    self._model = WhisperModel(
                        str(model_path),
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=self._cpu_threads,
                        num_workers=workers,
                    )
                except Exception as err:
                    self._set_runtime("error", f"Ошибка загрузки локальной модели: {err}")
                    raise
                self._set_runtime("ready", f"Локальная модель Whisper {self._selected_model} загружена.")
        return self._model

    @staticmethod
    def _write_temp_wav(wav_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            return tmp.name

    @staticmethod
    def _preprocess_audio_balanced(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        # "Баланс/скорость": легкая нормализация + downsample до 16k для быстрого local inference.
        try:
            if audio is None:
                return np.zeros((0,), dtype=np.float32), int(sample_rate)
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
            if pcm.size == 0:
                return pcm, int(sample_rate)

            # Remove DC offset.
            pcm = pcm - float(np.mean(pcm))

            # Soft amplitude normalization with clip protection.
            peak = float(np.max(np.abs(pcm)))
            if peak > 1e-6:
                target_peak = 0.85
                gain = target_peak / peak
                gain = max(0.4, min(8.0, gain))
                pcm = pcm * gain

            pcm = np.clip(pcm, -0.98, 0.98).astype(np.float32, copy=False)
            src_rate = max(8000, int(sample_rate))
            target_sr = 16000
            if src_rate != target_sr and pcm.size > 8:
                try:
                    duration = pcm.size / float(src_rate)
                    target_len = max(1, int(round(duration * target_sr)))
                    src_x = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False, dtype=np.float32)
                    dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
                    pcm = np.interp(dst_x, src_x, pcm).astype(np.float32, copy=False)
                    src_rate = target_sr
                except Exception:
                    pass
            return pcm, src_rate
        except Exception:
            logging.exception("Local preprocess failed; using raw audio")
            return np.asarray(audio, dtype=np.float32).reshape(-1), max(8000, int(sample_rate))

    @staticmethod
    def _trim_silence(audio: np.ndarray, sample_rate: int, threshold: float = 0.008, margin_sec: float = 0.08) -> np.ndarray:
        # Lightweight silence trimming to reduce local inference time on long pauses.
        try:
            pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
            if pcm.size < 64:
                return pcm
            abs_pcm = np.abs(pcm)
            active = np.where(abs_pcm > threshold)[0]
            if active.size == 0:
                return pcm
            margin = int(max(1, sample_rate * float(max(0.0, margin_sec))))
            start = max(0, int(active[0]) - margin)
            end = min(pcm.size, int(active[-1]) + margin + 1)
            if end <= start:
                return pcm
            return pcm[start:end]
        except Exception:
            return np.asarray(audio, dtype=np.float32).reshape(-1)

    @classmethod
    def _preprocess_wav_bytes_balanced(cls, wav_bytes: bytes) -> bytes:
        # Fallback path for environments where ndarray fast-path is unavailable.
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                channels = int(wav.getnchannels())
                sample_width = int(wav.getsampwidth())
                sample_rate = int(wav.getframerate())
                frames = wav.readframes(wav.getnframes())
            if channels != 1 or sample_width != 2:
                return wav_bytes
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            processed, processed_rate = cls._preprocess_audio_balanced(pcm, sample_rate)
            return float_audio_to_wav_bytes(processed, processed_rate)
        except Exception:
            return wav_bytes

    def transcribe(self, wav_bytes_or_audio, duration_sec: float, sample_rate: Optional[int] = None) -> tuple[str, dict]:
        self._set_runtime("loading", "Распознаю локально...")
        total_t0 = time.perf_counter()
        model = self._ensure_model()
        prepare_t0 = time.perf_counter()
        tmp_path = None
        infer_input = None
        infer_input_desc = "ndarray"
        effective_sr = max(8000, int(sample_rate or getattr(self.config, "sample_rate", 24000)))
        medium_turbo = self._is_medium_turbo()
        if isinstance(wav_bytes_or_audio, np.ndarray):
            processed_audio, _processed_rate = self._preprocess_audio_balanced(wav_bytes_or_audio, effective_sr)
            if medium_turbo:
                infer_input = self._trim_silence(processed_audio, _processed_rate, threshold=0.010, margin_sec=0.05)
            else:
                infer_input = self._trim_silence(processed_audio, _processed_rate)
        else:
            wav_bytes = bytes(wav_bytes_or_audio or b"")
            preprocessed = self._preprocess_wav_bytes_balanced(wav_bytes)
            tmp_path = self._write_temp_wav(preprocessed)
            infer_input = tmp_path
            infer_input_desc = "wav_file"
        prepare_t1 = time.perf_counter()
        try:
            quality_mode = bool(getattr(self.config, "max_text_quality", False))
            punctuate_mode = bool(getattr(self.config, "punctuate_text", True))
            clip_duration = max(0.0, float(duration_sec or 0.0))

            # Local decoding profile:
            # - default stays fast
            # - punctuation-aware mode gets a slightly stronger beam for better punctuation/word endings
            # - max quality enables the most accurate local profile
            if self._selected_model == "small":
                # Adaptive low-latency profile:
                # short clips can afford stronger decoding, long clips stay fast.
                if punctuate_mode and clip_duration <= 6.5:
                    beam_size = 2
                    best_of = 2
                else:
                    beam_size = 1
                    best_of = 1
                if quality_mode:
                    beam_size = 2
                    best_of = 2
            elif self._selected_model == "medium":
                if medium_turbo:
                    beam_size = 1
                    best_of = 1
                    if clip_duration > 12.0:
                        # Keep long clips in strict speed profile.
                        beam_size = 1
                        best_of = 1
                else:
                    beam_size = 1
                    best_of = 1
                    if quality_mode:
                        beam_size = 2
                        best_of = 2
            else:
                beam_size = 2
                best_of = 1
                if quality_mode:
                    beam_size = 3
                    best_of = 2

            if medium_turbo:
                condition_on_previous = False
            else:
                condition_on_previous = bool(quality_mode and clip_duration <= 14.0)
            initial_prompt = None
            if punctuate_mode and not (medium_turbo and clip_duration > 12.0):
                initial_prompt = (
                    "Русский литературный текст. Сохраняй исходные слова, "
                    "ставь естественные знаки препинания и корректные окончания слов."
                )
                if quality_mode and not medium_turbo:
                    initial_prompt += " Повышенный приоритет: грамматическая корректность и пунктуация."

            def _run_once(model_obj):
                return model_obj.transcribe(
                    infer_input,
                    language="ru",
                    task="transcribe",
                    beam_size=beam_size,
                    best_of=best_of,
                    temperature=0.0,
                    condition_on_previous_text=condition_on_previous,
                    without_timestamps=True,
                    word_timestamps=False,
                    initial_prompt=initial_prompt,
                    # In packaged Windows builds VAD asset lookup can fail on some machines.
                    # Keep local mode stable by disabling external VAD dependency.
                    vad_filter=False,
                )

            try:
                infer_t0 = time.perf_counter()
                segments, _info = _run_once(model)
                segments = list(segments)
                infer_t1 = time.perf_counter()
            except Exception as first_err:
                err_text = str(first_err).lower()
                cuda_failure = ("cublas" in err_text) or ("cuda" in err_text and "not found" in err_text)
                if not cuda_failure:
                    raise
                # Runtime fallback: if CUDA path is unavailable on user machine,
                # transparently switch to CPU and retry once.
                self._set_runtime("loading", "CUDA недоступна, переключаюсь на CPU...")
                self._force_cpu_only = True
                with self._model_lock:
                    self._model = None
                model = self._ensure_model()
                infer_t0 = time.perf_counter()
                segments, _info = _run_once(model)
                segments = list(segments)
                infer_t1 = time.perf_counter()
            post_t0 = time.perf_counter()
            raw_text = " ".join((str(seg.text or "").strip() for seg in segments)).strip()
            if not raw_text:
                self._set_runtime("ready", "Локальная модель загружена. Речь не распознана.")
                return "", {"input_tokens": 0, "output_tokens": 0}
            self._set_runtime("ready", "Локальная модель загружена.")
            out_text = self._postprocess_text(raw_text)
            post_t1 = time.perf_counter()
            total_t1 = time.perf_counter()
            logging.info(
                "Local transcribe timing: input=%s prepare_audio_ms=%.1f infer_ms=%.1f postprocess_ms=%.1f total_ms=%.1f",
                infer_input_desc,
                (prepare_t1 - prepare_t0) * 1000.0,
                (infer_t1 - infer_t0) * 1000.0,
                (post_t1 - post_t0) * 1000.0,
                (total_t1 - total_t0) * 1000.0,
            )
            return out_text, {"input_tokens": 0, "output_tokens": 0}
        except Exception as err:
            self._set_runtime("error", f"Ошибка локального распознавания: {err}")
            raise RuntimeError(f"Локальное распознавание недоступно: {err}")
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def transcribe_live_preview(self, wav_bytes: bytes, duration_sec: float) -> str:
        return ""


class LocalVoskTranscriber(OpenRouterTranscriber):
    MODEL_NAME = "vosk-model-small-ru-0.22"
    MODEL_URLS = {
        "vosk-model-small-ru-0.22": "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
        "vosk-model-ru-0.42": "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip",
    }
    REQUIRED_MODEL_FILES = ("am/final.mdl", "conf/model.conf")
    _download_lock = threading.Lock()

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._model = None
        self._model_lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._runtime_state = "init"
        self._runtime_message = "Модель Vosk не загружена."
        self._selected_model = self.normalize_model_name(str(getattr(self.config, "local_vosk_model", self.MODEL_NAME) or self.MODEL_NAME))
        self._model_label = f"Vosk {self._selected_model} (локально)"
        if self.is_model_present(self._selected_model):
            self._runtime_state = "ready"
            self._runtime_message = f"Локальная модель Vosk {self._selected_model} скачана и готова."
        if callable(vosk_set_log_level):
            try:
                vosk_set_log_level(-1)
            except Exception:
                pass

    @staticmethod
    def supports_live_preview() -> bool:
        return False

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        model = str(model_name or "").strip().lower()
        return model if model in cls.MODEL_URLS else cls.MODEL_NAME

    @classmethod
    def get_download_root(cls) -> Path:
        return CONFIG_DIR / "models" / "vosk"

    @classmethod
    def model_dir(cls, model_name: str) -> Path:
        model = cls.normalize_model_name(model_name)
        return cls.get_download_root() / model

    @classmethod
    def is_model_present(cls, model_name: str) -> bool:
        root = cls.model_dir(model_name)
        if not root.exists():
            return False
        for required in cls.REQUIRED_MODEL_FILES:
            if not (root / required).exists():
                return False
        return True

    @classmethod
    def model_cached_size_bytes(cls, model_name: str) -> int:
        root = cls.model_dir(model_name)
        if not root.exists():
            return 0
        total = 0
        for p in root.rglob("*"):
            try:
                if p.is_file():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        return total

    @classmethod
    def _pick_model_root(cls, root_dir: Path) -> Path:
        if any((root_dir / rel).exists() for rel in cls.REQUIRED_MODEL_FILES):
            return root_dir
        for child in root_dir.iterdir():
            if not child.is_dir():
                continue
            if any((child / rel).exists() for rel in cls.REQUIRED_MODEL_FILES):
                return child
        return root_dir

    @classmethod
    def download_model_snapshot(cls, model_name: str, progress_cb=None) -> Path:
        model = cls.normalize_model_name(model_name)
        url = cls.MODEL_URLS.get(model)
        if not url:
            raise RuntimeError(f"Неизвестная модель Vosk: {model}")
        target = cls.model_dir(model)
        target.mkdir(parents=True, exist_ok=True)
        with cls._download_lock:
            if cls.is_model_present(model):
                return target
            zip_path = target / f"{model}.zip"
            tmp_zip = target / f"{model}.zip.part"
            try:
                with requests.get(url, stream=True, timeout=(20, 300)) as resp:
                    resp.raise_for_status()
                    total_bytes = int(resp.headers.get("Content-Length") or 0)
                    written_bytes = 0
                    with open(tmp_zip, "wb") as tmp:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                tmp.write(chunk)
                                written_bytes += len(chunk)
                                if callable(progress_cb):
                                    try:
                                        progress_cb(
                                            state="downloading",
                                            downloaded=written_bytes,
                                            total=total_bytes,
                                            message=f"Скачивание модели {model}...",
                                        )
                                    except Exception:
                                        pass
                os.replace(tmp_zip, zip_path)
                if callable(progress_cb):
                    try:
                        progress_cb(
                            state="downloading",
                            downloaded=written_bytes,
                            total=max(total_bytes, written_bytes),
                            message=f"Распаковка модели {model}...",
                        )
                    except Exception:
                        pass
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(target)
                extracted_root = cls._pick_model_root(target)
                if extracted_root != target:
                    for item in extracted_root.iterdir():
                        dst = target / item.name
                        if dst.exists():
                            if dst.is_dir():
                                shutil.rmtree(dst, ignore_errors=True)
                            else:
                                dst.unlink(missing_ok=True)
                        shutil.move(str(item), str(dst))
                    shutil.rmtree(extracted_root, ignore_errors=True)
                if callable(progress_cb):
                    try:
                        final_size = cls.model_cached_size_bytes(model)
                        progress_cb(
                            state="downloading",
                            downloaded=final_size,
                            total=max(final_size, total_bytes, written_bytes),
                            message=f"Проверка файлов модели {model}...",
                        )
                    except Exception:
                        pass
            finally:
                try:
                    if tmp_zip.exists():
                        tmp_zip.unlink()
                except Exception:
                    pass
                try:
                    if zip_path.exists():
                        zip_path.unlink()
                except Exception:
                    pass
        if not cls.is_model_present(model):
            raise RuntimeError("После загрузки модель Vosk повреждена или неполная.")
        return target

    @classmethod
    def remove_cached_model(cls, model_name: str) -> int:
        model = cls.normalize_model_name(model_name)
        root = cls.model_dir(model)
        if not root.exists():
            return 0
        try:
            shutil.rmtree(root, ignore_errors=False)
            return 1
        except Exception:
            logging.exception("Failed to remove Vosk model folder: %s", root)
            return 0

    def _set_runtime(self, state: str, message: str) -> None:
        with self._runtime_lock:
            self._runtime_state = str(state or "init")
            self._runtime_message = str(message or "")

    def get_runtime_status(self) -> dict:
        with self._runtime_lock:
            state = str(self._runtime_state or "init")
            msg = str(self._runtime_message or "")
        downloaded = False
        try:
            downloaded = bool(self.is_model_present(self._selected_model))
        except Exception:
            downloaded = False
        loaded = self._model is not None
        if loaded:
            state = "ready"
            if not msg:
                msg = f"Локальная модель Vosk {self._selected_model} загружена."
        elif downloaded and state not in {"loading", "error"}:
            state = "ready"
            if not msg:
                msg = f"Локальная модель Vosk {self._selected_model} скачана и готова."
        elif not downloaded and state not in {"loading", "error"}:
            state = "init"
            if not msg:
                msg = f"Модель Vosk {self._selected_model} не загружена. Скачайте её в настройках."
        return {
            "backend": "local",
            "engine": "vosk",
            "model_label": self._model_label,
            "ready": state == "ready",
            "state": state,
            "message": msg,
            "model_loaded": loaded,
            "model_downloaded": downloaded,
        }

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        src_rate = max(8000, int(sample_rate))
        if pcm.size <= 1 or src_rate == 16000:
            return pcm
        duration = pcm.size / float(src_rate)
        target_len = max(1, int(round(duration * 16000)))
        src_x = np.linspace(0.0, 1.0, num=pcm.size, endpoint=False, dtype=np.float32)
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
        return np.interp(dst_x, src_x, pcm).astype(np.float32, copy=False)

    def _ensure_model(self):
        if self._model is not None:
            self._set_runtime("ready", f"Локальная модель Vosk {self._selected_model} загружена.")
            return self._model
        if VoskModel is None:
            details = f" Причина: {VOSK_IMPORT_ERROR}" if VOSK_IMPORT_ERROR else ""
            self._set_runtime("error", f"Не найден vosk в сборке приложения.{details}")
            raise RuntimeError(f"Локальный режим Vosk недоступен: не установлен модуль vosk.{details}")
        with self._model_lock:
            if self._model is None:
                self._set_runtime("loading", f"Загрузка локальной модели Vosk {self._selected_model}...")
                try:
                    model_path = self.model_dir(self._selected_model)
                    if not self.is_model_present(self._selected_model):
                        self.download_model_snapshot(self._selected_model)
                    self._model = VoskModel(str(model_path))
                except Exception as err:
                    self._set_runtime("error", f"Ошибка загрузки Vosk: {err}")
                    raise
                self._set_runtime("ready", f"Локальная модель Vosk {self._selected_model} загружена.")
        return self._model

    def transcribe(self, wav_bytes_or_audio, duration_sec: float, sample_rate: Optional[int] = None) -> tuple[str, dict]:
        self._set_runtime("loading", "Распознаю локально (Vosk)...")
        model = self._ensure_model()
        if isinstance(wav_bytes_or_audio, np.ndarray):
            audio = np.asarray(wav_bytes_or_audio, dtype=np.float32).reshape(-1)
            sr = max(8000, int(sample_rate or getattr(self.config, "sample_rate", 24000)))
        else:
            with wave.open(io.BytesIO(bytes(wav_bytes_or_audio or b"")), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                sr = int(wav.getframerate())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        audio16k = self._resample_to_16k(audio, sr)
        pcm16 = (np.clip(audio16k, -1.0, 1.0) * 32767.0).astype(np.int16, copy=False)
        rec = KaldiRecognizer(model, 16000.0)
        parts: list[str] = []
        try:
            chunk = 4000
            raw = pcm16.tobytes()
            step = chunk * 2
            for i in range(0, len(raw), step):
                chunk_bytes = raw[i:i + step]
                if rec.AcceptWaveform(chunk_bytes):
                    piece = json.loads(rec.Result() or "{}").get("text", "")
                    if piece:
                        parts.append(str(piece).strip())
            final_piece = json.loads(rec.FinalResult() or "{}").get("text", "")
            if final_piece:
                parts.append(str(final_piece).strip())
        except Exception as err:
            self._set_runtime("error", f"Ошибка локального распознавания Vosk: {err}")
            raise RuntimeError(f"Локальное распознавание Vosk недоступно: {err}")
        text = " ".join(p for p in parts if p).strip()
        if not text:
            self._set_runtime("ready", "Локальная модель Vosk загружена. Речь не распознана.")
            return "", {"input_tokens": 0, "output_tokens": 0}
        self._set_runtime("ready", "Локальная модель Vosk загружена.")
        out_text = self._postprocess_text(text)
        return out_text, {"input_tokens": 0, "output_tokens": 0}

    def transcribe_live_preview(self, wav_bytes: bytes, duration_sec: float) -> str:
        return ""


class LocalSherpaWhispeRuTranscriber(OpenRouterTranscriber):
    MODEL_NAME = "voicepro-gigaam-rnnt"
    LEGACY_MODEL_NAMES = {"whisperu-gigaam-rnnt"}
    MODEL_HINT_LABEL = "GigaAM RNNT"
    MODEL_REPOS = {
        # Public Sherpa ONNX Russian RNNT model (standalone, no external app dependency).
        "voicepro-gigaam-rnnt": "csukuangfj/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19",
    }
    REQUIRED_MODEL_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")
    REQUIRED_FILE_ALIASES = {
        "encoder.onnx": ("encoder.onnx", "encoder.int8.onnx"),
        "decoder.onnx": ("decoder.onnx",),
        "joiner.onnx": ("joiner.onnx",),
        "tokens.txt": ("tokens.txt",),
    }
    OPTIONAL_MODEL_FILES = ("silero_vad.onnx",)
    _download_lock = threading.Lock()

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._recognizer = None
        self._recognizer_lock = threading.Lock()
        self._runtime_lock = threading.Lock()
        self._runtime_state = "init"
        self._runtime_message = "Модель Sherpa не загружена."
        self._selected_model = self.normalize_model_name(str(getattr(self.config, "local_sherpa_model", self.MODEL_NAME) or self.MODEL_NAME))
        self._model_label = f"Sherpa {self.MODEL_HINT_LABEL} (локально)"
        if self.is_model_present(self._selected_model):
            self._runtime_state = "ready"
            self._runtime_message = "Локальная модель Sherpa скачана и готова."
        else:
            self._runtime_state = "init"
            self._runtime_message = "Модель Sherpa не загружена. Нажмите «Скачать модель»."
        # Stable fast profile: use more CPU threads for long local decode on modern desktop CPUs.
        # Capped to keep UI responsive while improving latency on long chunks.
        self._cpu_threads = max(4, min(12, (os.cpu_count() or 4)))

    @staticmethod
    def supports_live_preview() -> bool:
        return False

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        model = str(model_name or "").strip().lower()
        if model in cls.LEGACY_MODEL_NAMES:
            return cls.MODEL_NAME
        return model if model == cls.MODEL_NAME else cls.MODEL_NAME

    @classmethod
    def get_download_root(cls) -> Path:
        return CONFIG_DIR / "models" / "sherpa-onnx"

    @classmethod
    def model_dir(cls, model_name: str) -> Path:
        model = cls.normalize_model_name(model_name)
        root = cls.get_download_root()
        canonical = root / model
        if canonical.exists():
            return canonical
        # Backward compatibility: migrate old folder naming to branded naming.
        for legacy in cls.LEGACY_MODEL_NAMES:
            legacy_dir = root / legacy
            if legacy_dir.exists():
                try:
                    canonical.parent.mkdir(parents=True, exist_ok=True)
                    legacy_dir.replace(canonical)
                    return canonical
                except Exception:
                    return legacy_dir
        return canonical

    @classmethod
    def _bundled_source_dir(cls) -> Optional[Path]:
        candidates: list[Path] = []
        try:
            if getattr(sys, "frozen", False):
                mei = Path(getattr(sys, "_MEIPASS", "") or "")
                if str(mei):
                    candidates.append(mei / "voice_pro_ai" / "bundled_models" / "sherpa_rnnt")
            app_dir = Path(__file__).resolve().parent
            candidates.append(app_dir / "bundled_models" / "sherpa_rnnt")
        except Exception:
            pass
        for p in candidates:
            try:
                if p.exists() and all((p / req).exists() for req in cls.REQUIRED_MODEL_FILES):
                    return p
            except Exception:
                continue
        return None

    @classmethod
    def _repo_for_model(cls, model_name: str) -> str:
        model = cls.normalize_model_name(model_name)
        return str(cls.MODEL_REPOS.get(model, "")).strip()

    @classmethod
    def _resolve_file_path(cls, root: Path, required_name: str) -> Optional[Path]:
        aliases = tuple(cls.REQUIRED_FILE_ALIASES.get(required_name, (required_name,)))
        for alias in aliases:
            p = root / alias
            if p.exists() and p.is_file():
                return p
        return None

    @classmethod
    def model_cached_size_bytes(cls, model_name: str) -> int:
        root = cls.model_dir(model_name)
        if not root.exists():
            return 0
        total = 0
        for p in root.rglob("*"):
            try:
                if p.is_file():
                    total += int(p.stat().st_size)
            except Exception:
                continue
        return total

    @classmethod
    def is_model_present(cls, model_name: str) -> bool:
        root = cls.model_dir(model_name)
        if not root.exists():
            return False
        for required in cls.REQUIRED_MODEL_FILES:
            if cls._resolve_file_path(root, required) is None:
                return False
        return True

    @classmethod
    def download_model_snapshot(cls, model_name: str, progress_cb=None) -> Path:
        model = cls.normalize_model_name(model_name)
        target = cls.model_dir(model)
        target.mkdir(parents=True, exist_ok=True)
        with cls._download_lock:
            if cls.is_model_present(model):
                return target
            bundled = cls._bundled_source_dir()
            if bundled is not None:
                files_to_copy = list(cls.REQUIRED_MODEL_FILES) + [f for f in cls.OPTIONAL_MODEL_FILES if (bundled / f).exists()]
                total_bytes = 0
                for filename in files_to_copy:
                    try:
                        total_bytes += int((bundled / filename).stat().st_size)
                    except Exception:
                        continue
                copied = 0
                for filename in files_to_copy:
                    src = bundled / filename
                    dst = target / filename
                    if not src.exists():
                        continue
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    try:
                        with open(src, "rb") as in_f, open(tmp, "wb") as out_f:
                            while True:
                                chunk = in_f.read(1024 * 1024)
                                if not chunk:
                                    break
                                out_f.write(chunk)
                                copied += len(chunk)
                                if callable(progress_cb):
                                    try:
                                        progress_cb(
                                            state="downloading",
                                            downloaded=copied,
                                            total=max(total_bytes, copied),
                                            message=f"Подготовка встроенной модели {filename}...",
                                        )
                                    except Exception:
                                        pass
                        os.replace(tmp, dst)
                    except Exception:
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass
                        raise
                if cls.is_model_present(model):
                    return target
            repo_id = cls._repo_for_model(model)
            if not repo_id:
                raise RuntimeError(f"Не задан источник для модели Sherpa: {model}")
            if HfApi is None:
                raise RuntimeError("Модуль huggingface_hub не найден в сборке приложения")

            api = HfApi()
            info = api.model_info(repo_id, files_metadata=True)
            siblings = list(getattr(info, "siblings", []) or [])
            sibling_map: dict[str, object] = {
                str(getattr(s, "rfilename", "") or "").strip(): s
                for s in siblings
                if str(getattr(s, "rfilename", "") or "").strip()
            }

            selected_sources: list[tuple[str, str, int]] = []
            for required_name in cls.REQUIRED_MODEL_FILES:
                aliases = tuple(cls.REQUIRED_FILE_ALIASES.get(required_name, (required_name,)))
                src_name = ""
                src_size = 0
                for alias in aliases:
                    sib = sibling_map.get(alias)
                    if sib is None:
                        continue
                    src_name = alias
                    try:
                        src_size = int(getattr(sib, "size", 0) or 0)
                    except Exception:
                        src_size = 0
                    break
                if not src_name:
                    raise RuntimeError(f"В репозитории {repo_id} отсутствует обязательный файл: {required_name}")
                selected_sources.append((src_name, required_name, src_size))

            for optional_name in cls.OPTIONAL_MODEL_FILES:
                sib = sibling_map.get(optional_name)
                if sib is None:
                    continue
                try:
                    sz = int(getattr(sib, "size", 0) or 0)
                except Exception:
                    sz = 0
                selected_sources.append((optional_name, optional_name, sz))

            total_bytes = sum(max(0, int(sz or 0)) for _, _, sz in selected_sources)
            downloaded_total = 0
            for src_name, out_name, known_size in selected_sources:
                url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(src_name, safe='/')}"
                dst = target / out_name
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + ".part")
                try:
                    with requests.get(url, stream=True, timeout=(20, 300)) as resp:
                        resp.raise_for_status()
                        content_len = int(resp.headers.get("Content-Length") or 0)
                        file_total = content_len if content_len > 0 else int(known_size or 0)
                        file_downloaded = 0
                        with open(tmp, "wb") as out_f:
                            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                out_f.write(chunk)
                                file_downloaded += len(chunk)
                                if callable(progress_cb):
                                    try:
                                        progress_cb(
                                            state="downloading",
                                            downloaded=int(downloaded_total + file_downloaded),
                                            total=max(int(total_bytes), int(downloaded_total + file_total), int(downloaded_total + file_downloaded)),
                                            message=f"Скачивание {out_name}...",
                                        )
                                    except Exception:
                                        pass
                    os.replace(tmp, dst)
                    downloaded_total += max(file_downloaded, file_total)
                except Exception:
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    raise
        if not cls.is_model_present(model):
            raise RuntimeError("После скачивания модель Sherpa повреждена или неполная.")
        return target

    @classmethod
    def remove_cached_model(cls, model_name: str) -> int:
        model = cls.normalize_model_name(model_name)
        root = cls.model_dir(model)
        if not root.exists():
            return 0
        try:
            shutil.rmtree(root, ignore_errors=False)
            return 1
        except Exception:
            logging.exception("Failed to remove sherpa model folder: %s", root)
            return 0

    def _set_runtime(self, state: str, message: str) -> None:
        with self._runtime_lock:
            self._runtime_state = str(state or "init")
            self._runtime_message = str(message or "")

    def get_runtime_status(self) -> dict:
        with self._runtime_lock:
            state = str(self._runtime_state or "init")
            msg = str(self._runtime_message or "")
        downloaded = False
        try:
            downloaded = bool(self.is_model_present(self._selected_model))
        except Exception:
            downloaded = False
        loaded = self._recognizer is not None
        if loaded:
            state = "ready"
            if not msg:
                msg = "Локальная модель Sherpa загружена."
        elif downloaded and state not in {"loading", "error"}:
            state = "ready"
            if not msg:
                msg = "Локальная модель Sherpa скачана и готова."
        elif not downloaded and state not in {"loading", "error"}:
            state = "init"
            if not msg:
                msg = "Модель Sherpa не загружена. Нажмите «Скачать модель»."
        return {
            "backend": "local",
            "engine": "sherpa",
            "model_label": self._model_label,
            "ready": state == "ready",
            "state": state,
            "message": msg,
            "model_loaded": loaded,
            "model_downloaded": downloaded,
        }

    def _provider(self) -> str:
        # Keep deterministic latency profile close to WhispeRu.
        return "cpu"

    def _ensure_recognizer(self):
        if self._recognizer is not None:
            self._set_runtime("ready", "Локальная модель Sherpa загружена.")
            return self._recognizer
        if sherpa_onnx is None:
            details = f" Причина: {SHERPA_IMPORT_ERROR}" if SHERPA_IMPORT_ERROR else ""
            self._set_runtime("error", f"Не найден sherpa-onnx в сборке приложения.{details}")
            raise RuntimeError(f"Локальный режим Sherpa недоступен: не установлен модуль sherpa-onnx.{details}")
        with self._recognizer_lock:
            if self._recognizer is None:
                self._set_runtime("loading", "Загрузка локальной модели Sherpa...")
                try:
                    model_path = self.model_dir(self._selected_model)
                    if not self.is_model_present(self._selected_model):
                        self.download_model_snapshot(self._selected_model)
                    provider = self._provider()
                    self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                        encoder=str(model_path / "encoder.onnx"),
                        decoder=str(model_path / "decoder.onnx"),
                        joiner=str(model_path / "joiner.onnx"),
                        tokens=str(model_path / "tokens.txt"),
                        num_threads=self._cpu_threads,
                        sample_rate=16000,
                        feature_dim=64,
                        decoding_method="greedy_search",
                        model_type="nemo_transducer",
                        provider=provider,
                    )
                except Exception as err:
                    self._set_runtime("error", f"Ошибка загрузки Sherpa: {err}")
                    raise
                self._set_runtime("ready", "Локальная модель Sherpa загружена.")
        return self._recognizer

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        src_rate = max(8000, int(sample_rate))
        if pcm.size <= 1 or src_rate == 16000:
            return pcm
        # Fast path for common integer downsample ratios (e.g., 32k->16k, 48k->16k).
        if src_rate > 16000 and src_rate % 16000 == 0:
            step = max(1, int(src_rate // 16000))
            out = pcm[::step]
            return np.asarray(out, dtype=np.float32)
        duration = pcm.size / float(src_rate)
        target_len = max(1, int(round(duration * 16000)))
        # Faster linear resampling without constructing large normalized grids.
        positions = np.arange(target_len, dtype=np.float32) * (src_rate / 16000.0)
        left = np.floor(positions).astype(np.int32)
        right = np.minimum(left + 1, pcm.size - 1)
        frac = positions - left
        out = pcm[left] * (1.0 - frac) + pcm[right] * frac
        return np.asarray(out, dtype=np.float32)

    @staticmethod
    def _preprocess_audio_balanced(audio: np.ndarray) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return pcm
        # Remove DC offset and softly normalize input level for more stable local decode.
        pcm = pcm - float(np.mean(pcm))
        peak = float(np.max(np.abs(pcm)))
        if peak > 1e-6:
            target_peak = 0.88
            gain = max(0.45, min(6.0, target_peak / peak))
            pcm = pcm * gain
        return np.clip(pcm, -0.98, 0.98).astype(np.float32, copy=False)

    @staticmethod
    def _trim_silence(audio: np.ndarray, threshold: float = 0.008, margin_samples: int = 1600) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size < 64:
            return pcm
        active = np.where(np.abs(pcm) > float(threshold))[0]
        if active.size == 0:
            return pcm
        start = max(0, int(active[0]) - int(max(0, margin_samples)))
        end = min(pcm.size, int(active[-1]) + int(max(0, margin_samples)) + 1)
        if end <= start:
            return pcm
        return pcm[start:end]

    @staticmethod
    def _trim_silence_adaptive(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size < int(0.35 * sample_rate):
            return pcm
        win = max(160, int(sample_rate * 0.02))  # 20ms
        hops = max(1, pcm.size // win)
        if hops <= 2:
            return pcm
        head = pcm[: win * hops].reshape(hops, win)
        energy = np.sqrt(np.mean(head * head, axis=1))
        noise_floor = float(np.percentile(energy, 25))
        speech_thr = max(0.0045, min(0.03, noise_floor * 2.25))
        active = np.where(energy > speech_thr)[0]
        if active.size == 0:
            return pcm
        # Keep a small context around first/last speech frame.
        pad = 4
        start_h = max(0, int(active[0]) - pad)
        end_h = min(hops - 1, int(active[-1]) + pad)
        start = start_h * win
        end = min(pcm.size, (end_h + 1) * win)
        if end <= start:
            return pcm
        return pcm[start:end]

    @staticmethod
    def _compress_long_pauses(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        if pcm.size < int(8.0 * sample_rate):
            return pcm
        win = max(160, int(sample_rate * 0.02))  # 20ms
        hops = pcm.size // win
        if hops < 8:
            return pcm
        main = pcm[: hops * win].reshape(hops, win)
        tail = pcm[hops * win :]
        rms = np.sqrt(np.mean(main * main, axis=1))
        noise_floor = float(np.percentile(rms, 30))
        speech_thr = max(0.004, min(0.028, noise_floor * 2.2))
        keep = rms > speech_thr
        if not np.any(keep):
            return pcm
        # Preserve context around speech chunks so words are not clipped.
        pad = 5
        keep_idx = np.where(keep)[0]
        expanded = np.zeros_like(keep, dtype=bool)
        for idx in keep_idx:
            l = max(0, idx - pad)
            r = min(hops, idx + pad + 1)
            expanded[l:r] = True
        compressed = main[expanded].reshape(-1)
        if tail.size:
            compressed = np.concatenate([compressed, tail], axis=0)
        # Keep original if reduction is tiny (avoid pointless transform).
        if compressed.size >= int(pcm.size * 0.9):
            return pcm
        # Avoid over-compression on speech-rich clips.
        if compressed.size < int(pcm.size * 0.45):
            return pcm
        return compressed.astype(np.float32, copy=False)

    def transcribe(self, wav_bytes_or_audio, duration_sec: float, sample_rate: Optional[int] = None) -> tuple[str, dict]:
        self._set_runtime("loading", "Распознаю локально (Sherpa)...")
        total_t0 = time.perf_counter()
        rec = self._ensure_recognizer()
        prep_t0 = time.perf_counter()
        if isinstance(wav_bytes_or_audio, np.ndarray):
            audio = np.asarray(wav_bytes_or_audio, dtype=np.float32).reshape(-1)
            sr = max(8000, int(sample_rate or getattr(self.config, "sample_rate", 24000)))
        else:
            with wave.open(io.BytesIO(bytes(wav_bytes_or_audio or b"")), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
                sr = int(wav.getframerate())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            processed = self._preprocess_audio_balanced(audio)
            audio16k = self._resample_to_16k(processed, sr)
            orig_samples = int(audio16k.size)
            audio16k = self._trim_silence_adaptive(audio16k, sample_rate=16000)
            audio16k = self._trim_silence(audio16k)
            audio16k = self._compress_long_pauses(audio16k, sample_rate=16000)
            audio16k = np.ascontiguousarray(audio16k, dtype=np.float32)
            prep_t1 = time.perf_counter()
            if audio16k.size <= 0:
                self._set_runtime("ready", "Локальная модель Sherpa загружена. Речь не распознана.")
                return "", {"input_tokens": 0, "output_tokens": 0}
            infer_t0 = time.perf_counter()
            stream = rec.create_stream()
            stream.accept_waveform(16000, audio16k)
            rec.decode_stream(stream)
            text = str(getattr(stream.result, "text", "") or "").strip()
            infer_t1 = time.perf_counter()
        except Exception as err:
            self._set_runtime("error", f"Ошибка локального распознавания Sherpa: {err}")
            raise RuntimeError(f"Локальное распознавание Sherpa недоступно: {err}")
        if not text:
            self._set_runtime("ready", "Локальная модель Sherpa загружена. Речь не распознана.")
            return "", {"input_tokens": 0, "output_tokens": 0}
        post_t0 = time.perf_counter()
        self._set_runtime("ready", "Локальная модель Sherpa загружена.")
        out_text = self._postprocess_text(text)
        post_t1 = time.perf_counter()
        total_t1 = time.perf_counter()
        logging.info(
            "Sherpa transcribe timing: prepare_audio_ms=%.1f infer_ms=%.1f postprocess_ms=%.1f total_ms=%.1f samples16k=%s",
            (prep_t1 - prep_t0) * 1000.0,
            (infer_t1 - infer_t0) * 1000.0,
            (post_t1 - post_t0) * 1000.0,
            (total_t1 - total_t0) * 1000.0,
            int(audio16k.size),
        )
        if int(audio16k.size) != int(orig_samples):
            logging.info(
                "Sherpa audio optimization: samples16k %s -> %s (delta=%.1f%%)",
                int(orig_samples),
                int(audio16k.size),
                ((int(orig_samples) - int(audio16k.size)) / max(1.0, float(orig_samples))) * 100.0,
            )
        return out_text, {"input_tokens": 0, "output_tokens": 0}

    def transcribe_live_preview(self, wav_bytes: bytes, duration_sec: float) -> str:
        return ""


def float_audio_to_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(int16.tobytes())
    return output.getvalue()


class LocalMeetingSummaryGGUF:
    MODEL_CATALOG = {
        "qwen2.5-3b-instruct-q4_k_m": {
            "title": "Qwen2.5-3B Instruct Q4_K_M",
            "repo_id": "bartowski/Qwen2.5-3B-Instruct-GGUF",
            "filename": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        },
        "qwen2.5-1.5b-instruct-q4_k_m": {
            "title": "Qwen2.5-1.5B Instruct Q4_K_M",
            "repo_id": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            "filename": "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        },
    }
    DEFAULT_MODEL = "qwen2.5-3b-instruct-q4_k_m"
    _download_lock = threading.Lock()

    @classmethod
    def normalize_model_name(cls, model_name: str) -> str:
        key = str(model_name or "").strip().lower()
        if key in cls.MODEL_CATALOG:
            return key
        return cls.DEFAULT_MODEL

    @classmethod
    def model_title(cls, model_name: str) -> str:
        key = cls.normalize_model_name(model_name)
        return str(cls.MODEL_CATALOG.get(key, {}).get("title", key))

    @classmethod
    def get_download_root(cls) -> Path:
        root = CONFIG_DIR / "models" / "summary-gguf"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @classmethod
    def model_path(cls, model_name: str) -> Path:
        key = cls.normalize_model_name(model_name)
        item = cls.MODEL_CATALOG.get(key, cls.MODEL_CATALOG[cls.DEFAULT_MODEL])
        return cls.get_download_root() / key / str(item.get("filename", "model.gguf"))

    @classmethod
    def is_model_present(cls, model_name: str) -> bool:
        path = cls.model_path(model_name)
        return path.exists() and path.is_file() and path.stat().st_size > int(1 * 1024 * 1024)

    @classmethod
    def model_cached_size_bytes(cls, model_name: str) -> int:
        path = cls.model_path(model_name)
        if not path.exists():
            return 0
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    @classmethod
    def list_installed_models(cls) -> dict[str, bool]:
        out: dict[str, bool] = {}
        for name in cls.MODEL_CATALOG:
            out[name] = cls.is_model_present(name)
        return out

    @classmethod
    def list_installed_sizes(cls) -> dict[str, int]:
        out: dict[str, int] = {}
        for name in cls.MODEL_CATALOG:
            out[name] = cls.model_cached_size_bytes(name)
        return out

    @classmethod
    def estimate_total_size_bytes(cls, model_name: str) -> int:
        key = cls.normalize_model_name(model_name)
        item = cls.MODEL_CATALOG.get(key, cls.MODEL_CATALOG[cls.DEFAULT_MODEL])
        repo_id = str(item.get("repo_id", "")).strip()
        filename = str(item.get("filename", "")).strip()
        if not repo_id or not filename:
            return 0
        if HfApi is None:
            return 0
        try:
            info = HfApi().model_info(repo_id, files_metadata=True)
            for sibling in list(getattr(info, "siblings", []) or []):
                name = str(getattr(sibling, "rfilename", "") or "").strip()
                if name.lower() == filename.lower():
                    return int(getattr(sibling, "size", 0) or 0)
        except Exception:
            logging.exception("Failed to estimate GGUF summary model size")
        return 0

    @classmethod
    def download_model_snapshot(cls, model_name: str, progress_cb=None) -> Path:
        key = cls.normalize_model_name(model_name)
        item = cls.MODEL_CATALOG.get(key, cls.MODEL_CATALOG[cls.DEFAULT_MODEL])
        repo_id = str(item.get("repo_id", "")).strip()
        filename = str(item.get("filename", "")).strip()
        if not repo_id or not filename:
            raise RuntimeError("Конфигурация GGUF-модели некорректна")
        dst = cls.model_path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size > int(1 * 1024 * 1024):
            return dst
        url = f"https://huggingface.co/{repo_id}/resolve/main/{quote(filename, safe='/')}?download=1"
        tmp = dst.with_suffix(dst.suffix + ".part")
        with cls._download_lock:
            with requests.get(url, stream=True, timeout=(5.0, 120.0), headers={"User-Agent": "VoicePROAI/meeting-summary"}) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                if callable(progress_cb):
                    progress_cb("preparing", 0, total, f"Подготовка загрузки {cls.model_title(key)}...")
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 512):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        downloaded += len(chunk)
                        if callable(progress_cb):
                            progress_cb("downloading", downloaded, total, f"Скачивание {cls.model_title(key)}...")
            os.replace(tmp, dst)
        if callable(progress_cb):
            final_size = cls.model_cached_size_bytes(key)
            progress_cb("done", final_size, max(total, final_size), f"Модель {cls.model_title(key)} скачана")
        return dst

    @classmethod
    def delete_model(cls, model_name: str) -> int:
        key = cls.normalize_model_name(model_name)
        root = cls.model_path(key).parent
        removed = 0
        if root.exists():
            for path in sorted(root.rglob("*"), key=lambda p: len(str(p)), reverse=True):
                try:
                    if path.is_file():
                        path.unlink(missing_ok=True)
                        removed += 1
                    elif path.is_dir():
                        path.rmdir()
                except Exception:
                    continue
            try:
                root.rmdir()
            except Exception:
                pass
        return removed


class UiManager:
    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir).resolve() if base_dir else Path(sys.executable).resolve().parent
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Reset startup gate for fresh thread boot.
        self._started = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=2.5):
            logging.error("UI manager start timeout: Tk thread did not signal readiness")

    def show(self, mode: str) -> None:
        self.start()
        self._queue.put(("overlay_show", mode))

    def hide(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("overlay_hide", None))

    def set_live_preview_text(self, text: str) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("overlay_preview_text", str(text or "")))

    def clear_live_preview_text(self) -> None:
        self.set_live_preview_text("")

    def set_live_preview_enabled(self, enabled: bool) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("overlay_preview_enabled", bool(enabled)))

    def set_mic_quality_text(self, text: str) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("overlay_quality_text", str(text or "")))

    def clear_mic_quality_text(self) -> None:
        self.set_mic_quality_text("")

    def set_mic_quality_enabled(self, enabled: bool) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("overlay_quality_enabled", bool(enabled)))

    def open_settings(
        self,
        config_data: dict,
        on_save_cb,
        notify_cb,
        on_open_cb=None,
        on_close_cb=None,
        on_reset_stats_cb=None,
        on_set_history_enabled_cb=None,
        on_delete_history_item_cb=None,
        on_clear_history_cb=None,
        on_set_autoreplace_cb=None,
        on_reset_autoreplace_cb=None,
        on_open_windows_sound_cb=None,
        on_test_microphone_cb=None,
        on_open_logs_cb=None,
        on_start_local_model_download_cb=None,
        on_get_local_model_download_status_cb=None,
        on_delete_local_model_cb=None,
        on_reset_app_settings_cb=None,
        on_add_media_files_cb=None,
        on_get_media_queue_cb=None,
        on_start_media_processing_cb=None,
        on_clear_media_queue_cb=None,
        on_copy_media_result_cb=None,
        on_export_media_result_cb=None,
        on_start_meeting_capture_cb=None,
        on_pause_meeting_capture_cb=None,
        on_stop_meeting_capture_cb=None,
        on_get_meeting_session_status_cb=None,
        on_build_meeting_summary_cb=None,
        on_copy_meeting_text_cb=None,
        on_export_meeting_txt_cb=None,
        on_retry_meeting_loopback_cb=None,
        on_get_system_audio_devices_cb=None,
        on_get_ollama_status_cb=None,
        on_get_ollama_models_cb=None,
        on_download_ollama_model_cb=None,
        on_delete_ollama_model_cb=None,
        on_start_summary_model_download_cb=None,
        on_get_summary_model_download_status_cb=None,
        on_delete_summary_model_cb=None,
        on_check_updates_cb=None,
        on_start_update_install_cb=None,
        on_get_updater_status_cb=None,
    ) -> None:
        self.start()
        if not (self._thread and self._thread.is_alive()):
            logging.error("UI manager thread is not alive, cannot open settings")
            if callable(notify_cb):
                try:
                    notify_cb("Не удалось открыть окно настроек. Перезапустите приложение.")
                except Exception:
                    pass
            return
        self._queue.put(
            (
                "open_settings",
                {
                    "config": config_data,
                    "on_save": on_save_cb,
                    "notify": notify_cb,
                    "on_open": on_open_cb,
                    "on_close": on_close_cb,
                    "on_reset_stats": on_reset_stats_cb,
                    "on_set_history_enabled": on_set_history_enabled_cb,
                    "on_delete_history_item": on_delete_history_item_cb,
                    "on_clear_history": on_clear_history_cb,
                    "on_set_autoreplace": on_set_autoreplace_cb,
                    "on_reset_autoreplace": on_reset_autoreplace_cb,
                    "on_open_windows_sound": on_open_windows_sound_cb,
                    "on_test_microphone": on_test_microphone_cb,
                    "on_open_logs": on_open_logs_cb,
                    "on_start_local_model_download": on_start_local_model_download_cb,
                    "on_get_local_model_download_status": on_get_local_model_download_status_cb,
                    "on_delete_local_model": on_delete_local_model_cb,
                    "on_reset_app_settings": on_reset_app_settings_cb,
                    "on_add_media_files": on_add_media_files_cb,
                    "on_get_media_queue": on_get_media_queue_cb,
                    "on_start_media_processing": on_start_media_processing_cb,
                    "on_clear_media_queue": on_clear_media_queue_cb,
                    "on_copy_media_result": on_copy_media_result_cb,
                    "on_export_media_result": on_export_media_result_cb,
                    "on_start_meeting_capture": on_start_meeting_capture_cb,
                    "on_pause_meeting_capture": on_pause_meeting_capture_cb,
                    "on_stop_meeting_capture": on_stop_meeting_capture_cb,
                    "on_get_meeting_session_status": on_get_meeting_session_status_cb,
                    "on_build_meeting_summary": on_build_meeting_summary_cb,
                    "on_copy_meeting_text": on_copy_meeting_text_cb,
                    "on_export_meeting_txt": on_export_meeting_txt_cb,
                    "on_retry_meeting_loopback": on_retry_meeting_loopback_cb,
                    "on_get_system_audio_devices": on_get_system_audio_devices_cb,
                    "on_get_ollama_status": on_get_ollama_status_cb,
                    "on_get_ollama_models": on_get_ollama_models_cb,
                    "on_download_ollama_model": on_download_ollama_model_cb,
                    "on_delete_ollama_model": on_delete_ollama_model_cb,
                    "on_start_summary_model_download": on_start_summary_model_download_cb,
                    "on_get_summary_model_download_status": on_get_summary_model_download_status_cb,
                    "on_delete_summary_model": on_delete_summary_model_cb,
                    "on_check_updates": on_check_updates_cb,
                    "on_start_update_install": on_start_update_install_cb,
                    "on_get_updater_status": on_get_updater_status_cb,
                },
            )
        )

    def shutdown(self) -> None:
        if self._thread and self._thread.is_alive():
            self._queue.put(("shutdown", None))

    def _run(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox, ttk
        except Exception:
            logging.exception("Tkinter unavailable for UI manager")
            self._started.set()
            return

        root = tk.Tk()
        root.withdraw()
        tk_icon_cache: dict[str, object] = {"photo": None}

        def _apply_tk_window_icon(window) -> None:
            # Keep window icon consistent with tray/exe icon.
            icon_roots = [self.base_dir]
            try:
                exe_dir = Path(sys.executable).resolve().parent
                if exe_dir not in icon_roots:
                    icon_roots.append(exe_dir)
            except Exception:
                pass
            ico_candidates = ("app-icon.ico", "icon-128.ico")
            png_candidates = ("icon-128.png", "icon-64.png", "icon-48.png", "app-icon-1024.png")
            for root_dir in icon_roots:
                for ico_name in ico_candidates:
                    try:
                        icon_ico = root_dir / ico_name
                        if icon_ico.exists():
                            window.iconbitmap(str(icon_ico))
                            raise StopIteration
                    except StopIteration:
                        break
                    except Exception:
                        continue
                else:
                    continue
                break
            for root_dir in icon_roots:
                for png_name in png_candidates:
                    try:
                        icon_png = root_dir / png_name
                        if icon_png.exists():
                            if tk_icon_cache["photo"] is None:
                                tk_icon_cache["photo"] = tk.PhotoImage(file=str(icon_png))
                            window.iconphoto(True, tk_icon_cache["photo"])
                            raise StopIteration
                    except StopIteration:
                        break
                    except Exception:
                        continue
                else:
                    continue
                break

        _apply_tk_window_icon(root)

        overlay_win = {
            "status_win": None,
            "status_label": None,
            "dot": None,
            "status_row": None,
            "dot_phase": 0,
            "dot_job": None,
            "quality_label": None,
            "preview_win": None,
            "preview_label": None,
        }
        overlay_state = {
            "mode": "recording",
            "preview_text": "",
            "preview_enabled": True,
            "quality_text": "",
            "quality_enabled": True,
        }
        settings_win = {"win": None}

        def _draw_rounded_rect(canvas, x1, y1, x2, y2, radius, fill, outline, width=1):
            r = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
            if r == 0:
                canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=width, tags="rounded")
                return
            canvas.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="", tags="rounded")
            canvas.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="", tags="rounded")
            canvas.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, fill=fill, outline="", tags="rounded")
            canvas.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, fill=fill, outline="", tags="rounded")
            canvas.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, fill=fill, outline="", tags="rounded")
            canvas.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, fill=fill, outline="", tags="rounded")
            canvas.create_line(x1 + r, y1 + 0.5, x2 - r, y1 + 0.5, fill=outline, width=width, tags="rounded")
            canvas.create_line(x1 + r, y2 - 0.5, x2 - r, y2 - 0.5, fill=outline, width=width, tags="rounded")
            canvas.create_line(x1 + 0.5, y1 + r, x1 + 0.5, y2 - r, fill=outline, width=width, tags="rounded")
            canvas.create_line(x2 - 0.5, y1 + r, x2 - 0.5, y2 - r, fill=outline, width=width, tags="rounded")
            canvas.create_arc(x1, y1, x1 + 2 * r, y1 + 2 * r, start=90, extent=90, style="arc", outline=outline, width=width, tags="rounded")
            canvas.create_arc(x2 - 2 * r, y1, x2, y1 + 2 * r, start=0, extent=90, style="arc", outline=outline, width=width, tags="rounded")
            canvas.create_arc(x1, y2 - 2 * r, x1 + 2 * r, y2, start=180, extent=90, style="arc", outline=outline, width=width, tags="rounded")
            canvas.create_arc(x2 - 2 * r, y2 - 2 * r, x2, y2, start=270, extent=90, style="arc", outline=outline, width=width, tags="rounded")

        def _build_overlay_container(win, bg="#0f141d", border="#263243", radius=18):
            chroma = "#010203"
            try:
                win.configure(bg=chroma)
                win.wm_attributes("-transparentcolor", chroma)
                rounded = True
            except Exception:
                win.configure(bg=bg)
                rounded = False
            if not rounded:
                panel = tk.Frame(win, bg=bg, bd=1, relief="solid", highlightthickness=0)
                panel.pack(fill="both", expand=True)
                return panel, False

            canvas = tk.Canvas(win, bg=chroma, highlightthickness=0, bd=0)
            canvas.pack(fill="both", expand=True)
            panel = tk.Frame(canvas, bg=bg, bd=0, highlightthickness=0)
            panel_id = canvas.create_window((0, 0), window=panel, anchor="nw")

            def redraw(_event=None):
                panel.update_idletasks()
                w = max(1, panel.winfo_reqwidth())
                h = max(1, panel.winfo_reqheight())
                canvas.configure(width=w, height=h)
                canvas.coords(panel_id, 0, 0)
                canvas.delete("rounded")
                _draw_rounded_rect(canvas, 0, 0, w, h, radius=radius, fill=bg, outline=border, width=1)

            panel.bind("<Configure>", redraw)
            canvas.bind("<Configure>", redraw)
            win.bind("<Map>", redraw)
            return panel, True

        def ensure_overlay() -> None:
            if (
                overlay_win["status_win"] is not None
                and overlay_win["status_win"].winfo_exists()
                and overlay_win["preview_win"] is not None
                and overlay_win["preview_win"].winfo_exists()
            ):
                return
            status_win = tk.Toplevel(root)
            status_win.overrideredirect(True)
            status_win.attributes("-topmost", True)
            try:
                status_win.wm_attributes("-toolwindow", True)
            except Exception:
                pass

            panel, status_is_rounded = _build_overlay_container(status_win, bg="#0f141d", border="#263243", radius=18)
            if not status_is_rounded:
                status_win.attributes("-alpha", 0.98)
            status_row = tk.Frame(panel, bg="#0f141d")
            status_row.pack(fill="x", padx=12, pady=(8, 6))
            dot = tk.Label(status_row, text="\u25cf", font=("Segoe UI", 10, "bold"), bg="#0f141d", fg="#22c55e")
            dot.pack(side="left", padx=(0, 8))
            status_label = tk.Label(status_row, text="СЛУШАЮ...", font=("Segoe UI", 10, "bold"), bg="#0f141d", fg="#f8fafc")
            status_label.pack(side="left")
            quality_label = tk.Label(
                panel,
                text="",
                font=("Segoe UI", 9),
                bg="#0f141d",
                fg="#94a3b8",
                justify="left",
                anchor="w",
                padx=12,
            )

            preview_win = tk.Toplevel(root)
            preview_win.overrideredirect(True)
            preview_win.attributes("-topmost", True)
            try:
                preview_win.wm_attributes("-toolwindow", True)
            except Exception:
                pass
            preview_panel, preview_is_rounded = _build_overlay_container(preview_win, bg="#0f141d", border="#263243", radius=16)
            if not preview_is_rounded:
                preview_win.attributes("-alpha", 0.96)
            preview_label = tk.Label(
                preview_panel,
                text="Слушаю речь...",
                font=("Segoe UI", 10),
                bg="#0f141d",
                fg="#cbd5e1",
                justify="left",
                anchor="w",
                wraplength=520,
                padx=12,
                pady=8,
            )
            preview_label.pack(fill="both", expand=True)

            overlay_win["status_win"] = status_win
            overlay_win["status_label"] = status_label
            overlay_win["dot"] = dot
            overlay_win["status_row"] = status_row
            overlay_win["quality_label"] = quality_label
            overlay_win["preview_win"] = preview_win
            overlay_win["preview_label"] = preview_label
            status_win.withdraw()
            preview_win.withdraw()
            if overlay_win.get("dot_job") is None:
                animate_status_dot()

        def place_overlay() -> None:
            ensure_overlay()
            status_win = overlay_win["status_win"]
            preview_win = overlay_win["preview_win"]
            if status_win is None or preview_win is None:
                return
            status_win.update_idletasks()
            preview_win.update_idletasks()
            sw = status_win.winfo_screenwidth()
            sh = status_win.winfo_screenheight()

            status_w = status_win.winfo_reqwidth()
            status_h = status_win.winfo_reqheight()
            status_x = max(0, (sw - status_w) // 2)
            # Placed lower than center for less intrusive placement during work.
            status_y = max(0, int(sh * 0.69))
            status_win.geometry(f"{status_w}x{status_h}+{status_x}+{status_y}")

            preview_w = min(max(420, preview_win.winfo_reqwidth()), int(sw * 0.72))
            preview_h = preview_win.winfo_reqheight()
            preview_x = max(0, (sw - preview_w) // 2)
            preview_y = status_y + status_h + 8
            if preview_y + preview_h > sh - 16:
                preview_y = max(0, status_y - preview_h - 8)
            preview_win.geometry(f"{preview_w}x{preview_h}+{preview_x}+{preview_y}")

        def animate_status_dot() -> None:
            dot = overlay_win["dot"]
            if dot is None or not dot.winfo_exists():
                overlay_win["dot_job"] = None
                return

            mode = str(overlay_state.get("mode") or "recording")
            palette = {
                "recording": ("#22c55e", "#16a34a", "#15803d"),
                "transcribing": ("#38bdf8", "#0ea5e9", "#0284c7"),
                "mic_off": ("#ef4444", "#dc2626", "#b91c1c"),
            }.get(mode, ("#22c55e", "#16a34a", "#15803d"))

            phase = int(overlay_win.get("dot_phase") or 0)
            dot.configure(fg=palette[phase % len(palette)])
            overlay_win["dot_phase"] = (phase + 1) % len(palette)
            overlay_win["dot_job"] = root.after(190, animate_status_dot)

        def apply_overlay_mode(mode: str) -> None:
            ensure_overlay()
            status_win = overlay_win["status_win"]
            preview_win = overlay_win["preview_win"]
            label = overlay_win["status_label"]
            dot = overlay_win["dot"]
            quality_label = overlay_win["quality_label"]
            preview_label = overlay_win["preview_label"]
            if status_win is None or preview_win is None or label is None or dot is None or preview_label is None or quality_label is None:
                return
            overlay_state["mode"] = str(mode or "recording")
            mode_key = "recording"
            if mode == "transcribing":
                mode_key = "transcribing"
                label.configure(text="РАСПОЗНАЮ...")
                dot.configure(fg="#38bdf8")
            elif mode == "mic_off":
                mode_key = "mic_off"
                label.configure(text="МИКРОФОН ОТКЛЮЧЕН")
                dot.configure(fg="#ef4444")
            else:
                label.configure(text="СЛУШАЮ...")
                dot.configure(fg="#22c55e")
            if overlay_state["mode"] == "recording" and overlay_state["quality_enabled"]:
                quality_label.configure(text=overlay_state["quality_text"] or "Уровень: 0% · Слишком тихо")
            else:
                quality_label.configure(text="")
            if overlay_state["mode"] == "recording" and overlay_state["quality_enabled"]:
                if not quality_label.winfo_manager():
                    quality_label.pack(fill="x", padx=0, pady=(0, 8))
            else:
                if quality_label.winfo_manager():
                    quality_label.pack_forget()
            if overlay_state["mode"] == "recording":
                if overlay_state["preview_enabled"]:
                    preview_label.configure(text=overlay_state["preview_text"] or "Слушаю речь...")
                    preview_win.deiconify()
                else:
                    preview_win.withdraw()
            else:
                preview_win.withdraw()
            place_overlay()
            status_win.deiconify()
            status_win.lift()
            if overlay_state["mode"] == "recording" and overlay_state["preview_enabled"]:
                preview_win.lift()

        def apply_overlay_preview_text(text: str) -> None:
            ensure_overlay()
            preview_label = overlay_win["preview_label"]
            preview_win = overlay_win["preview_win"]
            if preview_label is None or preview_win is None:
                return
            trimmed = str(text or "").strip()
            overlay_state["preview_text"] = trimmed
            preview_label.configure(text=trimmed or "Слушаю речь...")
            if overlay_state["mode"] == "recording" and overlay_state["preview_enabled"]:
                preview_win.deiconify()
            else:
                preview_win.withdraw()
            place_overlay()

        def apply_overlay_preview_enabled(enabled: bool) -> None:
            overlay_state["preview_enabled"] = bool(enabled)
            preview_win = overlay_win["preview_win"]
            if preview_win is not None and preview_win.winfo_exists() and not overlay_state["preview_enabled"]:
                preview_win.withdraw()

        def apply_overlay_quality_text(text: str) -> None:
            ensure_overlay()
            quality_label = overlay_win["quality_label"]
            if quality_label is None:
                return
            trimmed = str(text or "").strip()
            overlay_state["quality_text"] = trimmed
            if overlay_state["mode"] == "recording" and overlay_state["quality_enabled"]:
                quality_label.configure(text=trimmed or "Уровень: 0% · Слишком тихо")
                if not quality_label.winfo_manager():
                    quality_label.pack(fill="x", padx=0, pady=(0, 8))
            else:
                quality_label.configure(text="")
                if quality_label.winfo_manager():
                    quality_label.pack_forget()
            place_overlay()

        def apply_overlay_quality_enabled(enabled: bool) -> None:
            overlay_state["quality_enabled"] = bool(enabled)
            quality_label = overlay_win["quality_label"]
            if quality_label is None:
                return
            if overlay_state["mode"] == "recording" and overlay_state["quality_enabled"]:
                quality_label.configure(text=overlay_state["quality_text"] or "Уровень: 0% · Слишком тихо")
                if not quality_label.winfo_manager():
                    quality_label.pack(fill="x", padx=0, pady=(0, 8))
            else:
                quality_label.configure(text="")
                if quality_label.winfo_manager():
                    quality_label.pack_forget()
            place_overlay()

        def hide_overlay() -> None:
            status_win = overlay_win["status_win"]
            preview_win = overlay_win["preview_win"]
            if status_win is not None and status_win.winfo_exists():
                status_win.withdraw()
            if preview_win is not None and preview_win.winfo_exists():
                preview_win.withdraw()

        def open_settings_window(payload: dict) -> None:
            open_started = time.perf_counter()
            win = settings_win["win"]
            if win is not None and win.winfo_exists():
                win.deiconify()
                win.lift()
                win.focus_force()
                return

            config_data = dict(payload.get("config") or {})
            on_save_cb = payload.get("on_save")
            notify_cb = payload.get("notify")
            on_open_cb = payload.get("on_open")
            on_close_cb = payload.get("on_close")
            on_reset_stats_cb = payload.get("on_reset_stats")
            on_set_history_enabled_cb = payload.get("on_set_history_enabled")
            on_delete_history_item_cb = payload.get("on_delete_history_item")
            on_clear_history_cb = payload.get("on_clear_history")
            on_set_autoreplace_cb = payload.get("on_set_autoreplace")
            on_reset_autoreplace_cb = payload.get("on_reset_autoreplace")
            on_open_windows_sound_cb = payload.get("on_open_windows_sound")
            on_test_microphone_cb = payload.get("on_test_microphone")
            on_open_logs_cb = payload.get("on_open_logs")
            on_start_local_model_download_cb = payload.get("on_start_local_model_download")
            on_get_local_model_download_status_cb = payload.get("on_get_local_model_download_status")
            on_delete_local_model_cb = payload.get("on_delete_local_model")
            on_reset_app_settings_cb = payload.get("on_reset_app_settings")
            on_add_media_files_cb = payload.get("on_add_media_files")
            on_get_media_queue_cb = payload.get("on_get_media_queue")
            on_start_media_processing_cb = payload.get("on_start_media_processing")
            on_clear_media_queue_cb = payload.get("on_clear_media_queue")
            on_copy_media_result_cb = payload.get("on_copy_media_result")
            on_export_media_result_cb = payload.get("on_export_media_result")
            on_start_meeting_capture_cb = payload.get("on_start_meeting_capture")
            on_pause_meeting_capture_cb = payload.get("on_pause_meeting_capture")
            on_stop_meeting_capture_cb = payload.get("on_stop_meeting_capture")
            on_get_meeting_session_status_cb = payload.get("on_get_meeting_session_status")
            on_build_meeting_summary_cb = payload.get("on_build_meeting_summary")
            on_copy_meeting_text_cb = payload.get("on_copy_meeting_text")
            on_export_meeting_txt_cb = payload.get("on_export_meeting_txt")
            on_retry_meeting_loopback_cb = payload.get("on_retry_meeting_loopback")
            on_get_system_audio_devices_cb = payload.get("on_get_system_audio_devices")
            on_get_ollama_status_cb = payload.get("on_get_ollama_status")
            on_get_ollama_models_cb = payload.get("on_get_ollama_models")
            on_download_ollama_model_cb = payload.get("on_download_ollama_model")
            on_delete_ollama_model_cb = payload.get("on_delete_ollama_model")
            on_start_summary_model_download_cb = payload.get("on_start_summary_model_download")
            on_get_summary_model_download_status_cb = payload.get("on_get_summary_model_download_status")
            on_delete_summary_model_cb = payload.get("on_delete_summary_model")
            on_check_updates_cb = payload.get("on_check_updates")
            on_start_update_install_cb = payload.get("on_start_update_install")
            on_get_updater_status_cb = payload.get("on_get_updater_status")

            win = tk.Toplevel(root)
            win.title(APP_NAME + " — Настройки")
            win.geometry("1320x860")
            win.minsize(1120, 760)
            win.configure(bg="#f3f6fb")
            _apply_tk_window_icon(win)
            settings_win["win"] = win
            ui_motion_state = {"dragging": False, "release_job": None}

            def _is_dragging_window() -> bool:
                return bool(ui_motion_state["dragging"])

            def _set_dragging_window(value: bool) -> None:
                ui_motion_state["dragging"] = bool(value)

            def _on_window_configure(_e=None) -> None:
                # While the window is being moved/resized, reduce visual churn from
                # hover-driven repainting. This improves perceived smoothness.
                _set_dragging_window(True)
                job = ui_motion_state.get("release_job")
                if job:
                    try:
                        win.after_cancel(job)
                    except Exception:
                        pass
                ui_motion_state["release_job"] = win.after(140, lambda: _set_dragging_window(False))

            win.bind("<Configure>", _on_window_configure, add="+")

            def _try_enable_dark_titlebar(window: tk.Toplevel) -> None:
                # Safe Windows-only hint for native dark title bar.
                # If API/OS does not support it, silently do nothing.
                if os.name != "nt":
                    return
                try:
                    hwnd = ctypes.c_void_p(int(window.winfo_id()))
                    dwmapi = ctypes.windll.dwmapi  # type: ignore[attr-defined]
                    value = ctypes.c_int(1)
                    value_size = ctypes.sizeof(value)
                    # Win11 uses 20, older Win10 builds may use 19.
                    for attr in (20, 19):
                        try:
                            result = dwmapi.DwmSetWindowAttribute(hwnd, ctypes.c_uint(attr), ctypes.byref(value), value_size)
                            if result == 0:
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # Premium desktop tokens
            c_bg = "#f3f6fb"
            c_surface = "#ffffff"
            c_surface_alt = "#f8fbff"
            c_border = "#dbe4ef"
            c_text = "#0f172a"
            c_text_muted = "#64748b"
            c_accent = "#14b8a6"
            c_accent_soft = "#e7fbf8"
            c_sidebar = "#0b1324"
            c_sidebar_panel = "#0f1b31"
            c_sidebar_hover = "#13233f"
            c_sidebar_active = "#173154"
            c_sidebar_border = "#1f2f4a"
            c_sidebar_text = "#cdd8ec"
            c_sidebar_muted_text = "#8ea3c2"
            c_sidebar_active_text = "#7bf2df"
            c_nav_icon_bg = "#132540"
            c_nav_icon_border = "#223b5f"
            c_nav_icon_active_bg = "#0f3f49"
            c_nav_icon_active_border = "#2ab8a8"
            c_danger = "#be123c"

            style = ttk.Style(win)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure(".", background=c_surface, foreground=c_text, fieldbackground=c_surface)
            style.configure("TFrame", background=c_bg)
            style.configure("TLabel", background=c_bg, foreground=c_text)
            style.configure("TLabelframe", background=c_surface, foreground=c_text, bordercolor=c_border, relief="solid")
            style.configure("TLabelframe.Label", background=c_surface, foreground=c_text)
            style.configure("TButton", background="#e8eefc", foreground=c_text, borderwidth=0, padding=(10, 7))
            style.map(
                "TButton",
                background=[("active", "#dbe7ff"), ("pressed", "#cfdefc"), ("disabled", "#eef2f7")],
                foreground=[("disabled", "#94a3b8"), ("!disabled", c_text)],
            )
            style.configure("Primary.TButton", background=c_accent, foreground="#062a2a", padding=(12, 8))
            style.map(
                "Primary.TButton",
                background=[("active", "#0fb6a6"), ("pressed", "#0aa094"), ("disabled", "#99e6df")],
                foreground=[("disabled", "#3d5d5c"), ("!disabled", "#062a2a")],
            )
            style.configure("Danger.TButton", background="#fee2e2", foreground=c_danger, padding=(10, 8))
            style.map(
                "Danger.TButton",
                background=[("active", "#fecaca"), ("pressed", "#fda4af"), ("disabled", "#fef2f2")],
                foreground=[("disabled", "#9f1239"), ("!disabled", c_danger)],
            )
            style.configure("GhostDanger.TButton", background="#fff1f2", foreground=c_danger, padding=(10, 8), borderwidth=0)
            style.map(
                "GhostDanger.TButton",
                background=[("active", "#ffe4e6"), ("pressed", "#fecdd3"), ("disabled", "#f8fafc")],
                foreground=[("disabled", "#94a3b8"), ("!disabled", c_danger)],
            )
            style.configure("TEntry", fieldbackground=c_surface, foreground=c_text, insertcolor=c_text, bordercolor=c_border)
            style.configure(
                "Settings.TEntry",
                fieldbackground="#f8fbff",
                foreground=c_text,
                insertcolor=c_text,
                bordercolor=c_border,
                lightcolor=c_border,
                darkcolor=c_border,
                padding=(8, 6),
            )
            style.configure(
                "TCombobox",
                fieldbackground=c_surface,
                background=c_surface,
                foreground=c_text,
                arrowcolor="#334155",
                selectbackground="#dbe7ff",
                selectforeground=c_text,
            )
            style.map(
                "TCombobox",
                fieldbackground=[("readonly", "#ffffff"), ("!readonly", "#ffffff")],
                foreground=[("readonly", "#0f172a"), ("!readonly", "#0f172a")],
                selectbackground=[("readonly", "#dbe7ff"), ("!readonly", "#dbe7ff")],
                selectforeground=[("readonly", "#0f172a"), ("!readonly", "#0f172a")],
                arrowcolor=[("readonly", "#334155"), ("!readonly", "#334155")],
            )
            # Force dark colors for readonly comboboxes on Windows themes.
            style.configure(
                "Dark.TCombobox",
                fieldbackground=c_surface,
                background=c_surface,
                foreground=c_text,
                arrowcolor="#334155",
            )
            style.map(
                "Dark.TCombobox",
                fieldbackground=[("readonly", "#ffffff"), ("disabled", "#f1f5f9"), ("!disabled", "#ffffff")],
                background=[("readonly", "#ffffff"), ("disabled", "#f1f5f9"), ("!disabled", "#ffffff")],
                foreground=[("readonly", "#0f172a"), ("disabled", "#94a3b8"), ("!disabled", "#0f172a")],
                selectbackground=[("readonly", "#dbe7ff"), ("!readonly", "#dbe7ff")],
                selectforeground=[("readonly", "#0f172a"), ("!readonly", "#0f172a")],
                arrowcolor=[("readonly", "#334155"), ("disabled", "#94a3b8"), ("!disabled", "#334155")],
            )
            style.configure(
                "Settings.TButton",
                background="#eaf2ff",
                foreground=c_text,
                borderwidth=0,
                padding=(11, 8),
            )
            style.map(
                "Settings.TButton",
                background=[("active", "#dbe7ff"), ("pressed", "#cfdefc"), ("disabled", "#eef2f7")],
                foreground=[("disabled", "#94a3b8"), ("!disabled", c_text)],
            )
            style.configure(
                "Card.TCheckbutton",
                background=c_surface,
                foreground=c_text,
                indicatorcolor="#ffffff",
                indicatormargin=(0, 0, 8, 0),
                padding=(0, 2),
            )
            style.map(
                "Card.TCheckbutton",
                background=[("active", c_surface), ("selected", c_surface), ("!disabled", c_surface)],
                foreground=[("disabled", "#94a3b8"), ("!disabled", c_text)],
                indicatorcolor=[("selected", c_accent), ("active", "#ffffff"), ("!disabled", "#ffffff")],
            )
            style.configure(
                "Accent.Horizontal.TProgressbar",
                troughcolor="#e2e8f0",
                background=c_accent,
                bordercolor=c_border,
                lightcolor=c_accent,
                darkcolor=c_accent,
            )
            # Dropdown list colors for ttk.Combobox popup.
            win.option_add("*TCombobox*Listbox.background", c_surface)
            win.option_add("*TCombobox*Listbox.foreground", c_text)
            win.option_add("*TCombobox*Listbox.selectBackground", "#dbe7ff")
            win.option_add("*TCombobox*Listbox.selectForeground", c_text)

            # Apply after native window is materialized.
            try:
                win.update_idletasks()
                _try_enable_dark_titlebar(win)
            except Exception:
                pass

            win.columnconfigure(1, weight=1)
            win.rowconfigure(0, weight=1)

            sidebar = tk.Frame(win, bg=c_sidebar, width=292)
            sidebar.grid(row=0, column=0, sticky="nsw")
            sidebar.grid_propagate(False)
            sidebar.columnconfigure(0, weight=1)

            content = ttk.Frame(win, padding=18)
            content.grid(row=0, column=1, sticky="nsew")
            content.columnconfigure(0, weight=1)
            content.rowconfigure(1, weight=1)

            brand = tk.Frame(sidebar, bg=c_sidebar_panel, highlightthickness=1, highlightbackground=c_sidebar_border)
            brand.grid(row=0, column=0, sticky="ew", padx=14, pady=(16, 12))
            brand.columnconfigure(1, weight=1)
            _brand_icon = None
            try:
                for icon_name in ("icon-32.png", "icon-48.png", "icon-64.png", "icon-128.png"):
                    brand_icon_path = self.base_dir / icon_name
                    if brand_icon_path.exists():
                        _brand_icon = tk.PhotoImage(file=str(brand_icon_path))
                        break
            except Exception:
                _brand_icon = None
            if _brand_icon is not None:
                brand_icon_lbl = tk.Label(brand, image=_brand_icon, bg=c_sidebar_panel, borderwidth=0, highlightthickness=0)
                brand_icon_lbl.image = _brand_icon
                brand_icon_lbl.grid(row=0, column=0, sticky="w", padx=(10, 8), pady=(10, 0))
            else:
                tk.Label(brand, text="\u25CF", fg=c_accent, bg=c_sidebar_panel, font=("Segoe UI", 13, "bold")).grid(
                    row=0, column=0, sticky="w", padx=(10, 8), pady=(10, 0)
                )
            tk.Label(brand, text=APP_NAME, fg="#f8fafc", bg=c_sidebar_panel, font=("Segoe UI", 17, "bold")).grid(
                row=0, column=1, sticky="w", pady=(10, 0)
            )
            tk.Label(
                brand,
                text="Премиальная AI-диктовка для Windows",
                fg=c_sidebar_muted_text,
                bg=c_sidebar_panel,
                font=("Segoe UI", 9),
            ).grid(row=1, column=1, sticky="w", pady=(2, 10))

            section_title_var = tk.StringVar(value="Настройки")
            section_desc_var = tk.StringVar(value="Управляйте всеми параметрами Voice PRO AI в одном месте.")
            top = ttk.Frame(content)
            top.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            top.columnconfigure(0, weight=1)
            ttk.Label(top, textvariable=section_title_var, font=("Segoe UI", 24, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Label(top, textvariable=section_desc_var, foreground=c_text_muted, font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=(2, 0))

            status_var = tk.StringVar(value="")
            status_lbl = ttk.Label(top, textvariable=status_var, foreground="#2b7a0b", font=("Segoe UI", 10, "bold"))
            status_lbl.grid(row=2, column=0, sticky="w", pady=(8, 0))

            page_host = ttk.Frame(content)
            page_host.grid(row=1, column=0, sticky="nsew")
            page_host.columnconfigure(0, weight=1)
            page_host.rowconfigure(0, weight=1)

            overview_page = ttk.Frame(page_host, padding=8)
            history_page = ttk.Frame(page_host, padding=8)
            files_page = ttk.Frame(page_host, padding=8)
            meeting_page = ttk.Frame(page_host, padding=8)
            rules_page = ttk.Frame(page_host, padding=8)
            settings_page = ttk.Frame(page_host, padding=8)
            help_page = ttk.Frame(page_host, padding=8)

            pages = {
                "Главная и статистика": overview_page,
                "История": history_page,
                "Транскрибация аудио и видео": files_page,
                "Voice Meet AI": meeting_page,
                "Автозамена": rules_page,
                "Настройки": settings_page,
                "Как пользоваться": help_page,
            }

            for frm in pages.values():
                frm.grid(row=0, column=0, sticky="nsew")

            def add_placeholder(page: ttk.Frame, title: str, text: str) -> None:
                card = tk.Frame(page, bg=c_surface, highlightthickness=1, highlightbackground=c_border)
                card.pack(fill="x", pady=(0, 12))
                tk.Label(card, text=title, bg=c_surface, fg=c_text, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
                tk.Label(card, text=text, justify="left", wraplength=920, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(0, 12))

            stats_data = dict(config_data.get("_stats") or {})

            def _fmt_num(v: int) -> str:
                try:
                    return f"{int(v):,}".replace(",", " ")
                except Exception:
                    return "0"

            stats_today_var = tk.StringVar(value=_fmt_num(int(stats_data.get("today_words", 0))))
            stats_total_var = tk.StringVar(value=_fmt_num(int(stats_data.get("total_words", 0))))
            stats_speed_var = tk.StringVar(value=str(int(stats_data.get("speed_wpm", 0))))
            stats_in_tokens_var = tk.StringVar(value=_fmt_num(int(stats_data.get("total_input_tokens", 0))))
            stats_out_tokens_var = tk.StringVar(value=_fmt_num(int(stats_data.get("total_output_tokens", 0))))
            motivation_var = tk.StringVar(value="")
            runtime_mode_var = tk.StringVar(value="")
            runtime_model_state_var = tk.StringVar(value="")
            runtime_model_message_var = tk.StringVar(value="")

            overview_page.columnconfigure(0, weight=1)
            overview_page.configure(style="TFrame")

            def _create_overview_kpi(parent, title: str, value_var: tk.StringVar, caption: str, big: bool = True):
                card = tk.Frame(parent, bg=c_surface, highlightthickness=1, highlightbackground=c_border)
                tk.Label(card, text=title, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
                tk.Label(
                    card,
                    textvariable=value_var,
                    bg=c_surface,
                    fg=c_text,
                    font=("Segoe UI", 30 if big else 22, "bold"),
                ).pack(anchor="w", padx=14, pady=(0, 2))
                tk.Label(card, text=caption, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=(0, 12))
                return card

            metrics = tk.Frame(overview_page, bg=c_bg)
            metrics.pack(fill="x", pady=(0, 12))
            for i in range(3):
                metrics.grid_columnconfigure(i, weight=1)

            _create_overview_kpi(metrics, "Слов сегодня", stats_today_var, "слов").grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            _create_overview_kpi(metrics, "Всего надиктовано", stats_total_var, "слов").grid(row=0, column=1, sticky="nsew", padx=6)
            _create_overview_kpi(metrics, "Скорость диктовки", stats_speed_var, "слов в минуту").grid(row=0, column=2, sticky="nsew", padx=(6, 0))

            tokens_metrics = tk.Frame(overview_page, bg=c_bg)
            tokens_metrics.pack(fill="x", pady=(0, 12))
            tokens_metrics.grid_columnconfigure(0, weight=1)
            tokens_metrics.grid_columnconfigure(1, weight=1)

            _create_overview_kpi(tokens_metrics, "Входящие токены", stats_in_tokens_var, "всего отправлено в модель", big=False).grid(
                row=0, column=0, sticky="nsew", padx=(0, 6)
            )
            _create_overview_kpi(tokens_metrics, "Исходящие токены", stats_out_tokens_var, "всего получено от модели", big=False).grid(
                row=0, column=1, sticky="nsew", padx=(6, 0)
            )

            runtime_card = tk.Frame(overview_page, bg=c_surface, highlightthickness=1, highlightbackground=c_border)
            runtime_card.pack(fill="x", pady=(0, 12))
            runtime_card.columnconfigure(0, weight=1)
            tk.Label(runtime_card, text="Статус распознавания", bg=c_surface, fg=c_text, font=("Segoe UI", 12, "bold")).grid(
                row=0, column=0, sticky="w", padx=14, pady=(12, 2)
            )
            tk.Label(runtime_card, textvariable=runtime_mode_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 10)).grid(
                row=1, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(runtime_card, textvariable=runtime_model_state_var, bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=2, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(
                runtime_card,
                textvariable=runtime_model_message_var,
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=980,
            ).grid(row=3, column=0, sticky="w", padx=14, pady=(0, 12))

            motivation_card = tk.Frame(overview_page, bg=c_accent_soft, highlightthickness=1, highlightbackground="#b7efe7")
            motivation_card.pack(fill="x", pady=(0, 12))
            tk.Label(motivation_card, text="Вы работаете быстрее", bg=c_accent_soft, fg="#0f766e", font=("Segoe UI", 13, "bold")).pack(
                anchor="w", padx=14, pady=(12, 4)
            )
            tk.Label(
                motivation_card,
                textvariable=motivation_var,
                justify="left",
                wraplength=980,
                bg=c_accent_soft,
                fg="#115e59",
                font=("Segoe UI", 11),
            ).pack(anchor="w", padx=14, pady=(0, 12))

            hotkey_preview_var = tk.StringVar(value=str(config_data.get("ptt_key", "f8")).upper())
            howto_step1_var = tk.StringVar()
            howto_card = tk.Frame(overview_page, bg=c_surface, highlightthickness=1, highlightbackground=c_border)
            howto_card.pack(fill="x", pady=(0, 12))
            tk.Label(howto_card, text=f"Как использовать {APP_NAME}", bg=c_surface, fg=c_text, font=("Segoe UI", 14, "bold")).pack(
                anchor="w", padx=14, pady=(12, 8)
            )
            tk.Label(howto_card, textvariable=howto_step1_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 10), justify="left").pack(
                anchor="w", padx=14, pady=(0, 4)
            )
            tk.Label(
                howto_card,
                text="2. Отпустите клавишу — приложение распознает речь и вставит текст в активное поле.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 10),
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 4))
            tk.Label(
                howto_card,
                text="3. Для лучшего результата говорите фразами по 4–12 секунд без длинных пауз.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 10),
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 4))
            tk.Label(
                howto_card,
                text="4. Дополнительные возможности доступны в разделах «История» и «Автозамена».",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 10),
                justify="left",
            ).pack(anchor="w", padx=14, pady=(0, 12))

            def _apply_stats_to_overview(snapshot: dict) -> None:
                howto_step1_var.set(f"1. Нажмите и удерживайте {hotkey_preview_var.get()}, затем диктуйте текст.")
                total_words = int(snapshot.get("total_words", 0) or 0)
                today_words = int(snapshot.get("today_words", 0) or 0)
                speed_wpm = int(snapshot.get("speed_wpm", 0) or 0)
                total_input_tokens = int(snapshot.get("total_input_tokens", 0) or 0)
                total_output_tokens = int(snapshot.get("total_output_tokens", 0) or 0)
                stats_today_var.set(_fmt_num(today_words))
                stats_total_var.set(_fmt_num(total_words))
                stats_speed_var.set(str(speed_wpm))
                stats_in_tokens_var.set(_fmt_num(total_input_tokens))
                stats_out_tokens_var.set(_fmt_num(total_output_tokens))
                ratio = (speed_wpm / 40.0) if speed_wpm > 0 else 0.0
                if ratio >= 1.1:
                    motivation_var.set(
                        f"Ваша скорость диктовки примерно в {ratio:.1f} раза выше условной скорости ручного набора. "
                        "Это экономит время каждый день."
                    )
                else:
                    motivation_var.set(
                        "Статистика набирается. Чем больше диктовок, тем точнее оценка скорости и прогресса."
                    )

            def _apply_transcriber_status_to_ui(snapshot: dict) -> None:
                data = dict(snapshot.get("_transcriber_status") or {})
                backend = str(data.get("backend", "api")).strip().lower()
                state = str(data.get("state", "init")).strip().lower()
                model_label = str(data.get("model_label", "")).strip()
                message = str(data.get("message", "")).strip()
                if backend == "local":
                    low_label = model_label.lower()
                    if "vosk" in low_label:
                        mode_text = "Локально (Vosk)"
                    elif "sherpa" in low_label or "gigaam" in low_label:
                        mode_text = "Локально (Sherpa)"
                    elif "whisper" in low_label:
                        mode_text = "Локально (Whisper)"
                    else:
                        mode_text = "Локально (на устройстве)"
                else:
                    mode_text = "Через API (OpenRouter)"
                if state == "ready":
                    state_text = "Модель: Загружена"
                elif state == "loading":
                    state_text = "Модель: Загружается"
                elif state == "error":
                    state_text = "Модель: Ошибка"
                elif backend == "local":
                    state_text = "Модель: Не загружена"
                else:
                    state_text = "Модель: Инициализация"
                runtime_mode_var.set(f"Режим: {mode_text}")
                if model_label:
                    runtime_model_state_var.set(f"{state_text} В· {model_label}")
                else:
                    runtime_model_state_var.set(state_text)
                runtime_model_message_var.set(message or " ")

            _apply_stats_to_overview(stats_data)
            _apply_transcriber_status_to_ui(config_data)

            reset_stats_btn = ttk.Button(top, text="Сбросить статистику")
            reset_stats_btn.grid(row=0, column=1, sticky="e")

            def _on_reset_stats() -> None:
                if not callable(on_reset_stats_cb):
                    status_var.set("Обработчик сброса недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message, snapshot = on_reset_stats_cb()
                except Exception as err:
                    ok, message, snapshot = False, f"Ошибка: {err}", {}
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    _apply_stats_to_overview(dict(snapshot or {}))

            reset_stats_btn.configure(command=_on_reset_stats)

            history_data = dict(config_data.get("_history") or {})
            history_enabled_var = tk.BooleanVar(value=bool(history_data.get("enabled", True)))
            history_rendered = {"value": False}

            history_page.columnconfigure(0, weight=1)
            card_bg = c_surface
            card_border = c_border
            text_primary = c_text
            text_secondary = c_text_muted

            history_toggle_card = tk.Frame(history_page, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            history_toggle_card.pack(fill="x", pady=(0, 10))
            history_toggle_card.grid_columnconfigure(0, weight=1)
            tk.Label(
                history_toggle_card,
                text="Активность модуля",
                bg=card_bg,
                fg=text_primary,
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
            tk.Label(
                history_toggle_card,
                text="Если переключатель включён, новые записи будут сохраняться в историю.",
                wraplength=760,
                justify="left",
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 11),
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))

            toggle_canvas = tk.Canvas(history_toggle_card, width=52, height=30, bg=card_bg, highlightthickness=0, bd=0)
            toggle_canvas.grid(row=0, column=1, rowspan=2, sticky="e", padx=14)

            toggle_state = {"value": bool(history_enabled_var.get())}

            def _draw_toggle() -> None:
                toggle_canvas.delete("all")
                on = bool(toggle_state["value"])
                track = c_accent if on else "#3a4254"
                knob_x = 36 if on else 16
                toggle_canvas.create_oval(8, 7, 22, 23, fill=track, outline=track)
                toggle_canvas.create_rectangle(15, 7, 37, 23, fill=track, outline=track)
                toggle_canvas.create_oval(30, 7, 44, 23, fill=track, outline=track)
                toggle_canvas.create_oval(knob_x - 7, 8, knob_x + 7, 22, fill="#f8fafc", outline="#d1d9e6")

            _draw_toggle()

            history_row_bg = c_surface
            history_row_hover = "#f6fbff"
            history_row_selected = "#eef8ff"
            history_row_border = c_border
            history_row_border_hover = "#cbddee"
            history_row_border_selected = "#9ec5e9"
            history_action_bg = "#edf3ff"
            history_action_hover = "#dbeafe"
            history_delete_bg = "#fff1f2"
            history_delete_hover = "#ffe4e6"

            history_list_wrap = tk.Frame(history_page, bg=c_bg)
            history_list_wrap.pack(fill="both", expand=True)
            history_list_wrap.columnconfigure(0, weight=1)
            history_list_wrap.rowconfigure(0, weight=1)

            history_canvas = tk.Canvas(history_list_wrap, highlightthickness=0, bd=0, bg="#f5f7fb")
            history_scroll = ttk.Scrollbar(history_list_wrap, orient="vertical", command=history_canvas.yview)
            history_canvas.configure(yscrollcommand=history_scroll.set)
            history_canvas.grid(row=0, column=0, sticky="nsew")
            history_scroll.grid(row=0, column=1, sticky="ns")

            history_items_frame = tk.Frame(history_canvas, bg=c_bg)
            history_window = history_canvas.create_window((0, 0), window=history_items_frame, anchor="nw")

            _history_layout_state = {"canvas_width": -1}

            def _on_history_items_config(_e=None):
                history_canvas.configure(scrollregion=history_canvas.bbox("all"))

            def _on_history_canvas_config(_e=None):
                current_width = int(history_canvas.winfo_width())
                if current_width <= 0:
                    return
                if _history_layout_state["canvas_width"] == current_width:
                    return
                _history_layout_state["canvas_width"] = current_width
                history_canvas.itemconfigure(history_window, width=current_width)

            history_items_frame.bind("<Configure>", _on_history_items_config)
            history_canvas.bind("<Configure>", _on_history_canvas_config)

            clear_history_btn = ttk.Button(top, text="Удалить все записи", style="GhostDanger.TButton")
            clear_history_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))
            reset_rules_btn = ttk.Button(top, text="Сбросить", style="Settings.TButton")
            reset_rules_btn.grid(row=0, column=3, sticky="e", padx=(6, 0))
            add_rule_btn = ttk.Button(top, text="Добавить правило", style="Primary.TButton")
            add_rule_btn.grid(row=0, column=4, sticky="e", padx=(6, 0))
            reset_settings_btn = ttk.Button(top, text="Сбросить")
            reset_settings_btn.grid(row=0, column=5, sticky="e", padx=(6, 0))

            def _copy_history_text(txt: str) -> None:
                try:
                    pyperclip.copy(txt)
                    status_var.set("Запись скопирована")
                    status_lbl.configure(foreground="#2b7a0b")
                except Exception as err:
                    status_var.set(f"Ошибка: {err}")
                    status_lbl.configure(foreground="#9f1239")

            history_selected_id = {"value": ""}

            def _set_clear_history_button_state(has_items: bool) -> None:
                try:
                    if has_items:
                        clear_history_btn.state(["!disabled"])
                    else:
                        clear_history_btn.state(["disabled"])
                except Exception:
                    pass

            def _render_history(snapshot: dict) -> None:
                for child in history_items_frame.winfo_children():
                    child.destroy()
                items = list(snapshot.get("items") or [])
                _set_clear_history_button_state(bool(items))
                if not items:
                    empty = tk.Frame(history_items_frame, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
                    empty.pack(fill="x", pady=(0, 8))
                    tk.Label(
                        empty,
                        text="История пуста",
                        bg=card_bg,
                        fg=text_primary,
                        font=("Segoe UI", 11, "bold"),
                    ).pack(anchor="w", padx=12, pady=(10, 2))
                    tk.Label(
                        empty,
                        text="Надиктуйте текст и отпустите горячую клавишу — запись появится здесь.",
                        wraplength=740,
                        justify="left",
                        bg=card_bg,
                        fg=text_secondary,
                        font=("Segoe UI", 10),
                    ).pack(anchor="w", padx=12, pady=(0, 10))
                    history_selected_id["value"] = ""
                    return

                for item in items:
                    item_id = str(item.get("id", "")).strip()
                    item_ts = str(item.get("ts", "")).strip()
                    item_text = str(item.get("text", "")).strip()
                    if not item_id or not item_text:
                        continue
                    card = tk.Frame(
                        history_items_frame,
                        bg=history_row_bg,
                        highlightthickness=1,
                        highlightbackground=history_row_border,
                        highlightcolor=history_row_border_selected,
                        padx=14,
                        pady=12,
                        takefocus=1,
                    )
                    card.pack(fill="x", pady=(0, 8))
                    card.columnconfigure(0, weight=1)
                    meta_row = tk.Frame(card, bg=history_row_bg)
                    meta_row.grid(row=0, column=0, sticky="ew")
                    meta_row.columnconfigure(0, weight=1)
                    ts_lbl = tk.Label(
                        meta_row,
                        text=item_ts or "Запись",
                        bg=history_row_bg,
                        fg=text_secondary,
                        font=("Segoe UI", 10),
                    )
                    ts_lbl.grid(row=0, column=0, sticky="w", pady=(0, 8))
                    actions = tk.Frame(meta_row, bg=history_row_bg)
                    actions.grid(row=0, column=1, sticky="e")

                    copy_btn = tk.Button(
                        actions,
                        text="Копировать",
                        command=lambda t=item_text: _copy_history_text(t),
                        bg=history_action_bg,
                        fg=c_text,
                        activebackground=history_action_hover,
                        activeforeground=c_text,
                        relief="flat",
                        bd=0,
                        padx=12,
                        pady=7,
                        cursor="hand2",
                        font=("Segoe UI", 9, "bold"),
                    )
                    copy_btn.grid(row=0, column=0, padx=(0, 8))

                    def _delete_one(iid=item_id):
                        if not callable(on_delete_history_item_cb):
                            status_var.set("Обработчик удаления недоступен")
                            status_lbl.configure(foreground="#9f1239")
                            return
                        if not messagebox.askyesno("Удаление записи", "Удалить выбранную запись из истории?", parent=win):
                            return
                        try:
                            ok, message, snap = on_delete_history_item_cb(iid)
                        except Exception as err:
                            ok, message, snap = False, f"Ошибка: {err}", snapshot
                        status_var.set(message)
                        status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                        if ok:
                            _render_history(dict(snap or {}))

                    delete_btn = tk.Button(
                        actions,
                        text="Удалить",
                        command=_delete_one,
                        bg=history_delete_bg,
                        fg=c_danger,
                        activebackground=history_delete_hover,
                        activeforeground=c_danger,
                        relief="flat",
                        bd=0,
                        padx=12,
                        pady=7,
                        cursor="hand2",
                        font=("Segoe UI", 9, "bold"),
                    )
                    delete_btn.grid(row=0, column=1)

                    text_lbl = tk.Label(
                        card,
                        text=item_text,
                        wraplength=620,
                        justify="left",
                        bg=history_row_bg,
                        fg=text_primary,
                        font=("Segoe UI", 12),
                    )
                    text_lbl.grid(row=1, column=0, sticky="ew", pady=(0, 2))

                    def _set_row_look(is_hover: bool = False, is_selected: bool = False) -> None:
                        if is_selected:
                            bg = history_row_selected
                            border = history_row_border_selected
                        elif is_hover:
                            bg = history_row_hover
                            border = history_row_border_hover
                        else:
                            bg = history_row_bg
                            border = history_row_border
                        card.configure(bg=bg, highlightbackground=border)
                        meta_row.configure(bg=bg)
                        actions.configure(bg=bg)
                        ts_lbl.configure(bg=bg)
                        text_lbl.configure(bg=bg)

                    _row_wrap_state = {"width": -1}

                    def _fit_history_wrap(_e=None, holder=card, lbl=text_lbl, btns=actions):
                        try:
                            holder_w = max(360, holder.winfo_width())
                            wrap = max(280, holder_w - 48)
                            if _row_wrap_state["width"] == wrap:
                                return
                            _row_wrap_state["width"] = wrap
                            lbl.configure(wraplength=wrap)
                        except Exception:
                            pass

                    def _select_row(_e=None):
                        history_selected_id["value"] = item_id
                        _render_history(snapshot)

                    def _hover_row(_e=None):
                        if _is_dragging_window():
                            return
                        _set_row_look(is_hover=True, is_selected=(history_selected_id["value"] == item_id))

                    def _unhover_row(_e=None):
                        if _is_dragging_window():
                            return
                        _set_row_look(is_hover=False, is_selected=(history_selected_id["value"] == item_id))

                    for w in (card, meta_row, ts_lbl, text_lbl, actions):
                        w.bind("<Button-1>", _select_row)
                        w.bind("<Enter>", _hover_row)
                        w.bind("<Leave>", _unhover_row)

                    card.bind("<Configure>", _fit_history_wrap)
                    card.after(0, _fit_history_wrap)
                    _set_row_look(False, history_selected_id["value"] == item_id)
                history_rendered["value"] = True

            def _on_toggle_history() -> None:
                enabled = bool(toggle_state["value"])
                if not callable(on_set_history_enabled_cb):
                    status_var.set("Обработчик переключателя истории недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message, snap = on_set_history_enabled_cb(enabled)
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", history_data
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    history_enabled_var.set(enabled)
                    _render_history(dict(snap or {}))
                else:
                    toggle_state["value"] = not enabled
                    _draw_toggle()

            def _toggle_click(_e=None):
                toggle_state["value"] = not bool(toggle_state["value"])
                _draw_toggle()
                _on_toggle_history()

            toggle_canvas.bind("<Button-1>", _toggle_click)

            def _on_clear_history() -> None:
                if not callable(on_clear_history_cb):
                    status_var.set("Обработчик очистки истории недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                if not messagebox.askyesno(
                    "Очистка истории",
                    "Удалить все записи из истории? Это действие нельзя отменить.",
                    parent=win,
                ):
                    return
                try:
                    ok, message, snap = on_clear_history_cb()
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", history_data
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    _render_history(dict(snap or {}))

            clear_history_btn.configure(command=_on_clear_history)

            def _ensure_history_rendered() -> None:
                if history_rendered["value"]:
                    return
                _render_history(history_data)
            files_page.columnconfigure(0, weight=1)

            media_data = dict(config_data.get("_media_queue") or {})
            media_selected_id = {"value": ""}
            media_poll_job = {"id": None}
            media_rendered = {"value": False}
            media_snapshot = {"value": {"items": [], "active_id": "", "is_processing": False}}
            media_result_state = {"last_item_id": "", "last_signature": "", "last_yview": (0.0, 1.0)}
            media_status_var = tk.StringVar(value="")
            media_result_title_var = tk.StringVar(value="Распознанный текст")
            media_result_meta_var = tk.StringVar(value="Выберите файл слева, чтобы посмотреть результат.")

            media_controls_card = tk.Frame(files_page, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            media_controls_card.pack(fill="x", pady=(0, 10))
            media_controls_card.columnconfigure(0, weight=1)
            tk.Label(
                media_controls_card,
                text="Очередь распознавания",
                bg=card_bg,
                fg=text_primary,
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
            tk.Label(
                media_controls_card,
                text="Добавьте аудио/видео файлы. Обработка выполняется последовательно, по одному файлу.",
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 10),
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 8))

            media_btns = tk.Frame(media_controls_card, bg=card_bg)
            media_btns.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 8))

            select_media_btn = ttk.Button(media_btns, text="Выбрать файлы", style="Primary.TButton")
            select_media_btn.grid(row=0, column=0, padx=(0, 8))
            start_media_btn = ttk.Button(media_btns, text="Запустить обработку", style="Settings.TButton")
            start_media_btn.grid(row=0, column=1, padx=(0, 8))
            clear_media_local_btn = ttk.Button(media_btns, text="Очистить очередь", style="GhostDanger.TButton")
            clear_media_local_btn.grid(row=0, column=2)

            media_status_label = tk.Label(
                media_controls_card,
                textvariable=media_status_var,
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 10),
                justify="left",
                wraplength=980,
            )
            media_status_label.grid(row=3, column=0, sticky="w", padx=14, pady=(0, 12))

            media_main = tk.Frame(files_page, bg=c_bg)
            media_main.pack(fill="both", expand=True)
            media_main.columnconfigure(0, weight=1)
            media_main.columnconfigure(1, weight=1)
            media_main.rowconfigure(0, weight=1)

            media_queue_card = tk.Frame(media_main, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            media_queue_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            media_queue_card.columnconfigure(0, weight=1)
            media_queue_card.rowconfigure(1, weight=1)
            tk.Label(media_queue_card, text="Файлы очереди", bg=card_bg, fg=text_primary, font=("Segoe UI", 11, "bold")).grid(
                row=0, column=0, sticky="w", padx=12, pady=(12, 6)
            )

            media_queue_wrap = tk.Frame(media_queue_card, bg=card_bg)
            media_queue_wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            media_queue_wrap.columnconfigure(0, weight=1)
            media_queue_wrap.rowconfigure(0, weight=1)

            media_queue_canvas = tk.Canvas(media_queue_wrap, highlightthickness=0, bd=0, bg="#f5f7fb")
            media_queue_scroll = ttk.Scrollbar(media_queue_wrap, orient="vertical", command=media_queue_canvas.yview)
            media_queue_canvas.configure(yscrollcommand=media_queue_scroll.set)
            media_queue_canvas.grid(row=0, column=0, sticky="nsew")
            media_queue_scroll.grid(row=0, column=1, sticky="ns")

            media_items_frame = tk.Frame(media_queue_canvas, bg=c_bg)
            media_queue_window = media_queue_canvas.create_window((0, 0), window=media_items_frame, anchor="nw")
            _media_layout_state = {"canvas_width": -1}

            def _on_media_items_config(_e=None):
                media_queue_canvas.configure(scrollregion=media_queue_canvas.bbox("all"))

            def _on_media_canvas_config(_e=None):
                current_width = int(media_queue_canvas.winfo_width())
                if current_width <= 0:
                    return
                if _media_layout_state["canvas_width"] == current_width:
                    return
                _media_layout_state["canvas_width"] = current_width
                media_queue_canvas.itemconfigure(media_queue_window, width=current_width)

            media_items_frame.bind("<Configure>", _on_media_items_config)
            media_queue_canvas.bind("<Configure>", _on_media_canvas_config)

            media_result_card = tk.Frame(media_main, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            media_result_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            media_result_card.columnconfigure(0, weight=1)
            media_result_card.rowconfigure(2, weight=1)

            tk.Label(media_result_card, textvariable=media_result_title_var, bg=card_bg, fg=text_primary, font=("Segoe UI", 11, "bold")).grid(
                row=0, column=0, sticky="w", padx=12, pady=(12, 2)
            )
            tk.Label(
                media_result_card,
                textvariable=media_result_meta_var,
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=520,
            ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 6))

            media_result_text_wrap = tk.Frame(media_result_card, bg=card_bg)
            media_result_text_wrap.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 8))
            media_result_text_wrap.columnconfigure(0, weight=1)
            media_result_text_wrap.rowconfigure(0, weight=1)

            media_result_text = tk.Text(
                media_result_text_wrap,
                wrap="word",
                bg="#f8fafc",
                fg=text_primary,
                insertbackground=text_primary,
                highlightthickness=1,
                highlightbackground=c_border,
                relief="flat",
                font=("Segoe UI", 10),
                padx=10,
                pady=8,
            )
            media_result_scroll = ttk.Scrollbar(media_result_text_wrap, orient="vertical", command=media_result_text.yview)
            media_result_text.configure(yscrollcommand=media_result_scroll.set)
            media_result_text.grid(row=0, column=0, sticky="nsew")
            media_result_scroll.grid(row=0, column=1, sticky="ns")
            media_result_text.configure(state="disabled")

            media_actions = tk.Frame(media_result_card, bg=card_bg)
            media_actions.grid(row=3, column=0, sticky="e", padx=12, pady=(0, 12))
            copy_media_btn = ttk.Button(media_actions, text="Копировать", style="Settings.TButton")
            copy_media_btn.grid(row=0, column=0, padx=(0, 8))
            export_txt_btn = ttk.Button(media_actions, text="Экспорт TXT", style="Settings.TButton")
            export_txt_btn.grid(row=0, column=1, padx=(0, 8))
            export_srt_btn = ttk.Button(media_actions, text="Экспорт SRT", style="Settings.TButton")
            export_srt_btn.grid(row=0, column=2)

            clear_media_btn = ttk.Button(top, text="Очистить очередь", style="GhostDanger.TButton")
            clear_media_btn.grid(row=0, column=2, sticky="e", padx=(6, 0))

            media_status_colors = {
                "queued": ("В очереди", "#334155"),
                "processing": ("Обработка", "#0369a1"),
                "done": ("Готово", "#047857"),
                "error": ("Ошибка", "#be123c"),
            }

            def _media_format_duration(duration_sec: float) -> str:
                seconds = max(0, int(round(float(duration_sec or 0.0))))
                mm = seconds // 60
                ss = seconds % 60
                return f"{mm:02d}:{ss:02d}"

            def _media_set_result_text(text: str, preserve_scroll: bool = False, reset_to_top: bool = False) -> None:
                yview_before = media_result_state.get("last_yview", (0.0, 1.0))
                if preserve_scroll:
                    try:
                        yview_before = media_result_text.yview()
                    except Exception:
                        yview_before = media_result_state.get("last_yview", (0.0, 1.0))
                media_result_text.configure(state="normal")
                media_result_text.delete("1.0", "end")
                media_result_text.insert("1.0", str(text or ""))
                media_result_text.configure(state="disabled")
                try:
                    if reset_to_top:
                        media_result_text.yview_moveto(0.0)
                    elif preserve_scroll:
                        media_result_text.yview_moveto(float(yview_before[0]))
                    media_result_state["last_yview"] = media_result_text.yview()
                except Exception:
                    pass

            def _set_media_buttons_state(selected_item: Optional[dict], snapshot: dict) -> None:
                has_text = bool(str((selected_item or {}).get("text", "")).strip())
                has_items = bool(snapshot.get("items"))
                processing = bool(snapshot.get("is_processing"))
                clear_state = "disabled" if processing or not has_items else "normal"
                start_state = "disabled" if processing or not has_items else "normal"
                copy_state = "normal" if has_text else "disabled"
                export_state = "normal" if has_text else "disabled"
                clear_media_local_btn.configure(state=clear_state)
                clear_media_btn.configure(state=clear_state)
                start_media_btn.configure(state=start_state)
                copy_media_btn.configure(state=copy_state)
                export_txt_btn.configure(state=export_state)
                export_srt_btn.configure(state=export_state)

            def _render_media_queue(snapshot: dict) -> None:
                items = list(snapshot.get("items") or [])
                active_id = str(snapshot.get("active_id", "") or "")
                processing = bool(snapshot.get("is_processing"))
                media_snapshot["value"] = {"items": items, "active_id": active_id, "is_processing": processing}
                selected_before = str(media_selected_id.get("value", "") or "")
                item_ids = {str(item.get("id", "")) for item in items}

                if selected_before and selected_before in item_ids:
                    media_selected_id["value"] = selected_before
                elif active_id and active_id in item_ids:
                    media_selected_id["value"] = active_id
                elif items:
                    media_selected_id["value"] = str(items[0].get("id", ""))
                else:
                    media_selected_id["value"] = ""

                for child in media_items_frame.winfo_children():
                    child.destroy()

                selected_item: Optional[dict] = None
                if not items:
                    empty = tk.Frame(media_items_frame, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
                    empty.pack(fill="x", pady=(0, 8))
                    tk.Label(empty, text="Очередь пуста", bg=card_bg, fg=text_primary, font=("Segoe UI", 11, "bold")).pack(
                        anchor="w", padx=12, pady=(12, 4)
                    )
                    tk.Label(
                        empty,
                        text="Нажмите «Выбрать файлы», чтобы добавить аудио или видео для пакетной обработки.",
                        bg=card_bg,
                        fg=text_secondary,
                        font=("Segoe UI", 10),
                        wraplength=500,
                        justify="left",
                    ).pack(anchor="w", padx=12, pady=(0, 12))
                    media_result_title_var.set("Распознанный текст")
                    media_result_meta_var.set("Выберите файл слева, чтобы посмотреть результат.")
                    _media_set_result_text("", reset_to_top=True)
                    media_result_state["last_item_id"] = ""
                    media_result_state["last_signature"] = ""
                    _set_media_buttons_state(None, snapshot)
                    return

                for item in items:
                    item_id = str(item.get("id", ""))
                    status_key = str(item.get("status", "queued")).lower()
                    status_text, status_color = media_status_colors.get(status_key, ("В очереди", "#334155"))
                    is_selected = media_selected_id["value"] == item_id
                    row_bg = "#eef8ff" if is_selected else c_surface
                    row_border = "#9ec5e9" if is_selected else c_border

                    card = tk.Frame(media_items_frame, bg=row_bg, highlightthickness=1, highlightbackground=row_border)
                    card.pack(fill="x", pady=(0, 8))
                    card.columnconfigure(0, weight=1)

                    top_row = tk.Frame(card, bg=row_bg)
                    top_row.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 2))
                    top_row.columnconfigure(0, weight=1)
                    tk.Label(
                        top_row,
                        text=str(item.get("name", "Без названия")),
                        bg=row_bg,
                        fg=text_primary,
                        font=("Segoe UI", 10, "bold"),
                        anchor="w",
                    ).grid(row=0, column=0, sticky="w")
                    tk.Label(
                        top_row,
                        text=status_text,
                        bg=row_bg,
                        fg=status_color,
                        font=("Segoe UI", 9, "bold"),
                        anchor="e",
                    ).grid(row=0, column=1, sticky="e")

                    meta_parts = []
                    if item.get("duration_sec"):
                        meta_parts.append(f"Длительность: {_media_format_duration(float(item.get('duration_sec', 0.0)))}")
                    if status_key == "processing":
                        try:
                            p = float(item.get("progress", 0.0))
                        except Exception:
                            p = 0.0
                        meta_parts.append(f"Прогресс: {max(0.0, min(100.0, p)):.0f}%")
                    if status_key == "error" and str(item.get("error", "")).strip():
                        meta_parts.append(str(item.get("error", "")).strip())
                    tk.Label(
                        card,
                        text=" · ".join(meta_parts) if meta_parts else "Ожидает обработки",
                        bg=row_bg,
                        fg=text_secondary,
                        font=("Segoe UI", 9),
                        justify="left",
                        anchor="w",
                    ).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))

                    if is_selected:
                        selected_item = item

                    def _select_media(_e=None, iid=item_id):
                        media_selected_id["value"] = iid
                        _render_media_queue(media_snapshot["value"])

                    for w in (card, top_row):
                        w.bind("<Button-1>", _select_media)
                        for sub in w.winfo_children():
                            sub.bind("<Button-1>", _select_media)

                if selected_item is None and items:
                    selected_item = items[0]
                    media_selected_id["value"] = str(selected_item.get("id", ""))

                title = str(selected_item.get("name", "Распознанный текст"))
                st = str(selected_item.get("status", "queued")).lower()
                st_label, _ = media_status_colors.get(st, ("В очереди", "#334155"))
                media_result_title_var.set(title)
                meta = [f"Статус: {st_label}"]
                if selected_item.get("duration_sec"):
                    meta.append(f"Длительность: {_media_format_duration(float(selected_item.get('duration_sec', 0.0)))}")
                if st == "error" and str(selected_item.get("error", "")).strip():
                    meta.append(f"Ошибка: {str(selected_item.get('error', '')).strip()}")
                media_result_meta_var.set(" В· ".join(meta))
                selected_id = str(selected_item.get("id", ""))
                selected_text = str(selected_item.get("text", ""))
                selected_signature = "||".join(
                    [
                        selected_id,
                        str(selected_item.get("status", "")),
                        str(selected_item.get("duration_sec", "")),
                        str(selected_item.get("error", "")),
                        selected_text,
                    ]
                )
                if media_result_state["last_item_id"] == selected_id:
                    if media_result_state["last_signature"] != selected_signature:
                        _media_set_result_text(selected_text, preserve_scroll=True)
                else:
                    _media_set_result_text(selected_text, reset_to_top=True)
                media_result_state["last_item_id"] = selected_id
                media_result_state["last_signature"] = selected_signature
                _set_media_buttons_state(selected_item, snapshot)
                media_rendered["value"] = True

            def _poll_media_queue() -> None:
                media_poll_job["id"] = None
                if not callable(on_get_media_queue_cb):
                    return
                active_media_page = section_title_var.get() == "Транскрибация аудио и видео"
                current_processing = bool((media_snapshot.get("value") or {}).get("is_processing", False))
                if not active_media_page and not current_processing:
                    if win.winfo_exists():
                        media_poll_job["id"] = win.after(2600, _poll_media_queue)
                    return
                try:
                    snap = on_get_media_queue_cb()
                    if isinstance(snap, dict):
                        media_data.clear()
                        media_data.update(dict(snap))
                        if active_media_page:
                            _render_media_queue(dict(snap))
                        else:
                            media_snapshot["value"] = {
                                "items": list(snap.get("items") or []),
                                "active_id": str(snap.get("active_id", "") or ""),
                                "is_processing": bool(snap.get("is_processing", False)),
                            }
                except Exception as err:
                    media_status_var.set(f"Ошибка чтения очереди: {err}")
                    media_status_label.configure(fg="#9f1239")
                finally:
                    if win.winfo_exists():
                        media_poll_job["id"] = win.after(900 if active_media_page else 1600, _poll_media_queue)

            def _restart_media_poll() -> None:
                poll_job = media_poll_job.get("id")
                if poll_job:
                    try:
                        win.after_cancel(poll_job)
                    except Exception:
                        pass
                media_poll_job["id"] = None
                _poll_media_queue()

            def _start_media_processing_ui() -> None:
                if not callable(on_start_media_processing_cb):
                    media_status_var.set("Обработчик запуска обработки недоступен")
                    media_status_label.configure(fg="#9f1239")
                    return
                try:
                    ok, message = on_start_media_processing_cb()
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                media_status_var.set(str(message))
                media_status_label.configure(fg="#2b7a0b" if ok else "#9f1239")
                _restart_media_poll()

            def _add_media_files_ui() -> None:
                paths = filedialog.askopenfilenames(
                    parent=win,
                    title="Выберите аудио или видео файлы",
                    filetypes=[
                        ("Медиафайлы", "*.wav *.mp3 *.m4a *.ogg *.flac *.aac *.wma *.mp4 *.mkv *.mov *.avi *.webm *.mpg *.mpeg"),
                        ("Аудио", "*.wav *.mp3 *.m4a *.ogg *.flac *.aac *.wma"),
                        ("Видео", "*.mp4 *.mkv *.mov *.avi *.webm *.mpg *.mpeg"),
                        ("Все файлы", "*.*"),
                    ],
                )
                if not paths:
                    return
                if not callable(on_add_media_files_cb):
                    media_status_var.set("Обработчик добавления файлов недоступен")
                    media_status_label.configure(fg="#9f1239")
                    return
                try:
                    ok, message, snap = on_add_media_files_cb(list(paths))
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", {}
                media_status_var.set(str(message))
                media_status_label.configure(fg="#2b7a0b" if ok else "#9f1239")
                if isinstance(snap, dict):
                    _render_media_queue(dict(snap))
                if ok:
                    _start_media_processing_ui()
                else:
                    _restart_media_poll()

            def _clear_media_queue_ui() -> None:
                if not callable(on_clear_media_queue_cb):
                    media_status_var.set("Обработчик очистки очереди недоступен")
                    media_status_label.configure(fg="#9f1239")
                    return
                if not messagebox.askyesno("Очистка очереди", "Удалить все файлы из очереди?", parent=win):
                    return
                try:
                    ok, message, snap = on_clear_media_queue_cb()
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", {}
                media_status_var.set(str(message))
                media_status_label.configure(fg="#2b7a0b" if ok else "#9f1239")
                if isinstance(snap, dict):
                    _render_media_queue(dict(snap))
                _restart_media_poll()

            def _selected_media_item() -> Optional[dict]:
                snap = dict(media_snapshot.get("value") or {})
                for item in list(snap.get("items") or []):
                    if str(item.get("id", "")) == str(media_selected_id.get("value", "")):
                        return dict(item)
                return None

            def _copy_media_result_ui() -> None:
                item = _selected_media_item()
                if not item:
                    return
                if callable(on_copy_media_result_cb):
                    try:
                        ok, message = on_copy_media_result_cb(str(item.get("id", "")))
                    except Exception as err:
                        ok, message = False, f"Ошибка: {err}"
                else:
                    txt = str(item.get("text", "")).strip()
                    if not txt:
                        ok, message = False, "Нет текста для копирования"
                    else:
                        try:
                            pyperclip.copy(txt)
                            ok, message = True, "Текст скопирован"
                        except Exception as err:
                            ok, message = False, f"Ошибка: {err}"
                media_status_var.set(str(message))
                media_status_label.configure(fg="#2b7a0b" if ok else "#9f1239")

            def _export_media_result_ui(fmt: str) -> None:
                item = _selected_media_item()
                if not item:
                    return
                base_name = Path(str(item.get("name", "result"))).stem
                safe_name = VoiceProApp._sanitize_filename(base_name)
                ext = "srt" if str(fmt).strip().lower() == "srt" else "txt"
                save_path = filedialog.asksaveasfilename(
                    parent=win,
                    title=("Экспорт SRT" if ext == "srt" else "Экспорт TXT"),
                    defaultextension=f".{ext}",
                    initialfile=f"{safe_name}.{ext}",
                    filetypes=(
                        [("SubRip (*.srt)", "*.srt"), ("Все файлы", "*.*")]
                        if ext == "srt"
                        else [("Текст (*.txt)", "*.txt"), ("Все файлы", "*.*")]
                    ),
                )
                if not save_path:
                    return
                if not callable(on_export_media_result_cb):
                    media_status_var.set("Обработчик экспорта недоступен")
                    media_status_label.configure(fg="#9f1239")
                    return
                try:
                    try:
                        ok, message = on_export_media_result_cb(str(item.get("id", "")), str(fmt), str(save_path))
                    except TypeError:
                        ok, message = on_export_media_result_cb(str(item.get("id", "")), str(fmt))
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                media_status_var.set(str(message))
                media_status_label.configure(fg="#2b7a0b" if ok else "#9f1239")

            select_media_btn.configure(command=_add_media_files_ui)
            start_media_btn.configure(command=_start_media_processing_ui)
            clear_media_local_btn.configure(command=_clear_media_queue_ui)
            clear_media_btn.configure(command=_clear_media_queue_ui)
            copy_media_btn.configure(command=_copy_media_result_ui)
            export_txt_btn.configure(command=lambda: _export_media_result_ui("txt"))
            export_srt_btn.configure(command=lambda: _export_media_result_ui("srt"))

            # Initial render is deferred until the user opens this page.

            meeting_page.columnconfigure(0, weight=1)
            meeting_data = dict(config_data.get("_meeting_session") or {})
            meeting_poll_job = {"id": None}

            meeting_controls_card = tk.Frame(meeting_page, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            meeting_controls_card.pack(fill="x", pady=(0, 10))
            meeting_controls_card.columnconfigure(0, weight=1)

            tk.Label(
                meeting_controls_card,
                text="Voice Meads AI · Онлайн-встречи",
                bg=card_bg,
                fg=text_primary,
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
            tk.Label(
                meeting_controls_card,
                text="Захват микрофона и системного звука, транскрипт встречи и автоматическое резюме.",
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 10),
                justify="left",
                wraplength=900,
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 10))

            meeting_status_var = tk.StringVar(value="Сессия не запущена.")
            meeting_runtime_var = tk.StringVar(value="")
            meeting_levels_var = tk.StringVar(value="")
            meeting_backend_var = tk.StringVar(value="")
            meeting_loopback_var = tk.StringVar(value="")
            meeting_error_var = tk.StringVar(value="")
            meeting_summary_status_var = tk.StringVar(value="")

            meeting_btns = tk.Frame(meeting_controls_card, bg=card_bg)
            meeting_btns.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))
            start_meeting_btn = ttk.Button(meeting_btns, text="Начать встречу", style="Primary.TButton")
            start_meeting_btn.grid(row=0, column=0, padx=(0, 8))
            pause_meeting_btn = ttk.Button(meeting_btns, text="Пауза", style="Settings.TButton")
            pause_meeting_btn.grid(row=0, column=1, padx=(0, 8))
            stop_meeting_btn = ttk.Button(meeting_btns, text="Остановить", style="Danger.TButton")
            stop_meeting_btn.grid(row=0, column=2)
            retry_loopback_btn = ttk.Button(meeting_btns, text="Повторить инициализацию loopback", style="Settings.TButton")
            retry_loopback_btn.grid(row=0, column=3, padx=(8, 0))

            tk.Label(meeting_controls_card, textvariable=meeting_status_var, bg=card_bg, fg="#0f766e", font=("Segoe UI", 10, "bold")).grid(
                row=3, column=0, sticky="w", padx=14, pady=(0, 4)
            )
            tk.Label(meeting_controls_card, textvariable=meeting_runtime_var, bg=card_bg, fg=text_secondary, font=("Segoe UI", 9)).grid(
                row=4, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(meeting_controls_card, textvariable=meeting_levels_var, bg=card_bg, fg=text_secondary, font=("Segoe UI", 9)).grid(
                row=5, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(meeting_controls_card, textvariable=meeting_backend_var, bg=card_bg, fg=text_secondary, font=("Segoe UI", 9)).grid(
                row=6, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(meeting_controls_card, textvariable=meeting_loopback_var, bg=card_bg, fg=text_secondary, font=("Segoe UI", 9)).grid(
                row=7, column=0, sticky="w", padx=14, pady=(0, 2)
            )
            tk.Label(meeting_controls_card, textvariable=meeting_error_var, bg=card_bg, fg="#b91c1c", font=("Segoe UI", 9), wraplength=980, justify="left").grid(
                row=8, column=0, sticky="w", padx=14, pady=(0, 12)
            )

            meeting_main = tk.Frame(meeting_page, bg=c_bg)
            meeting_main.pack(fill="both", expand=True)
            meeting_main.columnconfigure(0, weight=1)
            meeting_main.columnconfigure(1, weight=1)
            meeting_main.rowconfigure(0, weight=1)

            transcript_card = tk.Frame(meeting_main, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            transcript_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
            transcript_card.rowconfigure(1, weight=1)
            transcript_card.columnconfigure(0, weight=1)
            tk.Label(transcript_card, text="Транскрипт встречи", bg=card_bg, fg=text_primary, font=("Segoe UI", 12, "bold")).grid(
                row=0, column=0, sticky="w", padx=12, pady=(10, 8)
            )
            meeting_transcript_txt = tk.Text(
                transcript_card,
                wrap="word",
                bg="#f8fafc",
                fg=text_primary,
                font=("Segoe UI", 10),
                relief="flat",
                padx=10,
                pady=10,
            )
            meeting_transcript_txt.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
            meeting_transcript_txt.configure(state="disabled")
            transcript_actions = tk.Frame(transcript_card, bg=card_bg)
            transcript_actions.grid(row=2, column=0, sticky="e", padx=12, pady=(0, 12))
            copy_transcript_btn = ttk.Button(transcript_actions, text="Копировать", style="Settings.TButton")
            copy_transcript_btn.grid(row=0, column=0)

            summary_card = tk.Frame(meeting_main, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            summary_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
            summary_card.rowconfigure(1, weight=1)
            summary_card.columnconfigure(0, weight=1)
            tk.Label(summary_card, text="Резюме встречи", bg=card_bg, fg=text_primary, font=("Segoe UI", 12, "bold")).grid(
                row=0, column=0, sticky="w", padx=12, pady=(10, 8)
            )
            meeting_summary_txt = tk.Text(
                summary_card,
                wrap="word",
                bg="#f8fafc",
                fg=text_primary,
                font=("Segoe UI", 10),
                relief="flat",
                padx=10,
                pady=10,
            )
            meeting_summary_txt.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
            meeting_summary_txt.configure(state="disabled")
            tk.Label(summary_card, textvariable=meeting_summary_status_var, bg=card_bg, fg=text_secondary, font=("Segoe UI", 9)).grid(
                row=2, column=0, sticky="w", padx=12, pady=(0, 8)
            )
            summary_actions = tk.Frame(summary_card, bg=card_bg)
            summary_actions.grid(row=3, column=0, sticky="e", padx=12, pady=(0, 12))
            build_summary_btn = ttk.Button(summary_actions, text="Сформировать", style="Primary.TButton")
            build_summary_btn.grid(row=0, column=0, padx=(0, 8))
            copy_summary_btn = ttk.Button(summary_actions, text="Копировать", style="Settings.TButton")
            copy_summary_btn.grid(row=0, column=1, padx=(0, 8))
            export_meeting_btn = ttk.Button(summary_actions, text="Экспорт TXT", style="Settings.TButton")
            export_meeting_btn.grid(row=0, column=2)

            meeting_render_state = {"last_transcript": None, "last_summary": None}

            def _meeting_set_text(widget: tk.Text, value: str, state_key: str) -> None:
                txt = str(value or "").strip()
                old = meeting_render_state.get(state_key)
                if old == txt:
                    return
                meeting_render_state[state_key] = txt
                widget.configure(state="normal")
                widget.delete("1.0", "end")
                widget.insert("1.0", txt or "Пока нет данных.")
                widget.configure(state="disabled")

            def _meeting_fmt_duration(sec: float) -> str:
                s = max(0, int(sec))
                h = s // 3600
                m = (s % 3600) // 60
                ss = s % 60
                if h > 0:
                    return f"{h:02d}:{m:02d}:{ss:02d}"
                return f"{m:02d}:{ss:02d}"

            def _render_meeting_status(snapshot: dict) -> None:
                snap = dict(snapshot or {})
                st = str(snap.get("state", "idle")).strip().lower()
                state_map = {
                    "idle": "Ожидание запуска.",
                    "recording": "Слушаю встречу…",
                    "paused": "Пауза.",
                    "transcribing": "Идёт транскрибация…",
                    "summarizing": "Формирую резюме…",
                    "done": "Сессия завершена.",
                    "error": "Ошибка сессии.",
                }
                meeting_status_var.set(state_map.get(st, "Ожидание запуска."))
                duration_sec = float(snap.get("duration_sec", 0.0) or 0.0)
                meeting_runtime_var.set(f"Таймер: {_meeting_fmt_duration(duration_sec)}")
                levels = dict(snap.get("audio_levels") or {})
                meeting_levels_var.set(f"Уровни · Микрофон: {int(levels.get('mic', 0) or 0)}% · Системный звук: {int(levels.get('system', 0) or 0)}%")
                system_backend = str(snap.get("system_backend", "")).strip().lower()
                system_backend_label = (
                    "soundcard"
                    if system_backend == "soundcard"
                    else "stereo-mix"
                    if system_backend == "system_input"
                    else "sounddevice"
                    if system_backend == "sounddevice"
                    else "нет"
                )
                meeting_backend_var.set(
                    f"Режим распознавания: {str(snap.get('backend_label', '')).strip() or '—'} · Канал системного звука: {system_backend_label}"
                )
                lp_ok = bool(snap.get("loopback_available", False))
                lp_name = str(snap.get("loopback_device_name", "")).strip()
                lp_channels = int(snap.get("loopback_channels", 0) or 0)
                lp_err = str(snap.get("loopback_error", "")).strip()
                lp_code = str(snap.get("loopback_error_code", "")).strip()
                if lp_ok:
                    meeting_loopback_var.set(f"Системный звук: подключён ({lp_name or 'loopback'}, {lp_channels} кан.)")
                else:
                    extra = f" Причина: {lp_err}" if lp_err else ""
                    if lp_code:
                        extra += f" (код {lp_code})"
                    meeting_loopback_var.set(f"Системный звук: недоступен.{extra}")
                err = str(snap.get("error", "") or "").strip()
                meeting_error_var.set(err)
                summary_hint = "Используйте «Сформировать», чтобы получить итоги встречи."
                if st == "summarizing":
                    summary_hint = "Резюме формируется…"
                elif str(snap.get("summary_text", "")).strip():
                    summary_hint = "Резюме готово."
                summary_backend = str(snap.get("summary_backend", "")).strip()
                summary_model = str(snap.get("summary_model", "")).strip()
                if summary_backend:
                    summary_hint = f"{summary_hint} · Источник: {summary_backend}" + (f" · {summary_model}" if summary_model else "")
                meeting_summary_status_var.set(summary_hint)

                running = bool(snap.get("is_running", False))
                paused = bool(snap.get("is_paused", False))
                start_meeting_btn.configure(state=("disabled" if running else "normal"))
                pause_meeting_btn.configure(state=("normal" if running else "disabled"))
                pause_meeting_btn.configure(text=("Продолжить" if paused else "Пауза"))
                stop_meeting_btn.configure(state=("normal" if running or st in {"transcribing"} else "disabled"))
                transcript = str(snap.get("transcript_text", ""))
                summary = str(snap.get("summary_text", ""))
                _meeting_set_text(meeting_transcript_txt, transcript, "last_transcript")
                _meeting_set_text(meeting_summary_txt, summary, "last_summary")
                copy_transcript_btn.configure(state=("normal" if transcript.strip() else "disabled"))
                copy_summary_btn.configure(state=("normal" if summary.strip() else "disabled"))
                export_meeting_btn.configure(state=("normal" if transcript.strip() or summary.strip() else "disabled"))
                build_summary_btn.configure(state=("normal" if transcript.strip() and st != "summarizing" else "disabled"))
                retry_loopback_btn.configure(state=("normal" if running else "disabled"))

            def _poll_meeting_status() -> None:
                meeting_poll_job["id"] = None
                if not callable(on_get_meeting_session_status_cb):
                    return
                active_meeting_page = section_title_var.get() == "Voice Meet AI"
                current_state = str((meeting_data or {}).get("state", "idle")).strip().lower()
                current_running = bool((meeting_data or {}).get("is_running", False))
                if not active_meeting_page and not current_running and current_state in {"idle", "done"}:
                    if win.winfo_exists():
                        meeting_poll_job["id"] = win.after(2600, _poll_meeting_status)
                    return
                try:
                    snap = on_get_meeting_session_status_cb()
                    if isinstance(snap, dict):
                        meeting_data.clear()
                        meeting_data.update(dict(snap))
                        if active_meeting_page:
                            _render_meeting_status(dict(snap))
                except Exception as err:
                    meeting_error_var.set(f"Ошибка чтения состояния: {err}")
                finally:
                    if win.winfo_exists():
                        meeting_poll_job["id"] = win.after(900 if active_meeting_page else 1700, _poll_meeting_status)

            def _restart_meeting_poll() -> None:
                poll_job = meeting_poll_job.get("id")
                if poll_job:
                    try:
                        win.after_cancel(poll_job)
                    except Exception:
                        pass
                meeting_poll_job["id"] = None
                _poll_meeting_status()

            def _start_meeting_ui() -> None:
                if not callable(on_start_meeting_capture_cb):
                    meeting_error_var.set("Обработчик запуска встречи недоступен.")
                    return
                try:
                    ok, msg = on_start_meeting_capture_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                meeting_error_var.set("" if ok else str(msg))
                if ok:
                    meeting_summary_status_var.set("Идёт запись встречи.")
                _restart_meeting_poll()

            def _pause_meeting_ui() -> None:
                if not callable(on_pause_meeting_capture_cb):
                    meeting_error_var.set("Обработчик паузы недоступен.")
                    return
                try:
                    ok, msg = on_pause_meeting_capture_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))
                _restart_meeting_poll()

            def _stop_meeting_ui() -> None:
                if not callable(on_stop_meeting_capture_cb):
                    meeting_error_var.set("Обработчик остановки недоступен.")
                    return
                try:
                    ok, msg = on_stop_meeting_capture_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))
                _restart_meeting_poll()

            def _build_meeting_summary_ui() -> None:
                if not callable(on_build_meeting_summary_cb):
                    meeting_error_var.set("Обработчик резюме недоступен.")
                    return
                try:
                    ok, msg = on_build_meeting_summary_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))
                _restart_meeting_poll()

            def _copy_meeting_ui(kind: str) -> None:
                if not callable(on_copy_meeting_text_cb):
                    meeting_error_var.set("Обработчик копирования недоступен.")
                    return
                try:
                    ok, msg = on_copy_meeting_text_cb(str(kind))
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))

            def _export_meeting_ui() -> None:
                save_path = filedialog.asksaveasfilename(
                    parent=win,
                    title="Экспорт резюме встречи",
                    defaultextension=".txt",
                    initialfile=f"meeting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    filetypes=[("Текст (*.txt)", "*.txt"), ("Все файлы", "*.*")],
                )
                if not save_path:
                    return
                if not callable(on_export_meeting_txt_cb):
                    meeting_error_var.set("Обработчик экспорта недоступен.")
                    return
                try:
                    try:
                        ok, msg = on_export_meeting_txt_cb(str(save_path))
                    except TypeError:
                        ok, msg = on_export_meeting_txt_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))

            def _retry_loopback_ui() -> None:
                if not callable(on_retry_meeting_loopback_cb):
                    meeting_error_var.set("Обработчик переинициализации loopback недоступен.")
                    return
                try:
                    ok, msg = on_retry_meeting_loopback_cb()
                except Exception as err:
                    ok, msg = False, f"Ошибка: {err}"
                if not ok:
                    meeting_error_var.set(str(msg))
                _restart_meeting_poll()

            start_meeting_btn.configure(command=_start_meeting_ui)
            pause_meeting_btn.configure(command=_pause_meeting_ui)
            stop_meeting_btn.configure(command=_stop_meeting_ui)
            build_summary_btn.configure(command=_build_meeting_summary_ui)
            copy_transcript_btn.configure(command=lambda: _copy_meeting_ui("transcript"))
            copy_summary_btn.configure(command=lambda: _copy_meeting_ui("summary"))
            export_meeting_btn.configure(command=_export_meeting_ui)
            retry_loopback_btn.configure(command=_retry_loopback_ui)
            # Initial render is deferred until the user opens this page.

            rules_data = dict(config_data.get("_autoreplace") or {})
            rules_enabled_var = tk.BooleanVar(value=bool(rules_data.get("enabled", True)))
            rules_local: list[dict] = list(rules_data.get("rules") or [])
            rules_rendered = {"value": False}

            rules_page.columnconfigure(0, weight=1)

            rules_info = tk.Frame(rules_page, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            rules_info.pack(fill="x", pady=(0, 10))
            tk.Label(rules_info, text="Как работает автозамена", bg=card_bg, fg=text_primary, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(12, 6))
            tk.Label(
                rules_info,
                text=(
                    "Текст: простая замена одной фразы на другую (например, «щас» > «сейчас»).\n"
                    "RegExp: гибкая замена по регулярному выражению для сложных шаблонов.\n"
                    "Флаги для RegExp: i — без учёта регистра, m — многострочный режим, s — захват переносов, u — Unicode."
                ),
                justify="left",
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 10),
                wraplength=900,
            ).pack(anchor="w", padx=14, pady=(0, 12))

            rules_toggle_card = tk.Frame(rules_page, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
            rules_toggle_card.pack(fill="x", pady=(0, 10))
            rules_toggle_card.grid_columnconfigure(0, weight=1)
            tk.Label(
                rules_toggle_card,
                text="Активность модуля",
                bg=card_bg,
                fg=text_primary,
                font=("Segoe UI", 12, "bold"),
            ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 2))
            tk.Label(
                rules_toggle_card,
                text="Если переключатель включён, правила автозамены будут применяться к результату распознавания.",
                wraplength=760,
                justify="left",
                bg=card_bg,
                fg=text_secondary,
                font=("Segoe UI", 10),
            ).grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))

            rules_toggle_canvas = tk.Canvas(rules_toggle_card, width=52, height=30, bg=card_bg, highlightthickness=0, bd=0)
            rules_toggle_canvas.grid(row=0, column=1, rowspan=2, sticky="e", padx=14)
            rules_toggle_state = {"value": bool(rules_enabled_var.get())}

            def _draw_rules_toggle() -> None:
                rules_toggle_canvas.delete("all")
                on = bool(rules_toggle_state["value"])
                track = c_accent if on else "#3a4254"
                knob_x = 36 if on else 16
                rules_toggle_canvas.create_oval(8, 7, 22, 23, fill=track, outline=track)
                rules_toggle_canvas.create_rectangle(15, 7, 37, 23, fill=track, outline=track)
                rules_toggle_canvas.create_oval(30, 7, 44, 23, fill=track, outline=track)
                rules_toggle_canvas.create_oval(knob_x - 7, 8, knob_x + 7, 22, fill="#f8fafc", outline="#d1d9e6")

            _draw_rules_toggle()

            rules_row_bg = c_surface
            rules_row_hover = "#f6fbff"
            rules_row_selected = "#eef8ff"
            rules_row_disabled = "#f8fafc"
            rules_row_border = c_border
            rules_row_border_hover = "#cbddee"
            rules_row_border_selected = "#9ec5e9"

            rules_list_wrap = tk.Frame(rules_page, bg=c_bg)
            rules_list_wrap.pack(fill="both", expand=True)
            rules_list_wrap.columnconfigure(0, weight=1)
            rules_list_wrap.rowconfigure(0, weight=1)

            rules_canvas = tk.Canvas(rules_list_wrap, highlightthickness=0, bd=0, bg="#f5f7fb")
            rules_scroll = ttk.Scrollbar(rules_list_wrap, orient="vertical", command=rules_canvas.yview)
            rules_canvas.configure(yscrollcommand=rules_scroll.set)
            rules_canvas.grid(row=0, column=0, sticky="nsew")
            rules_scroll.grid(row=0, column=1, sticky="ns")

            rules_items_frame = tk.Frame(rules_canvas, bg=c_bg)
            rules_window = rules_canvas.create_window((0, 0), window=rules_items_frame, anchor="nw")

            _rules_layout_state = {"canvas_width": -1}

            def _on_rules_items_config(_e=None):
                rules_canvas.configure(scrollregion=rules_canvas.bbox("all"))

            def _on_rules_canvas_config(_e=None):
                current_width = int(rules_canvas.winfo_width())
                if current_width <= 0:
                    return
                if _rules_layout_state["canvas_width"] == current_width:
                    return
                _rules_layout_state["canvas_width"] = current_width
                rules_canvas.itemconfigure(rules_window, width=current_width)

            rules_items_frame.bind("<Configure>", _on_rules_items_config)
            rules_canvas.bind("<Configure>", _on_rules_canvas_config)

            def _normalize_rule(rule: dict, fallback_id: str) -> dict:
                raw_mode = str(rule.get("mode", "Text")).lower()
                return {
                    "id": str(rule.get("id", "")).strip() or fallback_id,
                    "enabled": bool(rule.get("enabled", True)),
                    "pattern": str(rule.get("pattern", "")),
                    "replacement": str(rule.get("replacement", "")),
                    "mode": "RegExp" if raw_mode in {"regexp", "regex", "rx"} else "Текст",
                    "flags": str(rule.get("flags", "u")),
                }

            def _collect_rules_from_widgets(rows: list[dict]) -> list[dict]:
                out: list[dict] = []
                for row in rows:
                    out.append(
                        _normalize_rule(
                            {
                                "id": row["id"],
                                "enabled": bool(row["enabled_var"].get()),
                                "pattern": row["pattern_var"].get(),
                                "replacement": row["replacement_var"].get(),
                                "mode": "RegExp" if row["mode_var"].get() == "RegExp" else "Text",
                                "flags": row["flags_var"].get(),
                            },
                            row["id"],
                        )
                    )
                return out

            rules_row_state: list[dict] = []
            rules_save_job: dict[str, object] = {"id": None}

            def _persist_rules(should_rerender: bool = True) -> None:
                nonlocal rules_local
                if not callable(on_set_autoreplace_cb):
                    status_var.set("Обработчик автозамены недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                payload_rules = _collect_rules_from_widgets(rules_row_state)
                try:
                    ok, message, snap = on_set_autoreplace_cb(bool(rules_toggle_state["value"]), payload_rules)
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", {"enabled": bool(rules_toggle_state["value"]), "rules": payload_rules}
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    rules_local = list((snap or {}).get("rules") or [])
                    rules_enabled_var.set(bool((snap or {}).get("enabled", True)))
                    rules_toggle_state["value"] = bool(rules_enabled_var.get())
                    _draw_rules_toggle()
                    if should_rerender:
                        _render_rules()

            def _schedule_persist_rules(delay_ms: int = 220) -> None:
                job = rules_save_job.get("id")
                if job:
                    try:
                        win.after_cancel(job)
                    except Exception:
                        pass
                rules_save_job["id"] = win.after(
                    max(80, int(delay_ms)),
                    lambda: (_persist_rules(False), rules_save_job.update({"id": None})),
                )

            def _render_rules() -> None:
                nonlocal rules_row_state
                for child in rules_items_frame.winfo_children():
                    child.destroy()
                rules_row_state = []
                if not rules_local:
                    empty = tk.Frame(rules_items_frame, bg=card_bg, highlightthickness=1, highlightbackground=card_border)
                    empty.pack(fill="x", pady=(0, 8))
                    tk.Label(empty, text="Правил пока нет", bg=card_bg, fg=text_primary, font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
                    tk.Label(empty, text="Нажмите «Добавить правило», чтобы создать первую автозамену.", bg=card_bg, fg=text_secondary, font=("Segoe UI", 10)).pack(anchor="w", padx=12, pady=(0, 10))
                    return

                header = tk.Frame(rules_items_frame, bg=card_bg, highlightthickness=1, highlightbackground=card_border, padx=10, pady=8)
                header.pack(fill="x", pady=(0, 6))
                header.grid_columnconfigure(0, minsize=56)
                header.grid_columnconfigure(1, weight=5, uniform="rules_cols")
                header.grid_columnconfigure(2, minsize=84)
                header.grid_columnconfigure(3, minsize=34)
                header.grid_columnconfigure(4, weight=5, uniform="rules_cols")
                header.grid_columnconfigure(5, minsize=118)
                header.grid_columnconfigure(6, minsize=108)
                tk.Label(header, text="Шаблон", bg=card_bg, fg=text_secondary, font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(0, 8))
                tk.Label(header, text="Флаги", bg=card_bg, fg=text_secondary, font=("Segoe UI", 9, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 8))
                tk.Label(header, text="Замена", bg=card_bg, fg=text_secondary, font=("Segoe UI", 9, "bold")).grid(row=0, column=4, sticky="w", padx=(0, 8))
                tk.Label(header, text="Тип", bg=card_bg, fg=text_secondary, font=("Segoe UI", 9, "bold")).grid(row=0, column=5, sticky="w", padx=(0, 8))

                def _pointer_inside(widget: tk.Widget, root_widget: tk.Widget) -> bool:
                    try:
                        px, py = root_widget.winfo_pointerxy()
                        hovered = root_widget.winfo_containing(px, py)
                        while hovered is not None:
                            if hovered == widget:
                                return True
                            hovered = hovered.master
                    except Exception:
                        return False
                    return False

                for idx, raw_rule in enumerate(rules_local):
                    rule = _normalize_rule(raw_rule, f"rule-{idx+1}")
                    row_card = tk.Frame(
                        rules_items_frame,
                        bg=rules_row_bg,
                        highlightthickness=1,
                        highlightbackground=rules_row_border,
                        highlightcolor=rules_row_border_selected,
                        padx=10,
                        pady=10,
                        takefocus=1,
                    )
                    row_card.pack(fill="x", pady=(0, 8))
                    row_card.grid_columnconfigure(0, minsize=56)
                    row_card.grid_columnconfigure(1, weight=5, uniform="rules_cols")
                    row_card.grid_columnconfigure(2, minsize=84)
                    row_card.grid_columnconfigure(3, minsize=34)
                    row_card.grid_columnconfigure(4, weight=5, uniform="rules_cols")
                    row_card.grid_columnconfigure(5, minsize=118)
                    row_card.grid_columnconfigure(6, minsize=108)

                    enabled_var = tk.BooleanVar(value=bool(rule["enabled"]))
                    pattern_var = tk.StringVar(value=rule["pattern"])
                    replacement_var = tk.StringVar(value=rule["replacement"])
                    mode_var = tk.StringVar(value=rule["mode"])
                    flags_var = tk.StringVar(value=rule["flags"])

                    toggle_canvas = tk.Canvas(row_card, width=42, height=24, bg=rules_row_bg, highlightthickness=0, bd=0, cursor="hand2")
                    toggle_canvas.grid(row=0, column=0, padx=(10, 8), pady=10)

                    def _draw_row_toggle(canvas=toggle_canvas, var=enabled_var) -> None:
                        canvas.delete("all")
                        on = bool(var.get())
                        track = c_accent if on else "#3a4254"
                        knob_x = 30 if on else 12
                        canvas.create_oval(6, 6, 18, 18, fill=track, outline=track)
                        canvas.create_rectangle(12, 6, 30, 18, fill=track, outline=track)
                        canvas.create_oval(24, 6, 36, 18, fill=track, outline=track)
                        canvas.create_oval(knob_x - 6, 6, knob_x + 6, 18, fill="#f8fafc", outline="#d1d9e6")

                    def _toggle_row_enabled(_e=None, var=enabled_var, redraw=_draw_row_toggle):
                        var.set(not bool(var.get()))
                        redraw()
                        _update_row_visual()
                        _persist_rules()

                    toggle_canvas.bind("<Button-1>", _toggle_row_enabled)
                    _draw_row_toggle()

                    pattern_entry = ttk.Entry(row_card, textvariable=pattern_var, style="Settings.TEntry")
                    pattern_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=10)

                    flags_entry = ttk.Entry(row_card, textvariable=flags_var, width=8, style="Settings.TEntry")
                    flags_entry.grid(row=0, column=2, padx=(0, 8), pady=10)

                    tk.Label(row_card, text=">", bg=rules_row_bg, fg=text_secondary, font=("Segoe UI", 14, "bold")).grid(row=0, column=3, padx=(0, 8), pady=10)

                    replacement_entry = ttk.Entry(row_card, textvariable=replacement_var, style="Settings.TEntry")
                    replacement_entry.grid(row=0, column=4, sticky="ew", padx=(0, 8), pady=10)

                    mode_combo = ttk.Combobox(row_card, textvariable=mode_var, values=["Текст", "RegExp"], width=9, state="readonly", style="Dark.TCombobox")
                    mode_combo.grid(row=0, column=5, padx=(0, 8), pady=10)

                    inline_hint_var = tk.StringVar(value="")
                    inline_hint = tk.Label(
                        row_card,
                        textvariable=inline_hint_var,
                        bg=rules_row_bg,
                        fg=text_secondary,
                        font=("Segoe UI", 9),
                        justify="left",
                        anchor="w",
                        height=1,
                    )
                    inline_hint.grid(row=1, column=1, columnspan=5, sticky="w", padx=(0, 8), pady=(0, 2))

                    def _short_hint(txt: str, max_len: int = 120) -> str:
                        clean = (txt or "").replace("\n", " ").strip()
                        if len(clean) <= max_len:
                            return clean
                        return clean[: max_len - 1] + "…"

                    def _validate_rule_state() -> bool:
                        if mode_var.get() != "RegExp":
                            inline_hint_var.set("Текстовый режим: выполняется прямое сопоставление фразы.")
                            inline_hint.configure(fg=text_secondary)
                            return True
                        flags_raw = (flags_var.get() or "").strip().lower()
                        bad_flags = [ch for ch in flags_raw if ch not in {"i", "m", "s", "u"}]
                        if bad_flags:
                            inline_hint_var.set("Некорректные флаги: используйте только i, m, s, u.")
                            inline_hint.configure(fg=c_danger)
                            return False
                        re_flags = 0
                        if "i" in flags_raw:
                            re_flags |= re.IGNORECASE
                        if "m" in flags_raw:
                            re_flags |= re.MULTILINE
                        if "s" in flags_raw:
                            re_flags |= re.DOTALL
                        try:
                            re.compile(pattern_var.get() or "", re_flags)
                            inline_hint_var.set("RegExp корректен и готов к применению.")
                            inline_hint.configure(fg="#0f766e")
                            return True
                        except re.error as err:
                            inline_hint_var.set(_short_hint(f"Ошибка RegExp: {err}"))
                            inline_hint.configure(fg=c_danger)
                            return False

                    def _delete_rule(i=idx):
                        nonlocal rules_local
                        if not messagebox.askyesno("Удаление правила", "Удалить выбранное правило автозамены?", parent=win):
                            return
                        rules_local = [r for j, r in enumerate(rules_local) if j != i]
                        _render_rules()
                        _persist_rules()

                    delete_btn = tk.Button(
                        row_card,
                        text="Удалить",
                        command=_delete_rule,
                        bg="#fff1f2",
                        fg=c_danger,
                        activebackground="#ffe4e6",
                        activeforeground=c_danger,
                        relief="flat",
                        bd=0,
                        padx=12,
                        pady=7,
                        cursor="hand2",
                        font=("Segoe UI", 9, "bold"),
                    )
                    delete_btn.grid(row=0, column=6, padx=(0, 10), pady=10)

                    def _set_row_visual(is_hover: bool = False, rc=row_card, tc=toggle_canvas, ih=inline_hint, ev=enabled_var) -> None:
                        if _is_dragging_window():
                            return
                        is_disabled = (not bool(rules_toggle_state["value"])) or (not bool(ev.get()))
                        if is_hover:
                            base_bg = rules_row_hover
                            border = rules_row_border_hover
                        elif is_disabled:
                            base_bg = rules_row_disabled
                            border = rules_row_border
                        else:
                            base_bg = rules_row_bg
                            border = rules_row_border
                        rc.configure(bg=base_bg, highlightbackground=border)
                        tc.configure(bg=base_bg)
                        ih.configure(bg=base_bg)
                        for w in rc.winfo_children():
                            if isinstance(w, tk.Label):
                                w.configure(bg=base_bg)

                    def _update_row_visual(_e=None, set_visual=_set_row_visual) -> None:
                        set_visual(False)

                    def _on_row_leave(_e=None, holder=row_card, set_visual=_set_row_visual) -> None:
                        if _is_dragging_window():
                            return
                        win.after(
                            1,
                            lambda h=holder, sv=set_visual: sv(False) if not _pointer_inside(h, win) else None,
                        )

                    row_state = {
                        "id": rule["id"],
                        "enabled_var": enabled_var,
                        "pattern_var": pattern_var,
                        "replacement_var": replacement_var,
                        "mode_var": mode_var,
                        "flags_var": flags_var,
                    }
                    rules_row_state.append(row_state)

                    for w in (pattern_entry, flags_entry, replacement_entry):
                        w.bind("<FocusOut>", lambda _e: _persist_rules())
                        w.bind("<KeyRelease>", lambda _e, vf=_validate_rule_state: (_schedule_persist_rules(), vf()))
                        w.bind("<Return>", lambda _e: (_persist_rules(), "break")[1])
                    mode_combo.bind("<<ComboboxSelected>>", lambda _e, vf=_validate_rule_state: (_persist_rules(), vf()))
                    flags_entry.bind("<FocusOut>", lambda _e, vf=_validate_rule_state: vf(), add="+")
                    pattern_entry.bind("<FocusOut>", lambda _e, vf=_validate_rule_state: vf(), add="+")

                    for w in (
                        row_card,
                        toggle_canvas,
                        inline_hint,
                        pattern_entry,
                        flags_entry,
                        replacement_entry,
                        mode_combo,
                        delete_btn,
                    ):
                        w.bind("<Enter>", lambda _e, sv=_set_row_visual: sv(True))
                        w.bind("<Leave>", _on_row_leave)

                    _validate_rule_state()
                    _set_row_visual(False)
                rules_rendered["value"] = True

            def _toggle_rules_click(_e=None):
                rules_toggle_state["value"] = not bool(rules_toggle_state["value"])
                _draw_rules_toggle()
                _persist_rules()

            rules_toggle_canvas.bind("<Button-1>", _toggle_rules_click)

            def _on_add_rule():
                nonlocal rules_local
                new_rule = {
                    "id": f"rule-{int(time.time()*1000)}",
                    "enabled": True,
                    "pattern": "",
                    "replacement": "",
                    "mode": "Текст",
                    "flags": "u",
                }
                rules_local.insert(0, new_rule)
                _render_rules()
                _persist_rules()

            def _on_reset_rules():
                nonlocal rules_local
                if not callable(on_reset_autoreplace_cb):
                    status_var.set("Обработчик сброса автозамены недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message, snap = on_reset_autoreplace_cb()
                except Exception as err:
                    ok, message, snap = False, f"Ошибка: {err}", {"enabled": bool(rules_enabled_var.get()), "rules": rules_local}
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    rules_local = list((snap or {}).get("rules") or [])
                    rules_enabled_var.set(bool((snap or {}).get("enabled", True)))
                    rules_toggle_state["value"] = bool(rules_enabled_var.get())
                    _draw_rules_toggle()
                    _render_rules()

            add_rule_btn.configure(command=_on_add_rule)
            reset_rules_btn.configure(command=_on_reset_rules)

            def _ensure_rules_rendered() -> None:
                if rules_rendered["value"]:
                    return
                _render_rules()

            add_placeholder(
                help_page,
                "Как пользоваться",
                "Раздел справки включает быстрый старт, рекомендации по качеству записи, частые ошибки "
                "и пошаговую диагностику микрофона, горячей клавиши и вставки текста.",
            )

            settings_page.columnconfigure(0, weight=1)
            settings_page.rowconfigure(0, weight=1)
            settings_scroll_wrap = ttk.Frame(settings_page)
            settings_scroll_wrap.grid(row=0, column=0, sticky="nsew")
            settings_scroll_wrap.columnconfigure(0, weight=1)
            settings_scroll_wrap.rowconfigure(0, weight=1)

            settings_canvas = tk.Canvas(settings_scroll_wrap, highlightthickness=0, bd=0, bg=c_bg)
            settings_scroll = ttk.Scrollbar(settings_scroll_wrap, orient="vertical", command=settings_canvas.yview)
            settings_canvas.configure(yscrollcommand=settings_scroll.set)
            settings_canvas.grid(row=0, column=0, sticky="nsew")
            settings_scroll.grid(row=0, column=1, sticky="ns")

            settings_content = ttk.Frame(settings_canvas, padding=(0, 0, 6, 0))
            settings_window = settings_canvas.create_window((0, 0), window=settings_content, anchor="nw")
            settings_content.columnconfigure(0, weight=1)

            _settings_layout_state = {"canvas_width": -1}

            def _on_settings_content_config(_e=None):
                # Recompute scroll region only when content geometry changes.
                settings_canvas.configure(scrollregion=settings_canvas.bbox("all"))

            def _on_settings_canvas_config(_e=None):
                # Keep content width in sync with viewport width.
                # Do not force full layout work on simple window move events.
                current_width = int(settings_canvas.winfo_width())
                if current_width <= 0:
                    return
                if _settings_layout_state["canvas_width"] == current_width:
                    return
                _settings_layout_state["canvas_width"] = current_width
                settings_canvas.itemconfigure(settings_window, width=current_width)

            settings_content.bind("<Configure>", _on_settings_content_config)
            settings_canvas.bind("<Configure>", _on_settings_canvas_config)

            settings_wheel_bound = {"value": False}

            def _wheel_settings(event):
                # Global wheel handler while settings window is open.
                # Does not depend on hover over the scrollbar itself.
                delta = getattr(event, "delta", 0)
                if delta == 0 and getattr(event, "num", None) in (4, 5):
                    delta = 120 if event.num == 4 else -120
                if not delta:
                    return "break"
                steps = max(1, int(abs(delta) / 120))
                settings_canvas.yview_scroll(-steps if delta > 0 else steps, "units")
                return "break"

            def _bind_settings_wheel() -> None:
                if settings_wheel_bound["value"]:
                    return
                win.bind_all("<MouseWheel>", _wheel_settings, add="+")
                win.bind_all("<Button-4>", _wheel_settings, add="+")
                win.bind_all("<Button-5>", _wheel_settings, add="+")
                settings_wheel_bound["value"] = True

            def _unbind_settings_wheel() -> None:
                if not settings_wheel_bound["value"]:
                    return
                try:
                    win.unbind_all("<MouseWheel>")
                    win.unbind_all("<Button-4>")
                    win.unbind_all("<Button-5>")
                except Exception:
                    pass
                settings_wheel_bound["value"] = False

            _bind_settings_wheel()

            def _make_dark_dropdown(parent, variable: tk.StringVar, values: list[str], width_chars: int = 26):
                bg = "#f8fbff"
                fg = "#0f172a"
                border = "#cbd5e1"
                btn = tk.Menubutton(
                    parent,
                    textvariable=variable,
                    relief="solid",
                    bd=1,
                    bg=bg,
                    fg=fg,
                    activebackground="#e2e8f0",
                    activeforeground="#0f172a",
                    highlightthickness=0,
                    anchor="w",
                    direction="below",
                    padx=8,
                    pady=6,
                    font=("Segoe UI", 10),
                )
                btn.configure(width=max(12, int(width_chars)))
                menu = tk.Menu(
                    btn,
                    tearoff=0,
                    bg=bg,
                    fg=fg,
                    activebackground="#dbe7ff",
                    activeforeground="#0f172a",
                    bd=1,
                    relief="solid",
                )
                if not values:
                    values = ["(нет доступных значений)"]
                for item in values:
                    menu.add_radiobutton(label=item, variable=variable, value=item)
                btn.configure(menu=menu)
                btn.configure(highlightbackground=border)
                return btn

            def _set_dark_dropdown_values(dropdown_btn: tk.Menubutton, variable: tk.StringVar, values: list[str]) -> None:
                menu = tk.Menu(
                    dropdown_btn,
                    tearoff=0,
                    bg="#f8fbff",
                    fg="#0f172a",
                    activebackground="#dbe7ff",
                    activeforeground="#0f172a",
                    bd=1,
                    relief="solid",
                )
                options = list(values or [])
                if not options:
                    options = ["(нет доступных значений)"]
                for item in options:
                    menu.add_radiobutton(label=item, variable=variable, value=item)
                dropdown_btn.configure(menu=menu)

            api_var = tk.StringVar(value=str(config_data.get("openrouter_api_key", "")))
            model_var = tk.StringVar(value=str(config_data.get("model", "google/gemini-2.5-flash")))
            backend_mode_map = {"api": "Через API (OpenRouter)", "local": "Локально (на устройстве)"}
            backend_mode_rev = {v: k for k, v in backend_mode_map.items()}
            backend_mode_var = tk.StringVar(
                value=backend_mode_map.get(str(config_data.get("transcription_backend", "api")).strip().lower(), "Через API (OpenRouter)")
            )
            local_engine_map = {"whisper": "Whisper", "vosk": "Vosk", "sherpa": "Sherpa (онлайн-модель)"}
            local_engine_rev = {v: k for k, v in local_engine_map.items()}
            local_engine_raw = str(config_data.get("local_backend_engine", "whisper")).strip().lower()
            if local_engine_raw not in local_engine_map:
                local_engine_raw = "whisper"
            local_engine_var = tk.StringVar(value=local_engine_map.get(local_engine_raw, "Whisper"))
            local_model_map = {
                "small": "small (быстро)",
                "medium": "medium (точнее)",
                "large-v3": "large-v3 (макс. качество)",
            }
            local_model_rev = {v: k for k, v in local_model_map.items()}
            local_model_raw = str(config_data.get("local_whisper_model", "small")).strip().lower()
            if local_model_raw not in local_model_map:
                local_model_raw = "small"
            local_model_var = tk.StringVar(value=local_model_map.get(local_model_raw, "small (быстро)"))
            vosk_model_map = {
                "vosk-model-small-ru-0.22": "small-ru (быстро)",
                "vosk-model-ru-0.42": "ru-0.42 (точнее)",
            }
            vosk_model_rev = {v: k for k, v in vosk_model_map.items()}
            local_vosk_raw = str(config_data.get("local_vosk_model", "vosk-model-small-ru-0.22")).strip().lower()
            if local_vosk_raw not in vosk_model_map:
                local_vosk_raw = "vosk-model-small-ru-0.22"
            local_vosk_model_var = tk.StringVar(value=vosk_model_map.get(local_vosk_raw, "small-ru (быстро)"))
            sherpa_model_map = {
                LocalSherpaWhispeRuTranscriber.MODEL_NAME: "GigaAM RNNT (быстрый, онлайн-установка)",
            }
            sherpa_model_rev = {v: k for k, v in sherpa_model_map.items()}
            local_sherpa_raw = str(config_data.get("local_sherpa_model", LocalSherpaWhispeRuTranscriber.MODEL_NAME)).strip().lower()
            if local_sherpa_raw not in sherpa_model_map:
                local_sherpa_raw = LocalSherpaWhispeRuTranscriber.MODEL_NAME
            local_sherpa_model_var = tk.StringVar(
                value=sherpa_model_map.get(local_sherpa_raw, "GigaAM RNNT (быстрый, онлайн-установка)")
            )
            hotkey_var = tk.StringVar(value=str(config_data.get("ptt_key", "f8")))
            activation_mode_map = {"hold": "Зажми и говори", "toggle": "Свободные руки"}
            activation_mode_rev = {v: k for k, v in activation_mode_map.items()}
            activation_mode_var = tk.StringVar(value=activation_mode_map.get(str(config_data.get("activation_mode", "hold")), "Зажми и говори"))
            auto_paste_var = tk.BooleanVar(value=bool(config_data.get("auto_paste", True)))
            copy_result_var = tk.BooleanVar(value=bool(config_data.get("copy_result_to_clipboard", False)))
            restore_clip_var = tk.BooleanVar(value=bool(config_data.get("restore_clipboard", True)))
            punctuate_var = tk.BooleanVar(value=bool(config_data.get("punctuate_text", True)))
            capitalize_var = tk.BooleanVar(value=bool(config_data.get("capitalize_sentences", True)))
            terminal_period_var = tk.BooleanVar(value=bool(config_data.get("terminal_period", True)))
            convert_numbers_var = tk.BooleanVar(value=bool(config_data.get("convert_numbers", False)))
            quotes_dash_var = tk.BooleanVar(value=bool(config_data.get("normalize_quotes_dashes", True)))
            clean_text_mode_map = {"off": "Выключено", "soft": "Мягкая очистка", "strong": "Сильная очистка"}
            clean_text_mode_rev = {v: k for k, v in clean_text_mode_map.items()}
            raw_clean_mode = str(config_data.get("clean_text_mode", "soft")).strip().lower()
            if raw_clean_mode not in clean_text_mode_map:
                raw_clean_mode = "soft" if bool(config_data.get("clean_text", False)) else "off"
            clean_text_mode_var = tk.StringVar(value=clean_text_mode_map.get(raw_clean_mode, "Выключено"))
            max_text_quality_var = tk.BooleanVar(value=bool(config_data.get("max_text_quality", False)))
            whisper_medium_turbo_var = tk.BooleanVar(value=bool(config_data.get("whisper_medium_turbo", False)))
            local_fast_refine_var = tk.BooleanVar(value=bool(config_data.get("local_fast_refine", True)))
            overlay_var = tk.BooleanVar(value=bool(config_data.get("show_overlay_widget", True)))
            live_preview_var = tk.BooleanVar(value=bool(config_data.get("show_live_preview_text", True)))
            mic_quality_var = tk.BooleanVar(value=bool(config_data.get("show_mic_quality_indicator", True)))
            record_sounds_var = tk.BooleanVar(value=bool(config_data.get("play_recording_sounds", True)))
            notify_sounds_var = tk.BooleanVar(value=bool(config_data.get("play_notification_sounds", True)))
            autostart_var = tk.BooleanVar(value=bool(config_data.get("autostart_windows", False)))
            auto_update_var = tk.BooleanVar(value=bool(config_data.get("auto_download_updates", False)))
            run_as_admin_var = tk.BooleanVar(value=bool(config_data.get("run_as_admin", False)))
            hw_accel_var = tk.BooleanVar(value=bool(config_data.get("hardware_acceleration", True)))
            no_sound_notify_var = tk.BooleanVar(value=bool(config_data.get("notify_on_no_sound", True)))
            sample_rate_var = tk.StringVar(value=str(config_data.get("sample_rate", 24000)))
            meeting_summary_backend_map = {"api": "API", "local_gguf": "Локально (GGUF)"}
            meeting_summary_backend_rev = {v: k for k, v in meeting_summary_backend_map.items()}
            meeting_summary_backend_raw = str(config_data.get("meeting_summary_backend", "api")).strip().lower()
            if meeting_summary_backend_raw not in meeting_summary_backend_map:
                meeting_summary_backend_raw = "api"
            meeting_summary_backend_var = tk.StringVar(
                value=meeting_summary_backend_map.get(meeting_summary_backend_raw, "API")
            )
            gguf_model_labels = [LocalMeetingSummaryGGUF.model_title(name) for name in LocalMeetingSummaryGGUF.MODEL_CATALOG.keys()]
            gguf_model_rev = {LocalMeetingSummaryGGUF.model_title(name): name for name in LocalMeetingSummaryGGUF.MODEL_CATALOG.keys()}
            summary_model_raw = LocalMeetingSummaryGGUF.normalize_model_name(
                str(config_data.get("meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
            )
            meeting_local_summary_model_var = tk.StringVar(
                value=LocalMeetingSummaryGGUF.model_title(summary_model_raw)
            )
            meeting_system_audio_mode_map = {"auto": "Авто", "manual": "Выбрать вручную"}
            meeting_system_audio_mode_rev = {v: k for k, v in meeting_system_audio_mode_map.items()}
            meeting_system_audio_mode_raw = str(config_data.get("meeting_system_audio_mode", "auto")).strip().lower()
            if meeting_system_audio_mode_raw not in meeting_system_audio_mode_map:
                meeting_system_audio_mode_raw = "auto"
            meeting_system_audio_mode_var = tk.StringVar(
                value=meeting_system_audio_mode_map.get(meeting_system_audio_mode_raw, "Авто")
            )
            system_audio_devices_raw = list(config_data.get("_system_audio_devices") or [])
            system_audio_device_items = [entry for entry in system_audio_devices_raw if isinstance(entry, dict)]
            system_audio_label_to_id = {
                str(entry.get("label", "")).strip(): str(entry.get("id", "")).strip()
                for entry in system_audio_device_items
                if str(entry.get("label", "")).strip() and str(entry.get("id", "")).strip()
            }
            system_audio_id_to_label = {v: k for k, v in system_audio_label_to_id.items()}
            selected_system_audio_id = str(config_data.get("meeting_system_audio_device_id", "") or "").strip()
            if selected_system_audio_id and selected_system_audio_id not in system_audio_id_to_label:
                system_audio_id_to_label[selected_system_audio_id] = f"Выбранное устройство ({selected_system_audio_id})"
                system_audio_label_to_id[system_audio_id_to_label[selected_system_audio_id]] = selected_system_audio_id
            system_audio_labels = list(system_audio_label_to_id.keys())
            meeting_system_audio_device_var = tk.StringVar(
                value=system_audio_id_to_label.get(
                    selected_system_audio_id,
                    (system_audio_labels[0] if system_audio_labels else ""),
                )
            )

            mic_devices = list(config_data.get("_audio_devices") or [])
            selected_device = str(config_data.get("input_device", "")).strip()
            if selected_device and selected_device not in mic_devices:
                mic_devices.insert(0, selected_device)
            mic_device_var = tk.StringVar(value=selected_device if selected_device else (mic_devices[0] if mic_devices else ""))

            def _settings_card(row_idx: int, title: str, desc: str = "") -> tk.Frame:
                card = tk.Frame(settings_content, bg=c_surface, highlightthickness=1, highlightbackground=c_border, padx=18, pady=14)
                card.grid(row=row_idx, column=0, sticky="ew", pady=(0, 12))
                card.columnconfigure(0, weight=1)
                tk.Label(card, text=title, bg=c_surface, fg=c_text, font=("Segoe UI", 13, "bold")).grid(
                    row=0, column=0, sticky="w"
                )
                if desc:
                    tk.Label(
                        card,
                        text=desc,
                        bg=c_surface,
                        fg=c_text_muted,
                        font=("Segoe UI", 10),
                        justify="left",
                        wraplength=920,
                    ).grid(row=1, column=0, sticky="w", pady=(4, 12))
                return card

            card_api = _settings_card(0, "Распознавание и модель", "Выберите режим работы: через OpenRouter API или локально на Whisper/Vosk/Sherpa.")
            tk.Label(card_api, text="Режим распознавания", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
            backend_dropdown = _make_dark_dropdown(
                card_api,
                backend_mode_var,
                ["Через API (OpenRouter)", "Локально (на устройстве)"],
                width_chars=36,
            )
            backend_dropdown.grid(row=3, column=0, sticky="w", pady=(4, 10))
            local_engine_label = tk.Label(card_api, text="Локальный движок", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            local_engine_label.grid(row=4, column=0, sticky="w")
            local_engine_dropdown = _make_dark_dropdown(
                card_api,
                local_engine_var,
                ["Whisper", "Vosk", "Sherpa (онлайн-модель)"],
                width_chars=24,
            )
            local_engine_dropdown.grid(row=5, column=0, sticky="w", pady=(4, 10))
            whisper_model_label = tk.Label(card_api, text="Локальная модель Whisper", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            whisper_model_label.grid(row=6, column=0, sticky="w")
            local_model_dropdown = _make_dark_dropdown(
                card_api,
                local_model_var,
                ["small (быстро)", "medium (точнее)", "large-v3 (макс. качество)"],
                width_chars=30,
            )
            local_model_dropdown.grid(row=7, column=0, sticky="w", pady=(4, 10))
            vosk_model_label = tk.Label(card_api, text="Локальная модель Vosk", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            vosk_model_label.grid(row=8, column=0, sticky="w")
            local_vosk_dropdown = _make_dark_dropdown(
                card_api,
                local_vosk_model_var,
                ["small-ru (быстро)", "ru-0.42 (точнее)"],
                width_chars=30,
            )
            local_vosk_dropdown.grid(row=9, column=0, sticky="w", pady=(4, 10))
            sherpa_model_label = tk.Label(card_api, text="Локальная модель Sherpa", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            sherpa_model_label.grid(row=10, column=0, sticky="w")
            local_sherpa_dropdown = _make_dark_dropdown(
                card_api,
                local_sherpa_model_var,
                ["GigaAM RNNT (быстрый, онлайн-установка)"],
                width_chars=36,
            )
            local_sherpa_dropdown.grid(row=11, column=0, sticky="w", pady=(4, 10))

            local_model_actions = tk.Frame(card_api, bg=c_surface)
            local_model_actions.grid(row=12, column=0, sticky="w", pady=(0, 10))
            model_download_status_var = tk.StringVar(value="Состояние модели: ожидание")
            model_download_bytes_var = tk.StringVar(value="0% В· 0.0 / 0.0 MB")
            model_download_progress_var = tk.DoubleVar(value=0.0)
            model_download_poll_job = {"id": None}

            download_model_btn = ttk.Button(local_model_actions, text="Скачать модель", style="Settings.TButton")
            download_model_btn.grid(row=0, column=0, padx=(0, 8))
            delete_model_btn = ttk.Button(local_model_actions, text="Удалить модель", style="Settings.TButton")
            delete_model_btn.grid(row=0, column=1)

            def _format_mb(bytes_value: int) -> float:
                try:
                    return round(max(0, int(bytes_value)) / (1024.0 * 1024.0), 1)
                except Exception:
                    return 0.0

            def _local_model_title(engine_name: str, model_name: str) -> str:
                normalized_engine = str(engine_name or "whisper").strip().lower()
                normalized_model = str(model_name or "").strip().lower()
                if normalized_engine == "sherpa":
                    return "Voice PRO AI GigaAM RNNT"
                if normalized_engine == "vosk":
                    return vosk_model_map.get(normalized_model, normalized_model or "Vosk")
                if normalized_engine == "whisper":
                    return local_model_map.get(normalized_model, normalized_model or "Whisper")
                return normalized_model or "модель"

            def _apply_model_download_status_ui(status: dict) -> None:
                state = str(status.get("state", "idle") or "idle").strip().lower()
                engine = str(status.get("engine", "whisper") or "whisper").strip().lower()
                model = str(status.get("model", "") or "").strip()
                message = str(status.get("message", "") or "").strip()
                progress = float(status.get("progress_percent", 0.0) or 0.0)
                progress = min(100.0, max(0.0, progress))
                downloaded = int(status.get("downloaded_bytes", 0) or 0)
                total = int(status.get("total_bytes", 0) or 0)
                err = str(status.get("error", "") or "").strip()
                installed_models = dict(status.get("installed_models") or {})
                installed_sizes = dict(status.get("installed_sizes") or {})
                selected_engine = local_engine_rev.get(local_engine_var.get().strip(), "whisper")
                selected_model = (
                    vosk_model_rev.get(local_vosk_model_var.get().strip(), LocalVoskTranscriber.MODEL_NAME)
                    if selected_engine == "vosk"
                    else sherpa_model_rev.get(local_sherpa_model_var.get().strip(), LocalSherpaWhispeRuTranscriber.MODEL_NAME)
                    if selected_engine == "sherpa"
                    else local_model_rev.get(local_model_var.get().strip(), "small")
                )
                selected_model_title = _local_model_title(selected_engine, selected_model)
                current_model_title = _local_model_title(engine, model)
                selected_installed = bool(installed_models.get(selected_model, False))
                selected_size = int(installed_sizes.get(selected_model, 0) or 0)

                if state == "preparing":
                    model_download_status_var.set(f"Состояние модели: подготовка ({current_model_title})")
                elif state == "downloading":
                    model_download_status_var.set(f"Состояние модели: скачивание ({current_model_title})")
                elif state == "error" and model == selected_model and engine == selected_engine:
                    model_download_status_var.set(f"Состояние модели: ошибка ({current_model_title})")
                elif selected_installed:
                    model_download_status_var.set(f"Состояние модели: загружена ({selected_model_title})")
                    progress = 100.0
                    downloaded = max(downloaded, selected_size)
                    total = max(total, selected_size, downloaded)
                else:
                    model_download_status_var.set(f"Состояние модели: не загружена ({selected_model_title})")
                    if state not in {"preparing", "downloading"}:
                        progress = 0.0
                        downloaded = selected_size
                        total = max(selected_size, 0)

                if message:
                    if state in {"preparing", "downloading"} or (model == selected_model and engine == selected_engine):
                        settings_runtime_message_var.set(message if not err else f"{message}. {err}")

                if total > 0:
                    model_download_bytes_var.set(
                        f"{progress:.0f}% В· {_format_mb(downloaded):.1f} / {_format_mb(total):.1f} MB"
                    )
                else:
                    model_download_bytes_var.set(f"{progress:.0f}% В· {_format_mb(downloaded):.1f} / 0.0 MB")
                model_download_progress_var.set(progress)

                busy_same_model = state in {"preparing", "downloading"} and model == selected_model and engine == selected_engine
                busy_any = state in {"preparing", "downloading"}
                download_model_btn.configure(state=("disabled" if busy_any else "normal"))
                delete_model_btn.configure(state=("disabled" if busy_same_model else "normal"))

            def _poll_model_download_status() -> None:
                model_download_poll_job["id"] = None
                if not callable(on_get_local_model_download_status_cb):
                    return
                try:
                    selected_engine = local_engine_rev.get(local_engine_var.get().strip(), "whisper")
                    status = on_get_local_model_download_status_cb(selected_engine)
                    if isinstance(status, dict):
                        _apply_model_download_status_ui(status)
                except Exception as err:
                    model_download_status_var.set("Состояние модели: ошибка")
                    model_download_bytes_var.set("0% В· 0.0 / 0.0 MB")
                    settings_runtime_message_var.set(f"Ошибка чтения статуса загрузки: {err}")
                finally:
                    if win.winfo_exists():
                        model_download_poll_job["id"] = win.after(600, _poll_model_download_status)

            def _restart_model_download_poll() -> None:
                poll_job = model_download_poll_job.get("id")
                if poll_job:
                    try:
                        win.after_cancel(poll_job)
                    except Exception:
                        pass
                model_download_poll_job["id"] = None
                _poll_model_download_status()

            def _start_model_download_from_ui() -> None:
                if not callable(on_start_local_model_download_cb):
                    status_var.set("Обработчик недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                selected_engine = local_engine_rev.get(local_engine_var.get().strip(), "whisper")
                selected = (
                    vosk_model_rev.get(local_vosk_model_var.get().strip(), LocalVoskTranscriber.MODEL_NAME)
                    if selected_engine == "vosk"
                    else sherpa_model_rev.get(local_sherpa_model_var.get().strip(), LocalSherpaWhispeRuTranscriber.MODEL_NAME)
                    if selected_engine == "sherpa"
                    else local_model_rev.get(local_model_var.get().strip(), "small")
                )
                try:
                    ok, message = on_start_local_model_download_cb(selected_engine, selected)
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                _restart_model_download_poll()

            def _delete_model_from_ui() -> None:
                if not callable(on_delete_local_model_cb):
                    status_var.set("Обработчик недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                selected_engine = local_engine_rev.get(local_engine_var.get().strip(), "whisper")
                selected = (
                    vosk_model_rev.get(local_vosk_model_var.get().strip(), LocalVoskTranscriber.MODEL_NAME)
                    if selected_engine == "vosk"
                    else sherpa_model_rev.get(local_sherpa_model_var.get().strip(), LocalSherpaWhispeRuTranscriber.MODEL_NAME)
                    if selected_engine == "sherpa"
                    else local_model_rev.get(local_model_var.get().strip(), "small")
                )

                def _worker():
                    try:
                        ok, message, snapshot = on_delete_local_model_cb(selected_engine, selected)
                    except Exception as err:
                        ok, message, snapshot = False, f"Ошибка: {err}", {}

                    def _apply():
                        status_var.set(message)
                        status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                        if ok and isinstance(snapshot, dict) and snapshot:
                            _apply_settings_snapshot_to_vars(dict(snapshot))
                        _restart_model_download_poll()

                    if win.winfo_exists():
                        win.after(0, _apply)

                threading.Thread(target=_worker, daemon=True).start()

            download_model_btn.configure(command=_start_model_download_from_ui)
            delete_model_btn.configure(command=_delete_model_from_ui)

            model_progress = ttk.Progressbar(
                card_api,
                variable=model_download_progress_var,
                maximum=100.0,
                mode="determinate",
                style="Accent.Horizontal.TProgressbar",
            )
            model_progress.grid(row=13, column=0, sticky="ew", pady=(2, 2))
            model_download_status_label = tk.Label(card_api, textvariable=model_download_status_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 9))
            model_download_status_label.grid(row=14, column=0, sticky="w", pady=(0, 1))
            model_download_bytes_label = tk.Label(card_api, textvariable=model_download_bytes_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 9))
            model_download_bytes_label.grid(row=15, column=0, sticky="w", pady=(0, 8))

            api_key_label = tk.Label(card_api, text="OpenRouter API-ключ", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            api_key_label.grid(row=16, column=0, sticky="w")
            api_row = tk.Frame(card_api, bg=c_surface)
            api_row.grid(row=17, column=0, sticky="ew", pady=(4, 12))
            api_row.columnconfigure(0, weight=1)
            api_entry = ttk.Entry(api_row, textvariable=api_var, show="*", width=46, style="Settings.TEntry")
            api_entry.grid(row=0, column=0, sticky="ew")
            api_model_label = tk.Label(card_api, text="Модель API", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold"))
            api_model_label.grid(row=18, column=0, sticky="w")
            model_entry = ttk.Entry(card_api, textvariable=model_var, width=46, style="Settings.TEntry")
            model_entry.grid(row=19, column=0, sticky="ew", pady=(4, 10))

            local_model_widgets = [
                whisper_model_label,
                local_model_dropdown,
                vosk_model_label,
                local_vosk_dropdown,
                sherpa_model_label,
                local_sherpa_dropdown,
            ]
            local_section_widgets = [
                local_engine_label,
                local_engine_dropdown,
                local_model_actions,
                model_progress,
                model_download_status_label,
                model_download_bytes_label,
            ] + local_model_widgets
            api_section_widgets = [
                api_key_label,
                api_row,
                api_model_label,
                model_entry,
            ]

            def _show_widgets(widgets: list[tk.Widget]) -> None:
                for widget in widgets:
                    try:
                        widget.grid()
                    except Exception:
                        pass

            def _hide_widgets(widgets: list[tk.Widget]) -> None:
                for widget in widgets:
                    try:
                        widget.grid_remove()
                    except Exception:
                        pass

            def _sync_local_engine_controls() -> None:
                backend = backend_mode_rev.get(backend_mode_var.get().strip(), "api")
                if backend != "local":
                    _hide_widgets(local_model_widgets)
                    return
                selected_engine = local_engine_rev.get(local_engine_var.get().strip(), "whisper")
                _hide_widgets(local_model_widgets)
                if selected_engine == "vosk":
                    _show_widgets([vosk_model_label, local_vosk_dropdown])
                elif selected_engine == "sherpa":
                    _show_widgets([sherpa_model_label, local_sherpa_dropdown])
                else:
                    _show_widgets([whisper_model_label, local_model_dropdown])

            def _sync_backend_visibility() -> None:
                backend = backend_mode_rev.get(backend_mode_var.get().strip(), "api")
                if backend == "local":
                    _show_widgets(local_section_widgets)
                    _hide_widgets(api_section_widgets)
                    _sync_local_engine_controls()
                else:
                    _hide_widgets(local_section_widgets)
                    _show_widgets(api_section_widgets)

            local_model_var.trace_add("write", lambda *_: _restart_model_download_poll())
            local_vosk_model_var.trace_add("write", lambda *_: _restart_model_download_poll())
            local_sherpa_model_var.trace_add("write", lambda *_: _restart_model_download_poll())
            local_engine_var.trace_add("write", lambda *_: (_sync_local_engine_controls(), _restart_model_download_poll()))
            backend_mode_var.trace_add("write", lambda *_: (_sync_backend_visibility(), _restart_model_download_poll()))
            _sync_backend_visibility()
            tk.Label(card_api, text="Частота дискретизации (Гц)", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=20, column=0, sticky="w"
            )
            ttk.Entry(card_api, textvariable=sample_rate_var, width=18, style="Settings.TEntry").grid(row=21, column=0, sticky="w", pady=(4, 2))
            settings_runtime_state_var = tk.StringVar(value="")
            settings_runtime_message_var = tk.StringVar(value="")
            tk.Label(card_api, textvariable=settings_runtime_state_var, bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=22, column=0, sticky="w", pady=(8, 2)
            )
            tk.Label(
                card_api,
                textvariable=settings_runtime_message_var,
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=920,
            ).grid(row=23, column=0, sticky="w", pady=(0, 2))

            card_mic = _settings_card(1, "Микрофон", "Выберите устройство и проверьте его работу перед началом диктовки.")
            tk.Label(card_mic, text="Устройство ввода", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
            mic_dropdown = _make_dark_dropdown(card_mic, mic_device_var, mic_devices, width_chars=80)
            mic_dropdown.grid(row=3, column=0, sticky="ew", pady=(4, 10))

            mic_buttons = tk.Frame(card_mic, bg=c_surface)
            mic_buttons.grid(row=4, column=0, sticky="w")

            def _open_windows_recording_settings() -> None:
                if not callable(on_open_windows_sound_cb):
                    status_var.set("Обработчик недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message = on_open_windows_sound_cb()
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")

            def _test_microphone() -> None:
                if not callable(on_test_microphone_cb):
                    status_var.set("Обработчик проверки микрофона недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return

                status_var.set("Проверка микрофона...")
                status_lbl.configure(foreground="#cbd5e1")

                def _worker():
                    try:
                        ok, message = on_test_microphone_cb(str(mic_device_var.get()).strip())
                    except Exception as err:
                        ok, message = False, f"Ошибка: {err}"
                    win.after(
                        0,
                        lambda: (
                            status_var.set(message),
                            status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239"),
                        ),
                    )

                threading.Thread(target=_worker, daemon=True).start()

            ttk.Button(mic_buttons, text="Открыть настройки записи", command=_open_windows_recording_settings, style="Settings.TButton").grid(
                row=0, column=0, padx=(0, 8)
            )
            ttk.Button(mic_buttons, text="Начать тест", command=_test_microphone, style="Settings.TButton").grid(row=0, column=1)

            card_hotkey = _settings_card(2, "Управление записью", "Настройте горячую клавишу и режим запуска записи под свой рабочий сценарий.")
            tk.Label(card_hotkey, text="Горячая клавиша", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky="w")
            ttk.Entry(card_hotkey, textvariable=hotkey_var, width=26, state="readonly", style="Settings.TEntry").grid(
                row=3, column=0, sticky="w", pady=(4, 10)
            )

            hotkey_btns = tk.Frame(card_hotkey, bg=c_surface)
            hotkey_btns.grid(row=4, column=0, sticky="w")
            tk.Label(card_hotkey, text="Режим активации", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(row=5, column=0, sticky="w", pady=(12, 4))
            activation_dropdown = _make_dark_dropdown(card_hotkey, activation_mode_var, ["Зажми и говори", "Свободные руки"], width_chars=24)
            activation_dropdown.grid(row=6, column=0, sticky="w")
            tk.Label(
                card_hotkey,
                text="«Зажми и говори» — удерживайте клавишу; «Свободные руки» — нажмите для старта и ещё раз для остановки.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 10),
                justify="left",
                wraplength=900,
            ).grid(row=7, column=0, sticky="w", pady=(6, 2))

            card_behavior = _settings_card(3, "Поведение вставки", "Определяет, как результат распознавания вставляется в активное поле и буфер обмена.")
            ttk.Checkbutton(card_behavior, text="Автовставка после распознавания", variable=auto_paste_var, style="Card.TCheckbutton").grid(
                row=2, column=0, sticky="w", pady=(2, 6)
            )
            ttk.Checkbutton(card_behavior, text="Сохранять текст в буфер обмена", variable=copy_result_var, style="Card.TCheckbutton").grid(
                row=3, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(
                card_behavior,
                text="Восстанавливать буфер обмена после вставки",
                variable=restore_clip_var,
                style="Card.TCheckbutton",
            ).grid(row=4, column=0, sticky="w", pady=(0, 2))

            card_text = _settings_card(4, "Обработка текста", "Автоматические правила форматирования для более чистого и читаемого результата.")
            tk.Label(card_text, text="Общие настройки (API и локально)", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=2, column=0, sticky="w", pady=(2, 4)
            )
            ttk.Checkbutton(card_text, text="Знаки препинания", variable=punctuate_var, style="Card.TCheckbutton").grid(row=3, column=0, sticky="w", pady=(2, 6))
            ttk.Checkbutton(card_text, text="Заглавная буква в начале предложений", variable=capitalize_var, style="Card.TCheckbutton").grid(
                row=4, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_text, text="Точка в конце", variable=terminal_period_var, style="Card.TCheckbutton").grid(row=5, column=0, sticky="w", pady=(0, 6))
            ttk.Checkbutton(card_text, text="Преобразование чисел в цифры", variable=convert_numbers_var, style="Card.TCheckbutton").grid(
                row=6, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_text, text="Кавычки «» и нормализация тире", variable=quotes_dash_var, style="Card.TCheckbutton").grid(
                row=7, column=0, sticky="w", pady=(0, 2)
            )
            tk.Label(card_text, text="Режим чистого текста", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=8, column=0, sticky="w", pady=(6, 2)
            )
            clean_text_mode_dropdown = _make_dark_dropdown(
                card_text,
                clean_text_mode_var,
                ["Выключено", "Мягкая очистка", "Сильная очистка"],
                width_chars=30,
            )
            clean_text_mode_dropdown.grid(row=9, column=0, sticky="w", pady=(2, 6))
            tk.Label(
                card_text,
                text="Эти параметры применяются в обоих режимах распознавания.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=920,
            ).grid(row=10, column=0, sticky="w", pady=(0, 8))

            tk.Label(card_text, text="Только локально", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=11, column=0, sticky="w", pady=(2, 4)
            )
            ttk.Checkbutton(
                card_text,
                text="Turbo для Whisper medium (ускоренная обработка 3–5 сек)",
                variable=whisper_medium_turbo_var,
                style="Card.TCheckbutton",
            ).grid(row=12, column=0, sticky="w", pady=(0, 4))
            ttk.Checkbutton(
                card_text,
                text="Быстрая нейро-коррекция после локальной модели (таймаут ~6с)",
                variable=local_fast_refine_var,
                style="Card.TCheckbutton",
            ).grid(row=13, column=0, sticky="w", pady=(0, 4))
            tk.Label(
                card_text,
                text="Эти параметры влияют только на локальные движки Whisper/Vosk/Sherpa.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=920,
            ).grid(row=14, column=0, sticky="w", pady=(0, 8))

            tk.Label(card_text, text="Только API", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=15, column=0, sticky="w", pady=(2, 4)
            )
            ttk.Checkbutton(
                card_text,
                text="Максимально корректный текст (усиленная правка)",
                variable=max_text_quality_var,
                style="Card.TCheckbutton",
            ).grid(row=16, column=0, sticky="w", pady=(0, 4))
            tk.Label(
                card_text,
                text="Усиленная правка применяется для финальной API-коррекции текста.",
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=920,
            ).grid(row=17, column=0, sticky="w", pady=(0, 2))

            card_ui = _settings_card(5, "Интерфейс и уведомления", "Визуальные индикаторы и звуки, которые помогают контролировать процесс диктовки.")
            ttk.Checkbutton(card_ui, text="Отображение виджета (слушаю/распознаю)", variable=overlay_var, style="Card.TCheckbutton").grid(
                row=2, column=0, sticky="w", pady=(2, 6)
            )
            ttk.Checkbutton(card_ui, text="Показывать живой текст во время записи", variable=live_preview_var, style="Card.TCheckbutton").grid(
                row=3, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_ui, text="Индикатор качества записи (уровень/тихо/перегруз)", variable=mic_quality_var, style="Card.TCheckbutton").grid(
                row=4, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_ui, text="Звуковые сигналы записи", variable=record_sounds_var, style="Card.TCheckbutton").grid(
                row=5, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_ui, text="Звуки уведомлений", variable=notify_sounds_var, style="Card.TCheckbutton").grid(
                row=6, column=0, sticky="w", pady=(0, 6)
            )
            ttk.Checkbutton(card_ui, text="Уведомлять при отсутствии звука", variable=no_sound_notify_var, style="Card.TCheckbutton").grid(
                row=7, column=0, sticky="w", pady=(0, 2)
            )

            card_system = _settings_card(6, "Система", "Системные параметры запуска и производительности приложения.")
            ttk.Checkbutton(card_system, text="Автозагрузка Windows", variable=autostart_var, style="Card.TCheckbutton").grid(
                row=2, column=0, sticky="w", pady=(2, 6)
            )
            ttk.Checkbutton(card_system, text="Автоматическое скачивание обновлений", variable=auto_update_var, style="Card.TCheckbutton").grid(
                row=3, column=0, sticky="w", pady=(0, 6)
            )
            update_status_var = tk.StringVar(value="")
            update_status_lbl = tk.Label(
                card_system,
                textvariable=update_status_var,
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
                justify="left",
                wraplength=920,
            )
            update_status_lbl.grid(row=4, column=0, sticky="w", pady=(0, 8))
            updater_poll_job = {"id": None}
            update_download_wrap = tk.Frame(card_system, bg=c_surface)
            update_download_wrap.grid(row=40, column=0, sticky="ew", pady=(0, 8))
            update_download_wrap.grid_remove()
            update_download_var = tk.DoubleVar(value=0.0)
            update_download_progress = ttk.Progressbar(
                update_download_wrap,
                orient="horizontal",
                mode="determinate",
                variable=update_download_var,
                maximum=100.0,
                style="Accent.Horizontal.TProgressbar",
            )
            update_download_progress.grid(row=0, column=0, sticky="ew")
            update_download_meta_var = tk.StringVar(value="")
            tk.Label(
                update_download_wrap,
                textvariable=update_download_meta_var,
                bg=c_surface,
                fg=c_text_muted,
                font=("Segoe UI", 9),
            ).grid(row=1, column=0, sticky="w", pady=(4, 0))
            update_download_wrap.columnconfigure(0, weight=1)

            def _fmt_bytes_short(value: int) -> str:
                try:
                    v = max(0, int(value))
                except Exception:
                    v = 0
                if v >= 1024 * 1024 * 1024:
                    return f"{v / (1024.0 * 1024.0 * 1024.0):.2f} GB"
                return f"{v / (1024.0 * 1024.0):.1f} MB"

            def _sync_update_download_ui(updater_state: dict) -> None:
                st = str(updater_state.get("state", "idle") or "idle").strip().lower()
                if st != "downloading":
                    update_download_wrap.grid_remove()
                    return
                done = int(updater_state.get("downloaded_bytes", 0) or 0)
                total = int(updater_state.get("total_bytes", 0) or 0)
                ver = str(updater_state.get("available_version", "") or "").strip()
                if total > 0:
                    pct = max(0.0, min(100.0, (float(done) / float(total)) * 100.0))
                    left = max(0, total - done)
                    update_download_var.set(pct)
                    update_download_meta_var.set(
                        f"Версия {ver or '?'} · {pct:.1f}% · {_fmt_bytes_short(done)} / {_fmt_bytes_short(total)} · осталось {_fmt_bytes_short(left)}"
                    )
                else:
                    update_download_var.set(0.0)
                    update_download_meta_var.set(f"Версия {ver or '?'} · скачивание…")
                update_download_wrap.grid()
            ttk.Checkbutton(
                card_system,
                text="Всегда запускать от имени администратора",
                variable=run_as_admin_var,
                style="Card.TCheckbutton",
            ).grid(row=5, column=0, sticky="w", pady=(0, 6))
            ttk.Checkbutton(card_system, text="Аппаратное ускорение (флаг)", variable=hw_accel_var, style="Card.TCheckbutton").grid(
                row=6, column=0, sticky="w", pady=(0, 6)
            )

            sys_btns = tk.Frame(card_system, bg=c_surface)
            sys_btns.grid(row=7, column=0, sticky="w", pady=(6, 2))

            def _open_logs() -> None:
                if not callable(on_open_logs_cb):
                    status_var.set("Обработчик логов недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message = on_open_logs_cb()
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")

            def _render_update_status(snapshot: dict | None = None) -> None:
                snap = dict(snapshot or {})
                updater_state = dict(snap.get("_updater_status") or {})
                _sync_update_download_ui(updater_state)
                state = str(updater_state.get("state", "idle") or "idle").strip().lower()
                current = str(snap.get("_app_version", APP_VERSION) or APP_VERSION).strip()
                available = str(updater_state.get("available_version", "") or "").strip()
                if state == "checking":
                    update_status_var.set(f"Обновления: проверка… (текущая {current})")
                    update_status_lbl.configure(fg="#1d4ed8")
                    return
                if state == "available" and available:
                    update_status_var.set(f"Обновления: доступна версия {available} (у вас {current})")
                    update_status_lbl.configure(fg="#0f766e")
                    return
                if state == "downloading":
                    update_status_var.set(f"Обновления: скачивание версии {available or '?'}…")
                    update_status_lbl.configure(fg="#1d4ed8")
                    return
                if state == "verifying":
                    update_status_var.set("Обновления: проверка целостности (SHA-256)…")
                    update_status_lbl.configure(fg="#1d4ed8")
                    return
                if state == "error":
                    msg = str(updater_state.get("message", "") or "").strip()
                    update_status_var.set(f"Обновления: ошибка. {msg}" if msg else "Обновления: ошибка проверки.")
                    update_status_lbl.configure(fg="#9f1239")
                    return
                if state == "up_to_date":
                    update_status_var.set(f"Обновления: актуальная версия {current}")
                    update_status_lbl.configure(fg=c_text_muted)
                    return
                update_status_var.set(f"Обновления: текущая версия {current}")
                update_status_lbl.configure(fg=c_text_muted)

            def _check_updates_ui() -> None:
                if not callable(on_check_updates_cb):
                    status_var.set("Проверка обновлений недоступна")
                    status_lbl.configure(foreground="#9f1239")
                    return

                update_status_var.set("Обновления: проверка…")
                update_status_lbl.configure(fg="#1d4ed8")

                def _worker() -> None:
                    try:
                        result = on_check_updates_cb()
                    except Exception as err:
                        result = (False, f"Ошибка: {err}", {})
                    ok = False
                    message = ""
                    snapshot = {}
                    update_info = None
                    if isinstance(result, tuple):
                        if len(result) > 0:
                            ok = bool(result[0])
                        if len(result) > 1:
                            message = str(result[1] or "")
                        if len(result) > 2 and isinstance(result[2], dict):
                            snapshot = dict(result[2] or {})
                        if len(result) > 3 and isinstance(result[3], dict):
                            update_info = dict(result[3] or {})

                    def _finish() -> None:
                        if message:
                            status_var.set(message)
                            status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                        if snapshot:
                            _apply_settings_snapshot_to_vars(snapshot)
                            _render_update_status(snapshot)
                        if update_info and callable(on_start_update_install_cb):
                            version = str(update_info.get("version", "") or "").strip()
                            notes = str(update_info.get("release_notes", "") or "").strip()
                            details = f"Доступна версия {version}.\n\n{notes}" if notes else f"Доступна версия {version}."
                            if messagebox.askyesno("Доступно обновление", details + "\n\nУстановить сейчас?"):
                                installer_url = str(update_info.get("installer_url", "") or "").strip()
                                expected_sha = str(update_info.get("sha256", "") or "").strip().lower()
                                published_at = str(update_info.get("published_at", "") or "").strip()
                                source_host = ""
                                try:
                                    from urllib.parse import urlparse
                                    source_host = str(urlparse(installer_url).netloc or "").strip()
                                except Exception:
                                    source_host = ""

                                verify_lines = [
                                    "Проверка подлинности обновления:",
                                    f"• Версия: {version or '?'}",
                                    f"• Источник: {source_host or installer_url or 'неизвестно'}",
                                    f"• SHA-256: {expected_sha or 'не указан'}",
                                ]
                                if published_at:
                                    verify_lines.append(f"• Дата релиза: {published_at}")
                                verify_lines.append("")
                                verify_lines.append("Продолжить скачивание и установку?")
                                if not messagebox.askyesno(
                                    "Проверка подлинности",
                                    "\n".join(verify_lines),
                                ):
                                    update_status_var.set(f"Обновления: установка версии {version or '?'} отменена пользователем")
                                    update_status_lbl.configure(fg=c_text_muted)
                                    return

                                update_status_var.set(f"Обновления: скачивание версии {version}…")
                                update_status_lbl.configure(fg="#1d4ed8")
                                _restart_updater_poll()

                                def _install_worker() -> None:
                                    try:
                                        ires = on_start_update_install_cb(dict(update_info))
                                    except Exception as err:
                                        ires = (False, f"Ошибка: {err}", {})
                                    iok, imsg, isnap = False, "", {}
                                    if isinstance(ires, tuple):
                                        if len(ires) > 0:
                                            iok = bool(ires[0])
                                        if len(ires) > 1:
                                            imsg = str(ires[1] or "")
                                        if len(ires) > 2 and isinstance(ires[2], dict):
                                            isnap = dict(ires[2] or {})

                                    def _finish_install() -> None:
                                        if imsg:
                                            status_var.set(imsg)
                                            status_lbl.configure(foreground="#2b7a0b" if iok else "#9f1239")
                                        if isnap:
                                            _apply_settings_snapshot_to_vars(isnap)
                                            _render_update_status(isnap)

                                    if win.winfo_exists():
                                        win.after(0, _finish_install)

                                threading.Thread(target=_install_worker, daemon=True).start()

                    if win.winfo_exists():
                        win.after(0, _finish)

                threading.Thread(target=_worker, daemon=True).start()

            def _poll_updater_status() -> None:
                updater_poll_job["id"] = None
                if not callable(on_get_updater_status_cb):
                    return
                try:
                    state = on_get_updater_status_cb()
                    if isinstance(state, dict):
                        _render_update_status({"_updater_status": dict(state), "_app_version": APP_VERSION})
                except Exception:
                    pass
                if win.winfo_exists():
                    updater_poll_job["id"] = win.after(1200, _poll_updater_status)

            def _restart_updater_poll() -> None:
                job = updater_poll_job.get("id")
                if job:
                    try:
                        win.after_cancel(job)
                    except Exception:
                        pass
                updater_poll_job["id"] = None
                _poll_updater_status()

            ttk.Button(sys_btns, text="Открыть логи", command=_open_logs, style="Settings.TButton").grid(row=0, column=0, padx=(0, 8))
            ttk.Button(sys_btns, text="Проверить обновления", command=_check_updates_ui, style="Settings.TButton").grid(row=0, column=1, padx=(0, 8))

            card_meeting_audio = _settings_card(7, "Системный звук (Voice Meads AI)", "Выберите источник системного канала для транскрибации онлайн-встреч.")
            tk.Label(card_meeting_audio, text="Режим системного звука", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=2, column=0, sticky="w"
            )
            meeting_system_audio_mode_dropdown = _make_dark_dropdown(
                card_meeting_audio,
                meeting_system_audio_mode_var,
                ["Авто", "Выбрать вручную"],
                width_chars=20,
            )
            meeting_system_audio_mode_dropdown.grid(row=3, column=0, sticky="w", pady=(4, 10))
            tk.Label(card_meeting_audio, text="Устройство системного звука", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=4, column=0, sticky="w"
            )
            meeting_system_audio_device_dropdown = _make_dark_dropdown(
                card_meeting_audio,
                meeting_system_audio_device_var,
                system_audio_labels,
                width_chars=62,
            )
            meeting_system_audio_device_dropdown.grid(row=5, column=0, sticky="w", pady=(4, 8))
            meeting_system_audio_status_var = tk.StringVar(value="")
            tk.Label(card_meeting_audio, textvariable=meeting_system_audio_status_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 9)).grid(
                row=6, column=0, sticky="w", pady=(0, 8)
            )
            meeting_system_audio_btns = tk.Frame(card_meeting_audio, bg=c_surface)
            meeting_system_audio_btns.grid(row=7, column=0, sticky="w")

            def _sync_meeting_system_audio_controls() -> None:
                manual = (meeting_system_audio_mode_rev.get(meeting_system_audio_mode_var.get().strip(), "auto") == "manual")
                meeting_system_audio_device_dropdown.configure(state=("normal" if manual else "disabled"))
                if manual:
                    meeting_system_audio_status_var.set("Ручной режим: будет использоваться только выбранное устройство.")
                else:
                    meeting_system_audio_status_var.set("Авто-режим: приложение само подбирает лучший backend системного звука.")

            def _refresh_system_audio_devices_ui() -> None:
                nonlocal system_audio_label_to_id, system_audio_id_to_label, system_audio_labels
                if callable(on_get_system_audio_devices_cb):
                    try:
                        fresh = on_get_system_audio_devices_cb()
                    except Exception:
                        fresh = []
                    device_entries = [entry for entry in list(fresh or []) if isinstance(entry, dict)]
                    if device_entries:
                        system_audio_label_to_id = {
                            str(entry.get("label", "")).strip(): str(entry.get("id", "")).strip()
                            for entry in device_entries
                            if str(entry.get("label", "")).strip() and str(entry.get("id", "")).strip()
                        }
                        system_audio_id_to_label = {v: k for k, v in system_audio_label_to_id.items()}
                        system_audio_labels = list(system_audio_label_to_id.keys())
                        _set_dark_dropdown_values(meeting_system_audio_device_dropdown, meeting_system_audio_device_var, system_audio_labels)
                        current_label = meeting_system_audio_device_var.get().strip()
                        current_id = system_audio_label_to_id.get(current_label, "")
                        if current_id and current_id in system_audio_id_to_label:
                            meeting_system_audio_device_var.set(system_audio_id_to_label[current_id])
                        elif system_audio_labels:
                            meeting_system_audio_device_var.set(system_audio_labels[0])
                _sync_meeting_system_audio_controls()

            ttk.Button(meeting_system_audio_btns, text="Обновить список устройств", command=_refresh_system_audio_devices_ui, style="Settings.TButton").grid(row=0, column=0, padx=(0, 8))
            meeting_system_audio_mode_var.trace_add("write", lambda *_args: _sync_meeting_system_audio_controls())
            _sync_meeting_system_audio_controls()

            card_meeting_summary = _settings_card(8, "Резюме встреч (Voice Meads AI)", "Настройте источник генерации резюме для встреч.")
            tk.Label(card_meeting_summary, text="Источник резюме", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=2, column=0, sticky="w"
            )
            meeting_summary_backend_dropdown = _make_dark_dropdown(
                card_meeting_summary,
                meeting_summary_backend_var,
                ["API", "Локально (GGUF)"],
                width_chars=24,
            )
            meeting_summary_backend_dropdown.grid(row=3, column=0, sticky="w", pady=(4, 10))
            tk.Label(card_meeting_summary, text="Локальная модель GGUF", bg=c_surface, fg=c_text, font=("Segoe UI", 10, "bold")).grid(
                row=4, column=0, sticky="w"
            )
            local_summary_model_dropdown = _make_dark_dropdown(
                card_meeting_summary,
                meeting_local_summary_model_var,
                gguf_model_labels,
                width_chars=44,
            )
            local_summary_model_dropdown.grid(row=5, column=0, sticky="w", pady=(4, 8))
            summary_model_status_var = tk.StringVar(value="Состояние модели резюме: ожидание")
            tk.Label(card_meeting_summary, textvariable=summary_model_status_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 9)).grid(
                row=6, column=0, sticky="w", pady=(0, 4)
            )
            summary_model_progress_var = tk.DoubleVar(value=0.0)
            summary_model_progress = ttk.Progressbar(
                card_meeting_summary,
                orient="horizontal",
                mode="determinate",
                variable=summary_model_progress_var,
                maximum=100.0,
                style="Accent.Horizontal.TProgressbar",
            )
            summary_model_progress.grid(row=7, column=0, sticky="ew", pady=(0, 8))
            summary_model_stats_var = tk.StringVar(value="")
            tk.Label(card_meeting_summary, textvariable=summary_model_stats_var, bg=c_surface, fg=c_text_muted, font=("Segoe UI", 9)).grid(
                row=8, column=0, sticky="w", pady=(0, 8)
            )
            summary_btns = tk.Frame(card_meeting_summary, bg=c_surface)
            summary_btns.grid(row=9, column=0, sticky="w")
            summary_model_poll_job = {"id": None}

            def _selected_summary_model_key() -> str:
                return str(gguf_model_rev.get(meeting_local_summary_model_var.get().strip(), LocalMeetingSummaryGGUF.DEFAULT_MODEL))

            def _apply_summary_model_status_ui(status: dict) -> None:
                st = str(status.get("state", "idle")).strip().lower()
                key = str(status.get("model", "")).strip().lower() or _selected_summary_model_key()
                title = LocalMeetingSummaryGGUF.model_title(key)
                progress = float(status.get("progress_percent", 0.0) or 0.0)
                downloaded = int(status.get("downloaded_bytes", 0) or 0)
                total = int(status.get("total_bytes", 0) or 0)
                message = str(status.get("message", "")).strip()
                summary_model_progress_var.set(max(0.0, min(100.0, progress)))
                if st == "preparing":
                    summary_model_status_var.set(f"Состояние модели резюме: подготовка ({title})")
                elif st == "downloading":
                    summary_model_status_var.set(f"Состояние модели резюме: скачивание ({title})")
                elif st == "done":
                    summary_model_status_var.set(f"Состояние модели резюме: загружена ({title})")
                elif st == "error":
                    summary_model_status_var.set(f"Состояние модели резюме: ошибка ({title})")
                else:
                    present = bool((status.get("installed_models", {}) or {}).get(_selected_summary_model_key(), False))
                    summary_model_status_var.set(
                        f"Состояние модели резюме: {'загружена' if present else 'не загружена'} ({LocalMeetingSummaryGGUF.model_title(_selected_summary_model_key())})"
                    )
                if total > 0:
                    summary_model_stats_var.set(f"{progress:.1f}% В· {format_bytes(downloaded)} / {format_bytes(total)}")
                else:
                    summary_model_stats_var.set(message or "")

            def _poll_summary_model_status() -> None:
                summary_model_poll_job["id"] = None
                if not callable(on_get_summary_model_download_status_cb):
                    return
                try:
                    status = on_get_summary_model_download_status_cb()
                    if isinstance(status, dict):
                        _apply_summary_model_status_ui(status)
                except Exception:
                    pass
                if win.winfo_exists():
                    summary_model_poll_job["id"] = win.after(1200, _poll_summary_model_status)

            def _restart_summary_model_poll() -> None:
                job = summary_model_poll_job.get("id")
                if job:
                    try:
                        win.after_cancel(job)
                    except Exception:
                        pass
                summary_model_poll_job["id"] = None
                _poll_summary_model_status()

            def _download_summary_model_ui() -> None:
                if not callable(on_start_summary_model_download_cb):
                    summary_model_status_var.set("Скачивание модели резюме недоступно.")
                    return
                key = _selected_summary_model_key()
                try:
                    ok, message = on_start_summary_model_download_cb(key)
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                summary_model_stats_var.set(str(message or ""))
                _restart_summary_model_poll()

            def _delete_summary_model_ui() -> None:
                if not callable(on_delete_summary_model_cb):
                    summary_model_status_var.set("Удаление модели резюме недоступно.")
                    return
                key = _selected_summary_model_key()
                title = LocalMeetingSummaryGGUF.model_title(key)
                if not messagebox.askyesno("Удаление модели", f"Удалить модель резюме «{title}»?"):
                    return
                try:
                    ok, message = on_delete_summary_model_cb(key)
                except Exception as err:
                    ok, message = False, f"Ошибка: {err}"
                summary_model_stats_var.set(str(message or ""))
                _restart_summary_model_poll()

            ttk.Button(summary_btns, text="Скачать модель", command=_download_summary_model_ui, style="Settings.TButton").grid(row=0, column=0, padx=(0, 8))
            ttk.Button(summary_btns, text="Удалить модель", command=_delete_summary_model_ui, style="Danger.TButton").grid(row=0, column=1)

            def _apply_settings_snapshot_to_vars(snapshot: dict) -> None:
                nonlocal system_audio_label_to_id, system_audio_id_to_label, system_audio_labels
                api_var.set(str(snapshot.get("openrouter_api_key", "")))
                model_var.set(str(snapshot.get("model", "google/gemini-2.5-flash")))
                backend_mode_var.set(backend_mode_map.get(str(snapshot.get("transcription_backend", "api")).strip().lower(), "Через API (OpenRouter)"))
                snap_engine = str(snapshot.get("local_backend_engine", "whisper")).strip().lower()
                if snap_engine not in local_engine_map:
                    snap_engine = "whisper"
                local_engine_var.set(local_engine_map.get(snap_engine, "Whisper"))
                snap_local_model = str(snapshot.get("local_whisper_model", "small")).strip().lower()
                if snap_local_model not in local_model_map:
                    snap_local_model = "small"
                local_model_var.set(local_model_map.get(snap_local_model, "small (быстро)"))
                snap_vosk_model = str(snapshot.get("local_vosk_model", LocalVoskTranscriber.MODEL_NAME)).strip().lower()
                if snap_vosk_model not in vosk_model_map:
                    snap_vosk_model = LocalVoskTranscriber.MODEL_NAME
                local_vosk_model_var.set(vosk_model_map.get(snap_vosk_model, "small-ru (быстро)"))
                snap_sherpa_model = str(snapshot.get("local_sherpa_model", LocalSherpaWhispeRuTranscriber.MODEL_NAME)).strip().lower()
                if snap_sherpa_model not in sherpa_model_map:
                    snap_sherpa_model = LocalSherpaWhispeRuTranscriber.MODEL_NAME
                local_sherpa_model_var.set(sherpa_model_map.get(snap_sherpa_model, "GigaAM RNNT (быстрый, онлайн-установка)"))
                hotkey_var.set(str(snapshot.get("ptt_key", "f8")))
                activation_mode_var.set(activation_mode_map.get(str(snapshot.get("activation_mode", "hold")), "Зажми и говори"))
                auto_paste_var.set(bool(snapshot.get("auto_paste", True)))
                copy_result_var.set(bool(snapshot.get("copy_result_to_clipboard", False)))
                restore_clip_var.set(bool(snapshot.get("restore_clipboard", True)))
                punctuate_var.set(bool(snapshot.get("punctuate_text", True)))
                capitalize_var.set(bool(snapshot.get("capitalize_sentences", True)))
                terminal_period_var.set(bool(snapshot.get("terminal_period", True)))
                convert_numbers_var.set(bool(snapshot.get("convert_numbers", False)))
                quotes_dash_var.set(bool(snapshot.get("normalize_quotes_dashes", True)))
                snap_clean_mode = str(snapshot.get("clean_text_mode", "soft")).strip().lower()
                if snap_clean_mode not in clean_text_mode_map:
                    snap_clean_mode = "soft" if bool(snapshot.get("clean_text", False)) else "off"
                clean_text_mode_var.set(clean_text_mode_map.get(snap_clean_mode, "Выключено"))
                max_text_quality_var.set(bool(snapshot.get("max_text_quality", False)))
                whisper_medium_turbo_var.set(bool(snapshot.get("whisper_medium_turbo", False)))
                local_fast_refine_var.set(bool(snapshot.get("local_fast_refine", True)))
                overlay_var.set(bool(snapshot.get("show_overlay_widget", True)))
                live_preview_var.set(bool(snapshot.get("show_live_preview_text", True)))
                mic_quality_var.set(bool(snapshot.get("show_mic_quality_indicator", True)))
                record_sounds_var.set(bool(snapshot.get("play_recording_sounds", True)))
                notify_sounds_var.set(bool(snapshot.get("play_notification_sounds", True)))
                autostart_var.set(bool(snapshot.get("autostart_windows", False)))
                auto_update_var.set(bool(snapshot.get("auto_download_updates", False)))
                run_as_admin_var.set(bool(snapshot.get("run_as_admin", False)))
                hw_accel_var.set(bool(snapshot.get("hardware_acceleration", True)))
                no_sound_notify_var.set(bool(snapshot.get("notify_on_no_sound", True)))
                sample_rate_var.set(str(snapshot.get("sample_rate", 24000)))
                summary_backend_raw = str(snapshot.get("meeting_summary_backend", "api")).strip().lower()
                if summary_backend_raw not in meeting_summary_backend_map:
                    summary_backend_raw = "api"
                meeting_summary_backend_var.set(meeting_summary_backend_map.get(summary_backend_raw, "API"))
                summary_model_raw = LocalMeetingSummaryGGUF.normalize_model_name(
                    str(snapshot.get("meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
                )
                meeting_local_summary_model_var.set(LocalMeetingSummaryGGUF.model_title(summary_model_raw))
                system_audio_mode_raw = str(snapshot.get("meeting_system_audio_mode", "auto")).strip().lower()
                if system_audio_mode_raw not in meeting_system_audio_mode_map:
                    system_audio_mode_raw = "auto"
                meeting_system_audio_mode_var.set(meeting_system_audio_mode_map.get(system_audio_mode_raw, "Авто"))
                system_audio_items = [entry for entry in list(snapshot.get("_system_audio_devices") or []) if isinstance(entry, dict)]
                if system_audio_items:
                    system_audio_label_to_id = {
                        str(entry.get("label", "")).strip(): str(entry.get("id", "")).strip()
                        for entry in system_audio_items
                        if str(entry.get("label", "")).strip() and str(entry.get("id", "")).strip()
                    }
                    system_audio_id_to_label = {v: k for k, v in system_audio_label_to_id.items()}
                    system_audio_labels = list(system_audio_label_to_id.keys())
                    _set_dark_dropdown_values(meeting_system_audio_device_dropdown, meeting_system_audio_device_var, system_audio_labels)
                system_audio_id = str(snapshot.get("meeting_system_audio_device_id", "") or "").strip()
                if system_audio_id and system_audio_id in system_audio_id_to_label:
                    meeting_system_audio_device_var.set(system_audio_id_to_label[system_audio_id])
                elif system_audio_labels:
                    meeting_system_audio_device_var.set(system_audio_labels[0])
                else:
                    meeting_system_audio_device_var.set("")
                mic_device_var.set(str(snapshot.get("input_device", "")))
                _apply_transcriber_status_to_ui(snapshot)
                dl_status = snapshot.get("_model_download_status", {})
                if isinstance(dl_status, dict):
                    _apply_model_download_status_ui(dl_status)
                summary_dl_status = snapshot.get("_summary_model_download_status", {})
                if isinstance(summary_dl_status, dict):
                    _apply_summary_model_status_ui(summary_dl_status)
                media_snap = snapshot.get("_media_queue")
                if isinstance(media_snap, dict):
                    media_data.clear()
                    media_data.update(dict(media_snap))
                    if section_title_var.get() == "Транскрибация аудио и видео":
                        _render_media_queue(dict(media_snap))
                    else:
                        media_rendered["value"] = False
                meeting_snap = snapshot.get("_meeting_session")
                if isinstance(meeting_snap, dict):
                    meeting_data.clear()
                    meeting_data.update(dict(meeting_snap))
                    if section_title_var.get() == "Voice Meet AI":
                        _render_meeting_status(dict(meeting_snap))
                settings_runtime_state_var.set(runtime_model_state_var.get())
                settings_runtime_message_var.set(runtime_model_message_var.get())
                _render_update_status(snapshot)
                _sync_meeting_system_audio_controls()
                _sync_local_engine_controls()
                _sync_backend_visibility()

            def _on_reset_settings_ui() -> None:
                if not callable(on_reset_app_settings_cb):
                    status_var.set("Обработчик сброса настроек недоступен")
                    status_lbl.configure(foreground="#9f1239")
                    return
                try:
                    ok, message, snapshot = on_reset_app_settings_cb()
                except Exception as err:
                    ok, message, snapshot = False, f"Ошибка: {err}", {}
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok:
                    _apply_settings_snapshot_to_vars(dict(snapshot or {}))

            reset_settings_btn.configure(command=_on_reset_settings_ui)

            bottom = tk.Frame(settings_content, bg=c_bg)
            bottom.grid(row=9, column=0, sticky="ew", pady=(4, 4))
            bottom.columnconfigure(0, weight=1)
            actions = tk.Frame(bottom, bg=c_bg)
            actions.grid(row=0, column=1, sticky="e")

            def _paste_into_entry(entry_widget) -> str:
                # Paste exactly once from clipboard to avoid accidental key duplication.
                try:
                    data = win.clipboard_get()
                except Exception:
                    try:
                        data = pyperclip.paste()
                    except Exception:
                        data = ""
                if not isinstance(data, str):
                    data = str(data)
                if data:
                    try:
                        sel_start = entry_widget.index("sel.first")
                        sel_end = entry_widget.index("sel.last")
                        entry_widget.delete(sel_start, sel_end)
                        insert_at = sel_start
                    except Exception:
                        insert_at = entry_widget.index("insert")
                    entry_widget.insert(insert_at, data)
                    entry_widget.icursor(insert_at + len(data))
                return "break"

            ttk.Button(api_row, text="Вставить", command=lambda: _paste_into_entry(api_entry), style="Settings.TButton").grid(
                row=0, column=1, padx=(8, 0)
            )

            def bind_clipboard_shortcuts(entry_widget) -> None:
                # Use native Tk shortcuts (Ctrl+C/Ctrl+V/Ctrl+X) for maximum reliability.
                # Keep dedicated "Paste" button as fallback.
                return

            bind_clipboard_shortcuts(api_entry)
            bind_clipboard_shortcuts(model_entry)

            capture = {"handle": None, "keys": set()}
            capture_reader = {"stop": threading.Event()}

            def stop_capture() -> None:
                if capture["handle"] is not None:
                    try:
                        keyboard.unhook(capture["handle"])
                    except Exception:
                        logging.exception("Failed to unhook hotkey capture")
                    capture["handle"] = None
                capture_reader["stop"].set()

            def start_capture() -> None:
                stop_capture()
                capture_reader["stop"] = threading.Event()
                capture["keys"] = set()
                status_var.set("Нажмите сочетание...")
                if callable(notify_cb):
                    notify_cb("Нажмите нужную комбинацию клавиш")
                def _reader() -> None:
                    try:
                        combo = keyboard.read_hotkey(suppress=False)
                    except Exception as err:
                        combo = ""
                        logging.exception("Hotkey capture failed: %s", err)
                    if capture_reader["stop"].is_set():
                        return
                    parts = [VoiceProApp._normalize_key_name(part) for part in str(combo or "").split("+")]
                    parsed = {k for k in parts if k}
                    if not parsed:
                        if win.winfo_exists():
                            win.after(0, lambda: status_var.set("Не удалось распознать сочетание. Повторите захват."))
                        return
                    formatted = VoiceProApp._format_hotkey(parsed)
                    if win.winfo_exists():
                        def _apply_combo():
                            hotkey_var.set(formatted)
                            hotkey_preview_var.set(formatted.upper())
                            howto_step1_var.set(f"1. Нажмите и удерживайте {hotkey_preview_var.get()}, затем диктуйте текст.")
                            status_var.set(f"Горячая клавиша установлена: {formatted}")
                        win.after(0, _apply_combo)
                threading.Thread(target=_reader, daemon=True).start()

            def on_close() -> None:
                stop_capture()
                poll_job = model_download_poll_job.get("id")
                if poll_job:
                    try:
                        win.after_cancel(poll_job)
                    except Exception:
                        pass
                    model_download_poll_job["id"] = None
                media_job = media_poll_job.get("id")
                if media_job:
                    try:
                        win.after_cancel(media_job)
                    except Exception:
                        pass
                    media_poll_job["id"] = None
                meeting_job = meeting_poll_job.get("id")
                if meeting_job:
                    try:
                        win.after_cancel(meeting_job)
                    except Exception:
                        pass
                    meeting_poll_job["id"] = None
                summary_job = summary_model_poll_job.get("id")
                if summary_job:
                    try:
                        win.after_cancel(summary_job)
                    except Exception:
                        pass
                    summary_model_poll_job["id"] = None
                updater_job = updater_poll_job.get("id")
                if updater_job:
                    try:
                        win.after_cancel(updater_job)
                    except Exception:
                        pass
                    updater_poll_job["id"] = None
                _unbind_settings_wheel()
                if callable(on_close_cb):
                    on_close_cb()
                settings_win["win"] = None
                win.destroy()

            def on_save() -> None:
                stop_capture()
                if not callable(on_save_cb):
                    status_var.set("Обработчик сохранения недоступен")
                    return
                values = {
                    "openrouter_api_key": VoiceProApp._normalize_api_key_input(api_var.get()),
                    "model": model_var.get().strip(),
                    "transcription_backend": backend_mode_rev.get(backend_mode_var.get().strip(), "api"),
                    "local_backend_engine": local_engine_rev.get(local_engine_var.get().strip(), "whisper"),
                    "local_whisper_model": local_model_rev.get(local_model_var.get().strip(), "small"),
                    "local_vosk_model": vosk_model_rev.get(local_vosk_model_var.get().strip(), LocalVoskTranscriber.MODEL_NAME),
                    "local_sherpa_model": sherpa_model_rev.get(local_sherpa_model_var.get().strip(), LocalSherpaWhispeRuTranscriber.MODEL_NAME),
                    "ptt_key": hotkey_var.get().strip(),
                    "activation_mode": activation_mode_rev.get(activation_mode_var.get().strip(), "hold"),
                    "input_device": mic_device_var.get().strip(),
                    "auto_paste": bool(auto_paste_var.get()),
                    "copy_result_to_clipboard": bool(copy_result_var.get()),
                    "restore_clipboard": bool(restore_clip_var.get()),
                    "history_enabled": bool(history_enabled_var.get()),
                    "autoreplace_enabled": bool(rules_toggle_state["value"]),
                    "punctuate_text": bool(punctuate_var.get()),
                    "capitalize_sentences": bool(capitalize_var.get()),
                    "terminal_period": bool(terminal_period_var.get()),
                    "convert_numbers": bool(convert_numbers_var.get()),
                    "normalize_quotes_dashes": bool(quotes_dash_var.get()),
                    "clean_text_mode": clean_text_mode_rev.get(clean_text_mode_var.get().strip(), "off"),
                    "clean_text": clean_text_mode_rev.get(clean_text_mode_var.get().strip(), "off") != "off",
                    "max_text_quality": bool(max_text_quality_var.get()),
                    "whisper_medium_turbo": bool(whisper_medium_turbo_var.get()),
                    "local_fast_refine": bool(local_fast_refine_var.get()),
                    "show_overlay_widget": bool(overlay_var.get()),
                    "show_live_preview_text": bool(live_preview_var.get()),
                    "show_mic_quality_indicator": bool(mic_quality_var.get()),
                    "play_recording_sounds": bool(record_sounds_var.get()),
                    "play_notification_sounds": bool(notify_sounds_var.get()),
                    "autostart_windows": bool(autostart_var.get()),
                    "auto_download_updates": bool(auto_update_var.get()),
                    "run_as_admin": bool(run_as_admin_var.get()),
                    "hardware_acceleration": bool(hw_accel_var.get()),
                    "notify_on_no_sound": bool(no_sound_notify_var.get()),
                    "sample_rate": sample_rate_var.get().strip(),
                    "meeting_summary_backend": meeting_summary_backend_rev.get(meeting_summary_backend_var.get().strip(), "api"),
                    "meeting_local_summary_model": gguf_model_rev.get(
                        meeting_local_summary_model_var.get().strip(),
                        LocalMeetingSummaryGGUF.DEFAULT_MODEL,
                    ),
                    "meeting_system_audio_mode": meeting_system_audio_mode_rev.get(
                        meeting_system_audio_mode_var.get().strip(),
                        "auto",
                    ),
                    "meeting_system_audio_device_id": system_audio_label_to_id.get(
                        meeting_system_audio_device_var.get().strip(),
                        "",
                    ),
                }
                try:
                    result = on_save_cb(values)
                    ok = False
                    message = "Ошибка сохранения"
                    snapshot = {}
                    if isinstance(result, tuple):
                        if len(result) >= 1:
                            ok = bool(result[0])
                        if len(result) >= 2:
                            message = str(result[1])
                        if len(result) >= 3 and isinstance(result[2], dict):
                            snapshot = dict(result[2] or {})
                    else:
                        ok, message = bool(result), ("Сохранено" if bool(result) else "Ошибка сохранения")
                except Exception as err:
                    ok, message, snapshot = False, f"Ошибка: {err}", {}
                status_var.set(message)
                status_lbl.configure(foreground="#2b7a0b" if ok else "#9f1239")
                if ok and isinstance(snapshot, dict) and snapshot:
                    _apply_settings_snapshot_to_vars(snapshot)

            ttk.Button(hotkey_btns, text="Захват клавиши", command=start_capture, style="Settings.TButton").grid(row=0, column=0, padx=(0, 8))
            ttk.Button(
                hotkey_btns,
                text="Установить F8",
                command=lambda: (hotkey_var.set("f8"), hotkey_preview_var.set("F8"), howto_step1_var.set("1. Нажмите и удерживайте F8, затем диктуйте текст.")),
                style="Settings.TButton",
            ).grid(row=0, column=1)

            ttk.Button(actions, text="Сохранить", command=on_save, style="Primary.TButton").grid(row=0, column=0, padx=(0, 8))
            ttk.Button(actions, text="Закрыть", command=on_close, style="Settings.TButton").grid(row=0, column=1)

            nav_buttons: dict[str, tk.Frame] = {}
            nav_primary_items = [
                "Главная и статистика",
                "История",
                "Транскрибация аудио и видео",
                "Voice Meet AI",
                "Автозамена",
                "Настройки",
            ]
            nav_help_items = ["Как пользоваться"]
            nav_state = {"hover": set(), "focus": ""}
            page_desc_map = {
                "Главная и статистика": "Ключевые метрики продуктивности и быстрый старт.",
                "История": "Просматривайте, копируйте и управляйте сохранёнными расшифровками.",
                "Транскрибация аудио и видео": "Пакетная обработка файлов и экспорт результатов.",
                "Voice Meet AI": "Транскрибация онлайн-встреч и формирование резюме.",
                "Автозамена": "Настраивайте правила замены текста и RegExp-преобразования.",
                "Настройки": "Параметры API, записи, вставки и поведения приложения.",
                "Как пользоваться": "Инструкции, советы по качеству и решение проблем.",
            }

            nav_wrap = tk.Frame(sidebar, bg=c_sidebar)
            nav_wrap.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            nav_wrap.columnconfigure(0, weight=1)

            nav_primary_wrap = tk.Frame(nav_wrap, bg=c_sidebar)
            nav_primary_wrap.grid(row=0, column=0, sticky="ew")
            nav_primary_wrap.columnconfigure(0, weight=1)

            divider = tk.Frame(nav_wrap, bg=c_sidebar_border, height=1)
            divider.grid(row=1, column=0, sticky="ew", padx=4, pady=(12, 8))

            tk.Label(
                nav_wrap,
                text="Справка",
                fg=c_sidebar_muted_text,
                bg=c_sidebar,
                font=("Segoe UI", 9, "bold"),
            ).grid(row=2, column=0, sticky="w", padx=6, pady=(0, 5))

            nav_help_wrap = tk.Frame(nav_wrap, bg=c_sidebar)
            nav_help_wrap.grid(row=3, column=0, sticky="ew")
            nav_help_wrap.columnconfigure(0, weight=1)

            def _draw_nav_icon(canvas: tk.Canvas, name: str, active: bool = False) -> None:
                canvas.delete("all")
                stroke = c_sidebar_active_text if active else c_sidebar_text
                soft = "#9cb2d2" if not active else c_sidebar_active_text
                if name == "Главная и статистика":
                    canvas.create_rectangle(2, 2, 7, 7, outline=stroke, width=1.5)
                    canvas.create_rectangle(9, 2, 14, 7, outline=stroke, width=1.5)
                    canvas.create_rectangle(2, 9, 7, 14, outline=stroke, width=1.5)
                    canvas.create_rectangle(9, 9, 14, 14, outline=stroke, width=1.5)
                elif name == "История":
                    canvas.create_oval(2.5, 2.5, 13.5, 13.5, outline=stroke, width=1.6)
                    canvas.create_line(8, 4.5, 8, 8, fill=stroke, width=1.6)
                    canvas.create_line(8, 8, 10.5, 9.8, fill=stroke, width=1.6)
                    canvas.create_line(3, 3.5, 1.7, 2.2, fill=soft, width=1.4)
                    canvas.create_line(13, 3.5, 14.3, 2.2, fill=soft, width=1.4)
                elif name == "Транскрибация аудио и видео":
                    canvas.create_line(2, 11, 4, 11, 4, 5, 6, 5, 6, 13, 8, 13, 8, 4, 10, 4, 10, 10, 12, 10, 12, 7, 14, 7, fill=stroke, width=1.5)
                elif name == "Voice Meet AI":
                    canvas.create_oval(4, 2, 12, 10, outline=stroke, width=1.6)
                    canvas.create_line(8, 10, 8, 13.5, fill=stroke, width=1.6)
                    canvas.create_line(5.5, 13.5, 10.5, 13.5, fill=stroke, width=1.6)
                    canvas.create_line(13, 3.5, 15, 3.5, fill=soft, width=1.5)
                    canvas.create_line(14, 2.5, 14, 4.5, fill=soft, width=1.5)
                elif name == "Автозамена":
                    canvas.create_line(2, 5, 14, 5, fill=stroke, width=1.5)
                    canvas.create_polygon(14, 5, 11.2, 3, 11.2, 7, fill=stroke, outline=stroke)
                    canvas.create_line(14, 11, 2, 11, fill=stroke, width=1.5)
                    canvas.create_polygon(2, 11, 4.8, 9, 4.8, 13, fill=stroke, outline=stroke)
                elif name == "Настройки":
                    canvas.create_oval(5.3, 5.3, 10.7, 10.7, outline=stroke, width=1.5)
                    for x1, y1, x2, y2 in [
                        (8, 1.8, 8, 4), (8, 12, 8, 14.2),
                        (1.8, 8, 4, 8), (12, 8, 14.2, 8),
                        (3.1, 3.1, 4.6, 4.6), (11.4, 11.4, 12.9, 12.9),
                        (11.4, 4.6, 12.9, 3.1), (3.1, 12.9, 4.6, 11.4),
                    ]:
                        canvas.create_line(x1, y1, x2, y2, fill=stroke, width=1.4)
                else:  # Как пользоваться
                    canvas.create_oval(2.3, 2.3, 13.7, 13.7, outline=stroke, width=1.6)
                    canvas.create_text(8, 8.1, text="?", fill=stroke, font=("Segoe UI", 8, "bold"))

            def _apply_nav_item_style(name: str) -> None:
                item = nav_buttons.get(name) or {}
                frame = item.get("frame")
                if frame is None:
                    return
                selected = section_title_var.get() == name
                hovered = name in nav_state["hover"]
                focused = nav_state.get("focus") == name
                bg = c_sidebar_active if selected else (c_sidebar_hover if hovered else c_sidebar)
                border = "#2e4e7a" if selected else ("#2a3e62" if (hovered or focused) else c_sidebar)
                fg = c_sidebar_active_text if selected else "#d6e3f8"
                icon_bg = c_nav_icon_active_bg if selected else c_nav_icon_bg
                icon_border = c_nav_icon_active_border if selected else c_nav_icon_border
                frame.configure(bg=bg, highlightbackground=border, highlightcolor=border)
                item["label"].configure(bg=bg, fg=fg)
                item["icon_wrap"].configure(bg=icon_bg, highlightbackground=icon_border, highlightcolor=icon_border)
                item["icon_canvas"].configure(bg=icon_bg)
                _draw_nav_icon(item["icon_canvas"], name, active=selected or hovered or focused)

            def _refresh_all_nav_styles() -> None:
                for key in nav_buttons.keys():
                    _apply_nav_item_style(key)

            def show_page(name: str) -> None:
                section_title_var.set(name)
                section_desc_var.set(page_desc_map.get(name, ""))
                frame_to_show = pages[name]
                frame_to_show.tkraise()
                if name == "Главная и статистика":
                    reset_stats_btn.grid()
                else:
                    reset_stats_btn.grid_remove()
                if name == "История":
                    clear_history_btn.grid()
                    _ensure_history_rendered()
                else:
                    clear_history_btn.grid_remove()
                if name == "Транскрибация аудио и видео":
                    clear_media_btn.grid()
                    if not media_rendered["value"]:
                        _render_media_queue(media_data)
                    _restart_media_poll()
                else:
                    clear_media_btn.grid_remove()
                if name == "Voice Meet AI":
                    _render_meeting_status(meeting_data)
                    _restart_meeting_poll()
                if name == "Автозамена":
                    reset_rules_btn.grid()
                    add_rule_btn.grid()
                    _ensure_rules_rendered()
                else:
                    reset_rules_btn.grid_remove()
                    add_rule_btn.grid_remove()
                if name == "Настройки":
                    reset_settings_btn.grid()
                else:
                    reset_settings_btn.grid_remove()
                _refresh_all_nav_styles()

            sidebar.rowconfigure(1, weight=1)
            version_wrap = tk.Frame(sidebar, bg=c_sidebar)
            version_wrap.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 14))
            tk.Label(
                version_wrap,
                text=f"Версия {APP_VERSION}",
                fg=c_sidebar_muted_text,
                bg=c_sidebar,
                font=("Segoe UI", 9),
            ).pack(side="left")
            if str(APP_RELEASE_BADGE or "").strip():
                tk.Label(
                    version_wrap,
                    text=str(APP_RELEASE_BADGE).strip(),
                    fg="#93c5fd",
                    bg="#0b2042",
                    font=("Segoe UI", 8, "bold"),
                    padx=8,
                    pady=1,
                    borderwidth=1,
                    relief="solid",
                    highlightthickness=0,
                ).pack(side="left", padx=(8, 0))

            def _create_nav_item(container: tk.Frame, item_name: str) -> None:
                row = tk.Frame(
                    container,
                    bg=c_sidebar,
                    highlightthickness=1,
                    highlightbackground=c_sidebar,
                    highlightcolor=c_sidebar,
                    cursor="hand2",
                )
                row.grid(sticky="ew", padx=2, pady=3)
                row.columnconfigure(1, weight=1)
                row.configure(takefocus=True)

                icon_wrap = tk.Frame(
                    row,
                    width=30,
                    height=30,
                    bg=c_nav_icon_bg,
                    highlightthickness=1,
                    highlightbackground=c_nav_icon_border,
                    highlightcolor=c_nav_icon_border,
                )
                icon_wrap.grid(row=0, column=0, padx=(10, 10), pady=8, sticky="w")
                icon_wrap.grid_propagate(False)
                icon_canvas = tk.Canvas(
                    icon_wrap,
                    width=16,
                    height=16,
                    bg=c_nav_icon_bg,
                    bd=0,
                    highlightthickness=0,
                )
                icon_canvas.place(relx=0.5, rely=0.5, anchor="center")

                lbl = tk.Label(
                    row,
                    text=item_name,
                    bg=c_sidebar,
                    fg=c_sidebar_text,
                    font=("Segoe UI", 10, "bold"),
                    justify="left",
                    wraplength=200,
                    anchor="w",
                )
                lbl.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=8)

                def _activate(_e=None, n=item_name):
                    show_page(n)

                def _enter(_e=None, n=item_name):
                    if _is_dragging_window():
                        return
                    nav_state["hover"].add(n)
                    _apply_nav_item_style(n)

                def _leave(_e=None, n=item_name):
                    if _is_dragging_window():
                        return
                    nav_state["hover"].discard(n)
                    _apply_nav_item_style(n)

                def _focus_in(_e=None, n=item_name):
                    nav_state["focus"] = n
                    _apply_nav_item_style(n)

                def _focus_out(_e=None, n=item_name):
                    if nav_state.get("focus") == n:
                        nav_state["focus"] = ""
                    _apply_nav_item_style(n)

                for widget in (row, lbl, icon_wrap, icon_canvas):
                    widget.bind("<Button-1>", _activate)
                    widget.bind("<Enter>", _enter)
                    widget.bind("<Leave>", _leave)
                    widget.bind("<FocusIn>", _focus_in)
                    widget.bind("<FocusOut>", _focus_out)

                row.bind("<Return>", _activate)
                row.bind("<space>", _activate)

                nav_buttons[item_name] = {
                    "frame": row,
                    "label": lbl,
                    "icon_wrap": icon_wrap,
                    "icon_canvas": icon_canvas,
                }
                _apply_nav_item_style(item_name)

            for nav_name in nav_primary_items:
                _create_nav_item(nav_primary_wrap, nav_name)
            for nav_name in nav_help_items:
                _create_nav_item(nav_help_wrap, nav_name)

            _apply_settings_snapshot_to_vars(config_data)
            show_page("Настройки")
            win.protocol("WM_DELETE_WINDOW", on_close)
            win.focus_force()
            api_entry.focus_set()
            # Defer heavier background tasks to keep initial window opening smooth.
            def _start_deferred_settings_tasks() -> None:
                if not win.winfo_exists():
                    return
                try:
                    win.after(90, _poll_model_download_status)
                    win.after(180, _poll_summary_model_status)
                    win.after(280, _poll_updater_status)
                    win.after(380, _poll_media_queue)
                    win.after(480, _poll_meeting_status)
                    win.after(650, _refresh_system_audio_devices_ui)
                except Exception:
                    logging.exception("Deferred settings tasks startup failed")

            win.after(10, _start_deferred_settings_tasks)
            if callable(on_open_cb):
                win.after(0, on_open_cb)
            logging.info("Settings window opened in %.1f ms", (time.perf_counter() - open_started) * 1000.0)

        def poll_queue() -> None:
            destroyed = False
            processed = 0
            while True:
                try:
                    action, value = self._queue.get_nowait()
                except queue.Empty:
                    break
                processed += 1

                try:
                    if action == "overlay_show":
                        apply_overlay_mode(str(value or "recording"))
                    elif action == "overlay_hide":
                        hide_overlay()
                    elif action == "overlay_preview_text":
                        apply_overlay_preview_text(str(value or ""))
                    elif action == "overlay_preview_enabled":
                        apply_overlay_preview_enabled(bool(value))
                    elif action == "overlay_quality_text":
                        apply_overlay_quality_text(str(value or ""))
                    elif action == "overlay_quality_enabled":
                        apply_overlay_quality_enabled(bool(value))
                    elif action == "open_settings":
                        open_settings_window(dict(value or {}))
                    elif action == "shutdown":
                        destroyed = True
                        root.destroy()
                        break
                except Exception:
                    logging.exception("UI manager action failed: %s", action)
                    if action == "open_settings":
                        try:
                            payload = dict(value or {})
                            notify_cb = payload.get("notify")
                            if callable(notify_cb):
                                notify_cb("Ошибка при открытии настроек. Проверьте app.log.")
                        except Exception:
                            pass

            if not destroyed:
                # Adaptive polling: lower idle CPU/GPU load for smoother window dragging.
                root.after(20 if processed else 100, poll_queue)

        ensure_overlay()
        hide_overlay()
        self._started.set()
        root.after(100, poll_queue)
        root.mainloop()


class VoiceProApp:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.config_store = ConfigStore(CONFIG_PATH)
        self.stats_store = StatsStore(STATS_PATH)
        self.history_store = HistoryStore(HISTORY_PATH)
        self.autoreplace_store = AutoReplaceStore(AUTOREPLACE_PATH)
        self.config = self.config_store.load()
        # Keep update feed embedded in app build (not editable in Settings).
        self.config.update_feed_url = DEFAULT_UPDATE_FEED_URL
        self.config.update_channel = DEFAULT_UPDATE_CHANNEL

        self.recorder = AudioRecorder(self.config)
        self._startup_warning = ""
        try:
            self.transcriber = self._create_transcriber(self.config)
        except Exception as err:
            logging.exception("Failed to initialize selected transcriber, fallback to API")
            self._startup_warning = f"Локальный режим недоступен: {err}"
            self.config.transcription_backend = "api"
            self.transcriber = OpenRouterTranscriber(self.config)
        self.overlay = UiManager(self.base_dir)

        self._icon: Optional[Icon] = None
        self._running = True
        self._recording = False
        self._busy = False
        self._clipboard_lock = threading.Lock()
        self._hotkey_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._rules_lock = threading.Lock()
        self._target_hwnd: int = 0
        self._mutex_handle: Optional[int] = None
        self._ptt_combo_keys: set[str] = set()
        self._pressed_keys: set[str] = set()
        self._ptt_armed = False
        self._hotkey_handles: list[object] = []
        self._ignore_hotkey_until = 0.0
        self._settings_open = False
        self._has_successful_insert = False
        self._last_insert_target_hwnd: Optional[int] = None
        self._autoreplace_rules_cache = self.autoreplace_store.load()
        self._live_preview_stop_event = threading.Event()
        self._live_preview_thread: Optional[threading.Thread] = None
        self._mic_quality_stop_event = threading.Event()
        self._mic_quality_thread: Optional[threading.Thread] = None
        self._ipc_stop_event = threading.Event()
        self._ipc_thread: Optional[threading.Thread] = None
        self._ipc_last_token = ""
        self._model_dl_lock = threading.Lock()
        self._model_dl_thread: Optional[threading.Thread] = None
        self._model_dl_stop_event = threading.Event()
        self._model_dl_status = {
            "state": "idle",  # idle | preparing | downloading | done | error
            "engine": "whisper",
            "model": "",
            "progress_percent": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "message": "Ожидание",
            "started_at": 0.0,
            "finished_at": 0.0,
            "error": "",
        }
        self._summary_model_dl_lock = threading.Lock()
        self._summary_model_dl_thread: Optional[threading.Thread] = None
        self._summary_model_dl_status = {
            "state": "idle",  # idle | preparing | downloading | done | error
            "model": LocalMeetingSummaryGGUF.DEFAULT_MODEL,
            "progress_percent": 0.0,
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "message": "Ожидание",
            "started_at": 0.0,
            "finished_at": 0.0,
            "error": "",
        }
        self._media_lock = threading.Lock()
        self._media_queue_items: list[dict] = []
        self._media_active_id = ""
        self._media_is_processing = False
        self._media_worker_thread: Optional[threading.Thread] = None
        self._media_id_counter = 0
        self._meeting_lock = threading.Lock()
        self._meeting_state = {
            "state": "idle",  # idle|recording|paused|transcribing|summarizing|done|error
            "started_at": 0.0,
            "duration_sec": 0.0,
            "transcript_text": "",
            "summary_text": "",
            "audio_levels": {"mic": 0, "system": 0},
            "error": "",
            "backend_label": "",
            "system_backend": "none",
            "loopback_available": True,
            "loopback_device_name": "",
            "loopback_channels": 0,
            "loopback_error": "",
            "loopback_error_code": "",
            "loopback_error_message": "",
            "summary_backend": "",
            "summary_model": "",
            "is_running": False,
            "is_paused": False,
            "updated_at": 0.0,
        }
        self._meeting_stop_event = threading.Event()
        self._meeting_thread: Optional[threading.Thread] = None
        self._meeting_mic_stream = None
        self._meeting_system_stream = None
        self._meeting_summary_worker: Optional[threading.Thread] = None
        self._meeting_last_loopback_device: Optional[int] = None
        self._meeting_reinit_loopback_event = threading.Event()
        self._meeting_loopback_info = {"name": "", "channels": 0, "error": ""}
        self._ollama_http = requests.Session()
        self._refine_http = requests.Session()
        self._summary_llm_lock = threading.Lock()
        self._summary_llm = None
        self._summary_llm_model_path = ""
        self._summary_llm_kind = ""
        self._perf_lock = threading.Lock()
        self._perf_window_total_ms = deque(maxlen=20)
        self._updater_lock = threading.Lock()
        self._updater_status = {
            "state": "idle",  # idle|checking|available|downloading|verifying|error|up_to_date
            "message": "",
            "available_version": "",
            "published_at": "",
            "release_notes": "",
            "downloaded_bytes": 0,
            "total_bytes": 0,
            "installer_path": "",
            "updated_at": 0.0,
        }
        self._ui_devices_cache_lock = threading.Lock()
        self._ui_devices_cache = {
            "input": {"ts": 0.0, "data": []},
            "system": {"ts": 0.0, "data": []},
        }

    def run(self) -> None:
        if not self._acquire_single_instance():
            return

        self._set_app_user_model_id()
        self._setup_logging()
        logging.info("Starting %s", APP_NAME)
        self._start_ipc_listener()

        self._warm_up_audio_stream()
        self._bind_hotkey()
        self._create_tray()
        self._schedule_auto_update_check()
        if self._startup_warning:
            self._notify(self._startup_warning)
        # Open settings on initial launch while keeping app running in tray.
        self._on_open_settings(None, None)

        try:
            self._icon.run()
        finally:
            self.shutdown()

    @staticmethod
    def _create_transcriber(config: AppConfig):
        mode = str(getattr(config, "transcription_backend", "api") or "api").strip().lower()
        if mode == "local":
            engine = str(getattr(config, "local_backend_engine", "whisper") or "whisper").strip().lower()
            if engine == "vosk":
                return LocalVoskTranscriber(config)
            if engine == "sherpa":
                return LocalSherpaWhispeRuTranscriber(config)
            return LocalWhisperSmallTranscriber(config)
        return OpenRouterTranscriber(config)

    def _get_transcriber_runtime_status(self) -> dict:
        backend = str(getattr(self.config, "transcription_backend", "api") or "api").strip().lower()
        fallback = {
            "backend": backend,
            "engine": str(getattr(self.config, "local_backend_engine", "whisper") or "whisper").strip().lower(),
            "model_label": (
                "Vosk (локально)"
                if backend == "local" and str(getattr(self.config, "local_backend_engine", "whisper")).strip().lower() == "vosk"
                else "Sherpa (локально)"
                if backend == "local" and str(getattr(self.config, "local_backend_engine", "whisper")).strip().lower() == "sherpa"
                else "Whisper (локально)"
            ) if backend == "local" else str(self.config.model or "OpenRouter"),
            "ready": True,
            "state": "ready",
            "message": "Готово",
        }
        try:
            if hasattr(self.transcriber, "get_runtime_status"):
                data = self.transcriber.get_runtime_status()
                if isinstance(data, dict):
                    fallback.update(data)
        except Exception:
            logging.exception("Failed to read transcriber runtime status")
        return fallback

    def _set_updater_status(self, **kwargs) -> None:
        with self._updater_lock:
            self._updater_status.update(kwargs)
            self._updater_status["updated_at"] = time.time()

    def _get_updater_status(self) -> dict:
        with self._updater_lock:
            return dict(self._updater_status)

    def _format_update_status_message(self, status: dict) -> str:
        state = str(status.get("state", "idle") or "idle").strip().lower()
        if state == "checking":
            return "Проверка обновлений..."
        if state == "available":
            ver = str(status.get("available_version", "") or "").strip()
            return f"Доступна версия {ver}" if ver else "Доступно обновление"
        if state == "downloading":
            total = int(status.get("total_bytes", 0) or 0)
            done = int(status.get("downloaded_bytes", 0) or 0)
            if total > 0:
                return f"Скачивание обновления: {int((done / total) * 100)}%"
            return "Скачивание обновления..."
        if state == "verifying":
            return "Проверка целостности обновления..."
        if state == "up_to_date":
            return f"Актуальная версия {APP_VERSION}"
        if state == "error":
            msg = str(status.get("message", "") or "").strip()
            return msg or "Ошибка проверки обновлений"
        return f"Текущая версия {APP_VERSION}"

    def _effective_update_feed_url(self) -> str:
        # Embedded release feed (not configurable from Settings UI).
        feed = str(DEFAULT_UPDATE_FEED_URL or "").strip()
        return feed

    def _check_for_updates(self, notify_if_latest: bool = False) -> tuple[bool, str, dict, Optional[dict]]:
        feed_url = self._effective_update_feed_url()
        channel = DEFAULT_UPDATE_CHANNEL
        if not feed_url:
            self._set_updater_status(state="error", message="Не задан встроенный update feed URL.")
            snap = self._settings_snapshot()
            return False, "Не задан встроенный update feed URL", snap, None

        self._set_updater_status(
            state="checking",
            message="Проверка обновлений...",
            available_version="",
            release_notes="",
            published_at="",
            downloaded_bytes=0,
            total_bytes=0,
            installer_path="",
        )
        try:
            info = updater.check_for_update(APP_VERSION, feed_url, channel=channel)
            if info is None:
                self._set_updater_status(state="up_to_date", message=f"Актуальная версия {APP_VERSION}")
                snap = self._settings_snapshot()
                msg = "Актуальная версия"
                if notify_if_latest:
                    self._notify(msg)
                return True, msg, snap, None

            self._set_updater_status(
                state="available",
                message=f"Доступна версия {info.version}",
                available_version=info.version,
                release_notes=info.release_notes,
                published_at=info.published_at,
            )
            snap = self._settings_snapshot()
            update_info = {
                "version": info.version,
                "channel": info.channel,
                "installer_url": info.installer_url,
                "sha256": info.sha256,
                "release_notes": info.release_notes,
                "published_at": info.published_at,
                "mandatory": bool(info.mandatory),
            }
            return True, f"Доступно обновление: {info.version}", snap, update_info
        except requests.exceptions.RequestException as err:
            # Network/feed errors are expected occasionally (offline DNS/proxy/release lag).
            logging.warning("Update check failed (network/feed): %s", err)
            self._set_updater_status(state="error", message=f"Ошибка проверки: {err}")
            return False, f"Ошибка проверки обновлений: {err}", self._settings_snapshot(), None
        except Exception as err:
            logging.exception("Update check failed")
            self._set_updater_status(state="error", message=f"Ошибка проверки: {err}")
            return False, f"Ошибка проверки обновлений: {err}", self._settings_snapshot(), None

    @staticmethod
    def _bytes_to_mb_text(value: int) -> str:
        try:
            return f"{max(0, int(value)) / (1024.0 * 1024.0):.1f} MB"
        except Exception:
            return "0.0 MB"

    def _start_update_install(self, update_payload: dict) -> tuple[bool, str, dict]:
        try:
            info = updater.UpdateInfo(
                version=str(update_payload.get("version", "") or "").strip(),
                channel=str(update_payload.get("channel", DEFAULT_UPDATE_CHANNEL) or DEFAULT_UPDATE_CHANNEL).strip() or DEFAULT_UPDATE_CHANNEL,
                installer_url=str(update_payload.get("installer_url", "") or "").strip(),
                sha256=str(update_payload.get("sha256", "") or "").strip(),
                release_notes=str(update_payload.get("release_notes", "") or "").strip(),
                published_at=str(update_payload.get("published_at", "") or "").strip(),
                mandatory=bool(update_payload.get("mandatory", False)),
            )
            if not info.version or not info.installer_url or not info.sha256:
                raise ValueError("Некорректные данные обновления")

            def _on_progress(downloaded: int, total: int) -> None:
                self._set_updater_status(
                    state="downloading",
                    message=f"Скачивание: {self._bytes_to_mb_text(downloaded)} / {self._bytes_to_mb_text(total)}",
                    downloaded_bytes=int(downloaded),
                    total_bytes=int(total),
                )

            self._set_updater_status(
                state="downloading",
                message=f"Скачивание версии {info.version}",
                available_version=info.version,
                downloaded_bytes=0,
                total_bytes=0,
            )
            installer_path = updater.download_installer(info, progress_cb=_on_progress)
            self._set_updater_status(state="verifying", message="Проверка SHA-256...", installer_path=str(installer_path))
            if not updater.verify_sha256(installer_path, info.sha256):
                self._set_updater_status(state="error", message="Ошибка проверки SHA-256. Установка отменена.")
                return False, "Ошибка проверки SHA-256. Установка отменена.", self._settings_snapshot()

            self._set_updater_status(
                state="available",
                message=f"Обновление {info.version} готово к установке",
                installer_path=str(installer_path),
            )
            try:
                self._notify("Запускаю установщик обновления...")
            except Exception:
                pass
            updater.launch_installer(installer_path, silent=False)
            threading.Thread(target=self._shutdown_after_update_launch, daemon=True).start()
            return True, "Запущено обновление поверх текущей версии. Приложение будет закрыто автоматически.", self._settings_snapshot()
        except Exception as err:
            logging.exception("Failed to start update install")
            self._set_updater_status(state="error", message=f"Ошибка обновления: {err}")
            return False, f"Ошибка обновления: {err}", self._settings_snapshot()

    def _shutdown_after_update_launch(self) -> None:
        time.sleep(0.7)
        self._running = False
        try:
            if self._icon is not None:
                self._icon.stop()
        except Exception:
            pass

    def _set_model_download_status(self, **kwargs) -> None:
        with self._model_dl_lock:
            self._model_dl_status.update(kwargs)

    def _get_model_download_status(self) -> dict:
        with self._model_dl_lock:
            return dict(self._model_dl_status)

    @staticmethod
    def _get_local_models_presence(engine: str) -> tuple[dict[str, bool], dict[str, int]]:
        present: dict[str, bool] = {}
        sizes: dict[str, int] = {}
        engine_name = str(engine or "whisper").strip().lower()
        if engine_name == "vosk":
            models = sorted(LocalVoskTranscriber.MODEL_URLS.keys())
            model_cls = LocalVoskTranscriber
        elif engine_name == "sherpa":
            models = [LocalSherpaWhispeRuTranscriber.MODEL_NAME]
            model_cls = LocalSherpaWhispeRuTranscriber
        else:
            models = sorted(LocalWhisperSmallTranscriber.ALLOWED_MODELS)
            model_cls = LocalWhisperSmallTranscriber
        for model in models:
            try:
                present[model] = bool(model_cls.is_model_present(model))
            except Exception:
                present[model] = False
            try:
                sizes[model] = int(model_cls.model_cached_size_bytes(model))
            except Exception:
                sizes[model] = 0
        return present, sizes

    @staticmethod
    def _estimate_repo_total_size_bytes(engine: str, model_name: str) -> int:
        engine_name = str(engine or "whisper").strip().lower()
        if engine_name == "vosk":
            # Approximate sizes for stable progress bar UX.
            rough = {
                "vosk-model-small-ru-0.22": int(50 * 1024 * 1024),
                "vosk-model-ru-0.42": int(1850 * 1024 * 1024),
            }
            return int(rough.get(LocalVoskTranscriber.normalize_model_name(model_name), 0))
        if engine_name == "sherpa":
            if HfApi is None:
                return int(250 * 1024 * 1024)
            try:
                normalized = LocalSherpaWhispeRuTranscriber.normalize_model_name(model_name)
                repo_id = LocalSherpaWhispeRuTranscriber._repo_for_model(normalized)
                if not repo_id:
                    return int(250 * 1024 * 1024)
                info = HfApi().model_info(repo_id, files_metadata=True)
                sizes = {}
                for sibling in list(getattr(info, "siblings", []) or []):
                    name = str(getattr(sibling, "rfilename", "") or "").strip()
                    if not name:
                        continue
                    try:
                        sizes[name] = int(getattr(sibling, "size", 0) or 0)
                    except Exception:
                        sizes[name] = 0
                total = 0
                for req in LocalSherpaWhispeRuTranscriber.REQUIRED_MODEL_FILES:
                    aliases = LocalSherpaWhispeRuTranscriber.REQUIRED_FILE_ALIASES.get(req, (req,))
                    for alias in aliases:
                        if alias in sizes:
                            total += max(0, int(sizes.get(alias, 0)))
                            break
                if total > 0:
                    return int(total)
            except Exception:
                logging.exception("Failed to estimate Sherpa model size")
            return int(250 * 1024 * 1024)
        if HfApi is None:
            return 0
        try:
            api = HfApi()
            info = api.model_info(LocalWhisperSmallTranscriber.repo_id_for_model(model_name), files_metadata=True)
            total = 0
            for sibling in list(getattr(info, "siblings", []) or []):
                size = getattr(sibling, "size", None)
                if isinstance(size, int) and size > 0:
                    total += size
            return int(total)
        except Exception:
            logging.exception("Failed to estimate model size")
            return 0

    @staticmethod
    def _normalize_local_download_target(engine: str, model_name: str) -> tuple[str, str]:
        engine_name = str(engine or "whisper").strip().lower()
        if engine_name == "vosk":
            return "vosk", LocalVoskTranscriber.normalize_model_name(model_name)
        if engine_name == "sherpa":
            return "sherpa", LocalSherpaWhispeRuTranscriber.normalize_model_name(model_name)
        return "whisper", LocalWhisperSmallTranscriber.normalize_model_name(model_name)

    @staticmethod
    def _local_model_title(engine: str, model_name: str) -> str:
        engine_name = str(engine or "whisper").strip().lower()
        model = str(model_name or "").strip().lower()
        if engine_name == "sherpa":
            return "Voice PRO AI GigaAM RNNT"
        if engine_name == "vosk":
            if model == "vosk-model-ru-0.42":
                return "Vosk ru-0.42 (точнее)"
            return "Vosk small-ru (быстро)"
        if model in {"large-v3", "large"}:
            return "Whisper large-v3 (макс. качество)"
        if model == "medium":
            return "Whisper medium (точнее)"
        return "Whisper small (быстро)"

    def start_local_model_download(self, engine: str, model_name: str) -> tuple[bool, str]:
        engine_name, model = self._normalize_local_download_target(engine, model_name)
        model_title = self._local_model_title(engine_name, model)
        model_cls = (
            LocalVoskTranscriber
            if engine_name == "vosk"
            else LocalSherpaWhispeRuTranscriber
            if engine_name == "sherpa"
            else LocalWhisperSmallTranscriber
        )
        if model_cls.is_model_present(model):
            downloaded = model_cls.model_cached_size_bytes(model)
            self._set_model_download_status(
                state="done",
                engine=engine_name,
                model=model,
                progress_percent=100.0,
                downloaded_bytes=downloaded,
                total_bytes=downloaded,
                message=f"Модель {model_title} уже скачана",
                started_at=time.time(),
                finished_at=time.time(),
                error="",
            )
            return True, f"Модель {model_title} уже скачана"
        with self._model_dl_lock:
            running = self._model_dl_thread is not None and self._model_dl_thread.is_alive()
            if running:
                current = str(self._model_dl_status.get("model", "")).strip() or "другая"
                current_engine = str(self._model_dl_status.get("engine", "whisper") or "whisper").strip().lower()
                current_title = self._local_model_title(current_engine, current)
                return False, f"Сейчас уже скачивается модель {current_title}"
            self._model_dl_stop_event = threading.Event()
            self._model_dl_status = {
                "state": "preparing",
                "engine": engine_name,
                "model": model,
                "progress_percent": 0.0,
                "downloaded_bytes": model_cls.model_cached_size_bytes(model),
                "total_bytes": 0,
                "message": f"Подготовка загрузки модели {model_title}...",
                "started_at": time.time(),
                "finished_at": 0.0,
                "error": "",
            }

        def _monitor_loop(stop_event: threading.Event, target_engine: str, target_model: str) -> None:
            target_cls = (
                LocalVoskTranscriber
                if target_engine == "vosk"
                else LocalSherpaWhispeRuTranscriber
                if target_engine == "sherpa"
                else LocalWhisperSmallTranscriber
            )
            while not stop_event.is_set() and not self._model_dl_stop_event.is_set():
                status = self._get_model_download_status()
                if status.get("model") != target_model or str(status.get("engine", "whisper")) != target_engine:
                    return
                downloaded = target_cls.model_cached_size_bytes(target_model)
                total = int(status.get("total_bytes") or 0)
                percent = float(status.get("progress_percent") or 0.0)
                if total > 0:
                    percent = min(99.0, (downloaded / float(total)) * 100.0)
                else:
                    percent = min(99.0, percent + 1.0)
                self._set_model_download_status(
                    state="downloading",
                    downloaded_bytes=downloaded,
                    progress_percent=percent,
                    message=f"Скачивание модели {target_model}...",
                )
                stop_event.wait(0.45)

        def _worker(target_engine: str, target_model: str) -> None:
            target_cls = (
                LocalVoskTranscriber
                if target_engine == "vosk"
                else LocalSherpaWhispeRuTranscriber
                if target_engine == "sherpa"
                else LocalWhisperSmallTranscriber
            )
            monitor_stop = threading.Event()
            monitor_started = False
            monitor_thread = threading.Thread(
                target=_monitor_loop,
                args=(monitor_stop, target_engine, target_model),
                daemon=True,
            )
            try:
                total = self._estimate_repo_total_size_bytes(target_engine, target_model)
                self._set_model_download_status(
                    state="preparing",
                    total_bytes=total,
                    message=f"Проверка файлов модели {target_model}...",
                )
                if target_engine == "whisper":
                    monitor_thread.start()
                    monitor_started = True
                    target_cls.download_model_snapshot(target_model)
                else:
                    estimated_total = max(int(total or 0), 0)

                    def _engine_progress_cb(state: str, downloaded: int, total: int, message: str) -> None:
                        nonlocal estimated_total
                        if estimated_total <= 0:
                            estimated_total = int(self._estimate_repo_total_size_bytes(target_engine, target_model) or 0)
                        total_val = max(int(total or 0), int(downloaded or 0), int(estimated_total or 0))
                        progress = 0.0
                        if total_val > 0:
                            progress = min(99.0, (int(downloaded or 0) / float(total_val)) * 100.0)
                        self._set_model_download_status(
                            state="downloading",
                            downloaded_bytes=max(0, int(downloaded or 0)),
                            total_bytes=max(0, int(total_val)),
                            progress_percent=max(0.0, float(progress)),
                            message=str(message or f"Скачивание модели {target_model}..."),
                            error="",
                        )

                    target_cls.download_model_snapshot(target_model, progress_cb=_engine_progress_cb)
                monitor_stop.set()
                if monitor_started:
                    monitor_thread.join(timeout=1.2)
                downloaded = target_cls.model_cached_size_bytes(target_model)
                final_total = max(int(self._get_model_download_status().get("total_bytes") or 0), downloaded)
                self._set_model_download_status(
                    state="done",
                    engine=target_engine,
                    model=target_model,
                    progress_percent=100.0,
                    downloaded_bytes=downloaded,
                    total_bytes=final_total,
                    message=f"Модель {target_model} скачана",
                    finished_at=time.time(),
                    error="",
                )
            except Exception as err:
                monitor_stop.set()
                if monitor_started:
                    monitor_thread.join(timeout=1.0)
                logging.exception("Local model download failed: %s/%s", target_engine, target_model)
                self._set_model_download_status(
                    state="error",
                    message=f"Ошибка скачивания модели {target_model}",
                    finished_at=time.time(),
                    error=str(err),
                )
            finally:
                with self._model_dl_lock:
                    self._model_dl_thread = None

        thread = threading.Thread(
            target=_worker,
            args=(engine_name, model),
            name=f"ModelDownload-{engine_name}-{model}",
            daemon=True,
        )
        with self._model_dl_lock:
            self._model_dl_thread = thread
        thread.start()
        return True, f"Загрузка модели {model} запущена"

    def get_local_model_download_status(self, engine: str) -> dict:
        engine_name = str(engine or "whisper").strip().lower()
        if engine_name not in {"whisper", "vosk", "sherpa"}:
            engine_name = "whisper"
        status = self._get_model_download_status()
        if str(status.get("engine", "") or "").strip().lower() == "":
            status["engine"] = engine_name
        present, sizes = self._get_local_models_presence(engine_name)
        status["installed_models"] = present
        status["installed_sizes"] = sizes
        return status

    def _set_summary_model_download_status(self, **kwargs) -> None:
        with self._summary_model_dl_lock:
            self._summary_model_dl_status.update(kwargs)

    def _get_summary_model_download_status_raw(self) -> dict:
        with self._summary_model_dl_lock:
            return dict(self._summary_model_dl_status)

    def get_summary_model_download_status(self) -> dict:
        status = self._get_summary_model_download_status_raw()
        status["installed_models"] = LocalMeetingSummaryGGUF.list_installed_models()
        status["installed_sizes"] = LocalMeetingSummaryGGUF.list_installed_sizes()
        if not str(status.get("model", "")).strip():
            status["model"] = LocalMeetingSummaryGGUF.normalize_model_name(
                str(getattr(self.config, "meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
            )
        return status

    def start_summary_model_download(self, model_name: str) -> tuple[bool, str]:
        model_key = LocalMeetingSummaryGGUF.normalize_model_name(model_name)
        model_title = LocalMeetingSummaryGGUF.model_title(model_key)
        if LocalMeetingSummaryGGUF.is_model_present(model_key):
            downloaded = LocalMeetingSummaryGGUF.model_cached_size_bytes(model_key)
            self._set_summary_model_download_status(
                state="done",
                model=model_key,
                progress_percent=100.0,
                downloaded_bytes=downloaded,
                total_bytes=downloaded,
                message=f"Модель резюме {model_title} уже скачана",
                started_at=time.time(),
                finished_at=time.time(),
                error="",
            )
            return True, f"Модель резюме {model_title} уже скачана"
        with self._summary_model_dl_lock:
            running = self._summary_model_dl_thread is not None and self._summary_model_dl_thread.is_alive()
            if running:
                current = str(self._summary_model_dl_status.get("model", "")).strip() or LocalMeetingSummaryGGUF.DEFAULT_MODEL
                return False, f"Сейчас уже скачивается модель резюме {LocalMeetingSummaryGGUF.model_title(current)}"
            self._summary_model_dl_status = {
                "state": "preparing",
                "model": model_key,
                "progress_percent": 0.0,
                "downloaded_bytes": LocalMeetingSummaryGGUF.model_cached_size_bytes(model_key),
                "total_bytes": 0,
                "message": f"Подготовка загрузки модели резюме {model_title}...",
                "started_at": time.time(),
                "finished_at": 0.0,
                "error": "",
            }

        def _worker(target_model: str) -> None:
            try:
                estimate = LocalMeetingSummaryGGUF.estimate_total_size_bytes(target_model)
                self._set_summary_model_download_status(
                    state="preparing",
                    model=target_model,
                    total_bytes=max(0, int(estimate)),
                    message=f"Проверка файлов модели резюме {LocalMeetingSummaryGGUF.model_title(target_model)}...",
                    error="",
                )

                def _progress(state: str, downloaded: int, total: int, message: str) -> None:
                    total_val = max(int(total or 0), int(downloaded or 0))
                    progress = 0.0
                    if total_val > 0:
                        progress = min(99.0, (int(downloaded or 0) / float(total_val)) * 100.0)
                    self._set_summary_model_download_status(
                        state="downloading" if state != "done" else "done",
                        model=target_model,
                        downloaded_bytes=max(0, int(downloaded or 0)),
                        total_bytes=max(0, int(total_val)),
                        progress_percent=max(0.0, float(progress if state != "done" else 100.0)),
                        message=str(message or ""),
                        error="",
                    )

                LocalMeetingSummaryGGUF.download_model_snapshot(target_model, progress_cb=_progress)
                downloaded = LocalMeetingSummaryGGUF.model_cached_size_bytes(target_model)
                final_total = max(int(self._get_summary_model_download_status_raw().get("total_bytes") or 0), downloaded)
                self._set_summary_model_download_status(
                    state="done",
                    model=target_model,
                    progress_percent=100.0,
                    downloaded_bytes=downloaded,
                    total_bytes=final_total,
                    message=f"Модель резюме {LocalMeetingSummaryGGUF.model_title(target_model)} скачана",
                    finished_at=time.time(),
                    error="",
                )
            except Exception as err:
                logging.exception("Summary model download failed: %s", target_model)
                self._set_summary_model_download_status(
                    state="error",
                    model=target_model,
                    message=f"Ошибка скачивания модели резюме {LocalMeetingSummaryGGUF.model_title(target_model)}",
                    finished_at=time.time(),
                    error=str(err),
                )
            finally:
                with self._summary_model_dl_lock:
                    self._summary_model_dl_thread = None

        th = threading.Thread(target=_worker, args=(model_key,), daemon=True, name=f"SummaryModelDownload-{model_key}")
        with self._summary_model_dl_lock:
            self._summary_model_dl_thread = th
        th.start()
        return True, f"Загрузка модели резюме {model_title} запущена"

    def delete_summary_model(self, model_name: str) -> tuple[bool, str]:
        model_key = LocalMeetingSummaryGGUF.normalize_model_name(model_name)
        model_title = LocalMeetingSummaryGGUF.model_title(model_key)
        with self._summary_model_dl_lock:
            running = self._summary_model_dl_thread is not None and self._summary_model_dl_thread.is_alive()
            current = str(self._summary_model_dl_status.get("model", "")).strip().lower()
            if running and current == model_key:
                return False, f"Нельзя удалить модель {model_title}: сейчас идёт скачивание"
        try:
            removed = LocalMeetingSummaryGGUF.delete_model(model_key)
            with self._summary_llm_lock:
                if self._summary_llm_model_path and str(LocalMeetingSummaryGGUF.model_path(model_key)).lower() in self._summary_llm_model_path.lower():
                    self._summary_llm = None
                    self._summary_llm_model_path = ""
                    self._summary_llm_kind = ""
            if removed > 0:
                return True, f"Модель резюме {model_title} удалена"
            return True, f"Кэш модели резюме {model_title} не найден"
        except Exception as err:
            logging.exception("Failed to delete summary model: %s", model_key)
            return False, f"Ошибка удаления модели резюме {model_title}: {err}"

    def _warm_up_audio_stream(self) -> None:
        try:
            self.recorder.ensure_stream()
        except Exception:
            # Non-fatal at startup: we'll retry on first push-to-talk press.
            logging.exception("Audio warm-up failed")

    def _acquire_single_instance(self) -> bool:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.CreateMutexW(None, False, MUTEX_NAME)
            if not handle:
                logging.error("CreateMutexW failed, err=%s", ctypes.get_last_error())
                return True
            self._mutex_handle = handle
            already_exists = ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
            if already_exists:
                logging.info("Another instance is already running")
                self._send_ipc_open_settings_signal()
                return False
            return True
        except Exception:
            # If mutex creation fails, continue rather than blocking app start.
            logging.exception("Single-instance mutex failed")
            return True

    def _send_ipc_open_settings_signal(self) -> None:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            token = f"open_settings|{time.time():.6f}|{os.getpid()}"
            IPC_COMMAND_PATH.write_text(token, encoding="utf-8")
        except Exception:
            logging.exception("Failed to write IPC command")

    def _start_ipc_listener(self) -> None:
        if self._ipc_thread and self._ipc_thread.is_alive():
            return

        self._ipc_stop_event.clear()

        def _loop() -> None:
            while not self._ipc_stop_event.is_set():
                try:
                    if IPC_COMMAND_PATH.exists():
                        token = IPC_COMMAND_PATH.read_text(encoding="utf-8").strip()
                        if token and token != self._ipc_last_token:
                            self._ipc_last_token = token
                            parts = token.split("|")
                            if len(parts) >= 2 and parts[0] == "open_settings":
                                ts = float(parts[1])
                                # Ignore stale commands left from previous sessions.
                                if (time.time() - ts) <= 15.0:
                                    self._on_open_settings(None, None)
                except Exception:
                    logging.exception("IPC listener error")
                finally:
                    self._ipc_stop_event.wait(0.35)

        self._ipc_thread = threading.Thread(target=_loop, name="VoiceProIPC", daemon=True)
        self._ipc_thread.start()

    def _setup_logging(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            filename=LOG_PATH,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8",
        )

    @staticmethod
    def _set_app_user_model_id() -> None:
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except Exception:
            # Non-fatal; app can run without this, but icon grouping may be worse.
            pass

    def _load_icon_image(self) -> Image.Image:
        icon_ico_path = self.base_dir / "icon-128.ico"
        if icon_ico_path.exists():
            return Image.open(icon_ico_path)

        icon_path = self.base_dir / "icon-128.png"
        if icon_path.exists():
            return Image.open(icon_path)

        img = Image.new("RGBA", (64, 64), (30, 99, 230, 255))
        return img

    def _create_tray(self) -> None:
        menu = Menu(
            MenuItem("Настройки", self._on_open_settings),
            MenuItem("Проверить обновления", self._on_check_updates_tray),
            MenuItem("Копировать последнюю расшифровку", self._on_copy_last),
            MenuItem("Выход", self._on_quit),
        )
        self._icon = Icon(APP_NAME, self._load_icon_image(), APP_NAME, menu)

    def _notify(self, message: str) -> None:
        safe_message = str(message or "").strip()
        if len(safe_message) > 240:
            safe_message = safe_message[:237] + "..."
        if self.config.play_notification_sounds and winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_OK)
            except Exception:
                logging.exception("Notification sound failed")
        if self._icon is None:
            return
        try:
            self._icon.notify(safe_message, APP_NAME)
        except Exception:
            logging.exception("Tray notify failed")

    def _on_check_updates_tray(self, _icon: Optional[Icon], _item: Optional[MenuItem]) -> None:
        def _worker() -> None:
            ok, message, _snap, update_info = self._check_for_updates(notify_if_latest=True)
            if update_info:
                ver = str(update_info.get("version", "") or "").strip()
                if ver:
                    self._notify(f"Доступно обновление {ver}. Откройте Настройки и нажмите «Проверить обновления».")
            elif not ok:
                self._notify(message or "Ошибка проверки обновлений")

        threading.Thread(target=_worker, name="VoiceProCheckUpdateTray", daemon=True).start()

    def _bind_hotkey(self) -> None:
        with self._hotkey_lock:
            self._unbind_hotkey()
            key = self.config.ptt_key.lower().strip() or "f8"
            parsed_keys = self._parse_hotkey_keys(key)
            if not parsed_keys:
                parsed_keys = {"f8"}
            key = self._format_hotkey(parsed_keys)
            try:
                self._register_hotkey_combo(key)
                self.config.ptt_key = key
                logging.info("PTT key bound: %s", key)
            except Exception:
                logging.exception("Failed to bind hotkey: %s", key)
                # Startup safety: fallback to F8 so first launch never crashes.
                fallback = "f8"
                self._unbind_hotkey()
                try:
                    self._register_hotkey_combo(fallback)
                    self.config.ptt_key = fallback
                    self.config_store.save(self.config)
                    logging.info("PTT fallback key bound: %s", fallback)
                except Exception:
                    logging.exception("Failed to bind fallback hotkey")

    def _register_hotkey_combo(self, key: str) -> None:
        self._ptt_combo_keys = self._parse_hotkey_keys(key)
        self._pressed_keys = set()
        self._ptt_armed = False
        for k in self._ptt_combo_keys:
            bound_for_key = False
            for variant in self._hotkey_key_variants(k):
                try:
                    down_handle = keyboard.on_press_key(variant, lambda e, kk=k: self._on_hotkey_press(kk), suppress=False)
                    up_handle = keyboard.on_release_key(variant, lambda e, kk=k: self._on_hotkey_release(kk), suppress=False)
                    self._hotkey_handles.extend([down_handle, up_handle])
                    bound_for_key = True
                except Exception:
                    # Some non-latin layout aliases may be unavailable on specific machines.
                    # Keep binding canonical key and continue gracefully.
                    logging.debug("Skip hotkey variant bind: %s", variant)
            if not bound_for_key:
                # Fallback: always try canonical key as last resort.
                down_handle = keyboard.on_press_key(k, lambda e, kk=k: self._on_hotkey_press(kk), suppress=False)
                up_handle = keyboard.on_release_key(k, lambda e, kk=k: self._on_hotkey_release(kk), suppress=False)
                self._hotkey_handles.extend([down_handle, up_handle])

    def _unbind_hotkey(self) -> None:
        for handle in self._hotkey_handles:
            try:
                keyboard.unhook(handle)
            except KeyError:
                # Already removed by keyboard internals; safe to ignore.
                pass
            except Exception:
                logging.exception("Failed to unhook keyboard handle")
        self._hotkey_handles = []

    def _on_hotkey_press(self, key_name: str) -> None:
        if self._settings_open:
            # Self-heal: if settings flag is stale, unblock hotkey automatically.
            if self._is_settings_window_foreground():
                return
            with self._hotkey_lock:
                self._settings_open = False
                self._pressed_keys = set()
                self._ptt_armed = False
        if time.monotonic() < self._ignore_hotkey_until:
            return
        self._pressed_keys.add(key_name)
        if not self._ptt_armed and self._ptt_combo_keys.issubset(self._pressed_keys):
            self._ptt_armed = True
            if self.config.activation_mode == "toggle":
                if self._recording:
                    self._on_ptt_up()
                else:
                    self._on_ptt_down()
            else:
                self._on_ptt_down()

    def _on_hotkey_release(self, key_name: str) -> None:
        self._pressed_keys.discard(key_name)
        if self._ptt_armed and self.config.activation_mode != "toggle":
            self._ptt_armed = False
            self._on_ptt_up()
        elif self.config.activation_mode == "toggle":
            # Rearm only after full combo release to avoid repeat-trigger on key repeats.
            if not self._ptt_combo_keys.intersection(self._pressed_keys):
                self._ptt_armed = False

    @staticmethod
    def _normalize_key_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        n = name.strip().lower()
        # Fix occasional mojibake names produced by some keyboard hooks/logging paths.
        if any(ch in n for ch in ("р", "с", "ё", "й")) and len(n) <= 4:
            try:
                repaired = n.encode("cp1251", errors="strict").decode("utf-8", errors="strict").strip().lower()
                if repaired:
                    n = repaired
            except Exception:
                pass
        aliases = {
            "left ctrl": "ctrl",
            "right ctrl": "ctrl",
            "left alt": "alt",
            "right alt": "alt",
            "alt gr": "alt",
            "left shift": "shift",
            "right shift": "shift",
            "left windows": "windows",
            "right windows": "windows",
            "win": "windows",
            "command": "windows",
            "cmd": "windows",
            "return": "enter",
        }
        n = aliases.get(n, n)
        ru_to_en = {
            "й": "q", "ц": "w", "у": "e", "к": "r", "е": "t", "н": "y", "г": "u", "ш": "i", "щ": "o", "з": "p",
            "ф": "a", "ы": "s", "в": "d", "а": "f", "п": "g", "р": "h", "о": "j", "л": "k", "д": "l",
            "я": "z", "ч": "x", "с": "c", "м": "v", "и": "b", "т": "n", "ь": "m",
        }
        return ru_to_en.get(n, n)

    @staticmethod
    def _hotkey_key_variants(canonical_key: str) -> set[str]:
        key = str(canonical_key or "").strip().lower()
        if not key:
            return set()
        en_to_ru = {
            "q": "й", "w": "ц", "e": "у", "r": "к", "t": "е", "y": "н", "u": "г", "i": "ш", "o": "щ", "p": "з",
            "a": "ф", "s": "ы", "d": "в", "f": "а", "g": "п", "h": "р", "j": "о", "k": "л", "l": "д",
            "z": "я", "x": "ч", "c": "с", "v": "м", "b": "и", "n": "т", "m": "ь",
        }
        variants = {key}
        ru = en_to_ru.get(key)
        if ru:
            variants.add(ru)
        return variants

    def _parse_hotkey_keys(self, hotkey: str) -> set[str]:
        keys = [self._normalize_key_name(part) for part in hotkey.split("+")]
        clean = {k for k in keys if k}
        return clean

    @staticmethod
    def _format_hotkey(keys: set[str]) -> str:
        if not keys:
            return ""
        order = {"ctrl": 0, "alt": 1, "shift": 2, "windows": 3}
        return "+".join(sorted(keys, key=lambda k: (order.get(k, 10), k)))

    @staticmethod
    def _normalize_api_key_input(raw_value: object) -> str:
        key = str(raw_value or "").strip()
        if not key:
            return ""
        key = key.strip("\"'В«В»")
        key = re.sub(r"\s+", "", key)
        # Guard against accidental double-paste where key is repeated twice.
        if len(key) % 2 == 0:
            half = len(key) // 2
            if key[:half] == key[half:]:
                key = key[:half]
        return key

    def _on_ptt_down(self) -> None:
        if self._busy or self._recording:
            return

        try:
            supports_preview = False
            try:
                supports_preview = bool(self.transcriber.supports_live_preview())
            except Exception:
                supports_preview = False
            self._target_hwnd = self._get_foreground_window()
            self.recorder.start_recording()
            self._recording = True
            if self.config.show_overlay_widget:
                self.overlay.set_live_preview_enabled(bool(self.config.show_live_preview_text and supports_preview))
                self.overlay.set_mic_quality_enabled(bool(self.config.show_mic_quality_indicator))
                self.overlay.show("recording")
                if self.config.show_live_preview_text and supports_preview:
                    self.overlay.clear_live_preview_text()
                if self.config.show_mic_quality_indicator:
                    self.overlay.set_mic_quality_text("Уровень: 0% · Слишком тихо")
                else:
                    self.overlay.clear_mic_quality_text()
            self._start_live_preview_loop()
            self._start_mic_quality_loop()
            if self.config.play_recording_sounds and winsound is not None:
                try:
                    winsound.MessageBeep(winsound.MB_ICONASTERISK)
                except Exception:
                    logging.exception("Start recording sound failed")
            self._notify("Идёт запись...")
        except Exception as err:
            logging.exception("Failed to start recording")
            if self.config.show_overlay_widget:
                self.overlay.show("mic_off")
            self._notify(f"Микрофон недоступен: {err}")
            if self.config.show_overlay_widget:
                threading.Thread(target=lambda: (time.sleep(2.2), self.overlay.hide()), daemon=True).start()

    def _on_ptt_up(self) -> None:
        if not self._recording or self._busy:
            return

        self._recording = False
        self._stop_live_preview_loop()
        self._stop_mic_quality_loop()
        audio, duration = self.recorder.stop_recording()
        if self.config.play_recording_sounds and winsound is not None:
            try:
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
            except Exception:
                logging.exception("Stop recording sound failed")
        if audio is None or duration < 0.2:
            if self.config.show_overlay_widget:
                self.overlay.clear_live_preview_text()
                self.overlay.clear_mic_quality_text()
                self.overlay.hide()
            return

        if not self._has_audio_signal(audio):
            if self.config.show_overlay_widget:
                self.overlay.clear_live_preview_text()
                self.overlay.clear_mic_quality_text()
                self.overlay.show("mic_off")
            if self.config.notify_on_no_sound:
                self._notify("Микрофон выключен или нет входного сигнала")
            if self.config.show_overlay_widget:
                threading.Thread(target=lambda: (time.sleep(2.2), self.overlay.hide()), daemon=True).start()
            return

        if self.config.show_overlay_widget:
            self.overlay.clear_live_preview_text()
            self.overlay.clear_mic_quality_text()
            self.overlay.show("transcribing")
        thread = threading.Thread(target=self._transcribe_and_output, args=(audio, duration), daemon=True)
        thread.start()

    def _start_live_preview_loop(self) -> None:
        if not self.config.show_live_preview_text:
            return
        if not self.config.show_overlay_widget:
            return
        supports_preview = False
        try:
            supports_preview = bool(self.transcriber.supports_live_preview())
        except Exception:
            supports_preview = False
        if not supports_preview:
            return
        if self._live_preview_thread and self._live_preview_thread.is_alive():
            return
        self._live_preview_stop_event.clear()
        self._live_preview_thread = threading.Thread(target=self._live_preview_worker, daemon=True)
        self._live_preview_thread.start()

    def _stop_live_preview_loop(self) -> None:
        self._live_preview_stop_event.set()
        th = self._live_preview_thread
        self._live_preview_thread = None
        if th and th.is_alive():
            th.join(timeout=0.25)

    def _start_mic_quality_loop(self) -> None:
        if not self.config.show_overlay_widget:
            return
        if not self.config.show_mic_quality_indicator:
            return
        if self._mic_quality_thread and self._mic_quality_thread.is_alive():
            return
        self._mic_quality_stop_event.clear()
        self._mic_quality_thread = threading.Thread(target=self._mic_quality_worker, daemon=True)
        self._mic_quality_thread.start()

    def _stop_mic_quality_loop(self) -> None:
        self._mic_quality_stop_event.set()
        th = self._mic_quality_thread
        self._mic_quality_thread = None
        if th and th.is_alive():
            th.join(timeout=0.25)

    def _mic_quality_worker(self) -> None:
        last_text = ""
        while not self._mic_quality_stop_event.is_set():
            try:
                if not self._recording:
                    break
                level_pct, state = self.recorder.get_signal_quality()
                if state == "overload":
                    text = f"Уровень: {level_pct}% · Перегруз"
                elif state == "quiet":
                    text = f"Уровень: {level_pct}% · Слишком тихо"
                else:
                    text = f"Уровень: {level_pct}% · Норма"
                if text != last_text:
                    last_text = text
                    self.overlay.set_mic_quality_text(text)
            except Exception:
                logging.exception("Mic quality worker failed")
                break
            time.sleep(0.12)

    def _live_preview_worker(self) -> None:
        last_sent_duration = 0.0
        last_text = ""
        failure_count = 0
        empty_count = 0
        while not self._live_preview_stop_event.is_set():
            try:
                if not self._recording:
                    break
                audio, duration = self.recorder.get_recording_snapshot()
                if audio is None or duration < 0.8:
                    time.sleep(0.12)
                    continue
                if duration - last_sent_duration < 0.75:
                    time.sleep(0.12)
                    continue
                if not self._has_audio_signal(audio):
                    time.sleep(0.1)
                    continue
                preview_audio, preview_rate = self._prepare_preview_audio_for_realtime(audio, self.config.sample_rate)
                wav_bytes = float_audio_to_wav_bytes(preview_audio, preview_rate)
                preview_text = self.transcriber.transcribe_live_preview(wav_bytes, duration)
                if preview_text:
                    empty_count = 0
                    if preview_text != last_text:
                        last_text = preview_text
                        self.overlay.set_live_preview_text(preview_text)
                    failure_count = 0
                else:
                    empty_count += 1
                    if empty_count >= 6:
                        self.overlay.set_live_preview_text("Слушаю речь...")
                last_sent_duration = duration
            except Exception:
                logging.exception("Live preview worker failed")
                failure_count += 1
                if failure_count >= 3:
                    # Do not spam network/API when connection is unstable.
                    self.overlay.set_live_preview_text("Проблема сети. Продолжаю запись без live-текста...")
                    break
            time.sleep(0.1)

    @staticmethod
    def _prepare_preview_audio_for_realtime(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        if audio.size <= 0:
            return audio, sample_rate

        src_rate = max(8000, int(sample_rate))
        max_seconds = 12.0
        max_len = int(src_rate * max_seconds)
        tail = audio[-max_len:] if audio.size > max_len else audio
        # Keep original sample-rate for better recognition fidelity in live mode.
        return tail, src_rate

    @staticmethod
    def _has_audio_signal(samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        # Heuristic for "no signal" (hardware mute / dead silence).
        abs_samples = np.abs(samples)
        peak = float(np.max(abs_samples))
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return not (peak < 0.003 and rms < 0.0007)

    def _transcribe_and_output(self, audio: np.ndarray, duration: float) -> None:
        if self._busy:
            return
        self._busy = True
        pipeline_t0 = time.perf_counter()
        transcribe_ms = 0.0
        refine_ms = 0.0
        try:
            self._notify("Идёт распознавание...")
            transcribe_t0 = time.perf_counter()
            if isinstance(self.transcriber, (LocalWhisperSmallTranscriber, LocalVoskTranscriber, LocalSherpaWhispeRuTranscriber)):
                text, usage = self.transcriber.transcribe(audio, duration, self.config.sample_rate)
            else:
                wav_bytes = float_audio_to_wav_bytes(audio, self.config.sample_rate)
                text, usage = self.transcriber.transcribe(wav_bytes, duration)
            transcribe_ms = (time.perf_counter() - transcribe_t0) * 1000.0
            if not text:
                self._notify("Речь не распознана")
                return
            text, refine_usage, refine_ms = self._fast_refine_local_text(text, duration)
            usage = self._merge_usage(usage, refine_usage)

            text = self._apply_autoreplace(text)
            self._update_stats(text, duration, usage)
            self._append_history_entry(text)
            self._set_last_transcript(text)

            if self.config.auto_paste:
                text_for_paste = self._prepare_text_for_paste(text, self._target_hwnd)
                ok = self._paste_text(text_for_paste, self._target_hwnd)
                if ok:
                    self._has_successful_insert = True
                    self._last_insert_target_hwnd = self._target_hwnd
                    if self.config.copy_result_to_clipboard:
                        threading.Thread(target=lambda: (time.sleep(0.45), self._copy_text(text)), daemon=True).start()
                    self._notify("Текст вставлен")
                else:
                    self._notify("Текст скопирован. Нажмите Ctrl+V")
            else:
                self._copy_text(text)
                self._notify("Текст скопирован")
        except Exception as err:
            logging.exception("Transcription pipeline failed")
            message = str(err)
            if "User not found" in message:
                self._notify("Ошибка: неверный OpenRouter API key. Проверьте настройки.")
            elif "faster-whisper" in message.lower():
                self._notify("Ошибка локального режима: не найден faster-whisper. Переустановите приложение.")
            elif "sherpa" in message.lower():
                self._notify("Ошибка локального режима: не найден/не загружен Sherpa. Проверьте модель в Настройках.")
            else:
                self._notify(f"Ошибка: {err}")
        finally:
            total_ms = (time.perf_counter() - pipeline_t0) * 1000.0
            with self._perf_lock:
                self._perf_window_total_ms.append(float(total_ms))
                rolling_avg_ms = (sum(self._perf_window_total_ms) / float(len(self._perf_window_total_ms))) if self._perf_window_total_ms else total_ms
            logging.info(
                "Pipeline timing: transcribe_ms=%.1f refine_ms=%.1f total_ms=%.1f rolling_avg20_ms=%.1f",
                float(transcribe_ms),
                float(refine_ms),
                float(total_ms),
                float(rolling_avg_ms),
            )
            self._busy = False
            if self.config.show_overlay_widget:
                self.overlay.clear_live_preview_text()
                self.overlay.hide()

    @staticmethod
    def _merge_usage(base: Optional[dict], extra: Optional[dict]) -> dict:
        b = dict(base or {})
        e = dict(extra or {})
        try:
            in_tokens = int(b.get("input_tokens", 0) or 0) + int(e.get("input_tokens", 0) or 0)
        except Exception:
            in_tokens = 0
        try:
            out_tokens = int(b.get("output_tokens", 0) or 0) + int(e.get("output_tokens", 0) or 0)
        except Exception:
            out_tokens = 0
        return {"input_tokens": max(0, in_tokens), "output_tokens": max(0, out_tokens)}

    def _fast_refine_local_text(self, text: str, duration_sec: float) -> tuple[str, dict, float]:
        # Optional hybrid refinement:
        # local ASR stays primary, OpenRouter is used only for ultra-fast punctuation/ending correction.
        # If network/API is slow or unavailable, we instantly keep local output.
        if not isinstance(self.transcriber, (LocalWhisperSmallTranscriber, LocalVoskTranscriber, LocalSherpaWhispeRuTranscriber)):
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        if not bool(getattr(self.config, "local_fast_refine", True)):
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        api_key = str(getattr(self.config, "openrouter_api_key", "") or "").strip()
        if not api_key:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        src = str(text or "").strip()
        if not src:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        is_vosk_local = isinstance(self.transcriber, LocalVoskTranscriber)
        is_medium_turbo = bool(
            isinstance(self.transcriber, LocalWhisperSmallTranscriber)
            and str(getattr(self.transcriber, "_selected_model", "")).strip().lower() == "medium"
            and bool(getattr(self.config, "whisper_medium_turbo", False))
        )
        is_sherpa_local = isinstance(self.transcriber, LocalSherpaWhispeRuTranscriber)
        # Keep this stage fast by default. For Vosk allow a bit longer chunks,
        # because punctuation/ending correction is often required.
        max_chars = 2200 if (is_vosk_local or is_sherpa_local) else (1100 if is_medium_turbo else 900)
        max_duration = 45.0 if (is_vosk_local or is_sherpa_local) else (16.0 if is_medium_turbo else 20.0)
        if len(src) > max_chars or float(duration_sec or 0.0) > max_duration:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0

        t0 = time.perf_counter()
        payload = {
            "model": str(self.config.model or "google/gemini-2.5-flash"),
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        (
                            "Исправь ошибки распознавания в русском тексте: орфография, окончания слов и пунктуация "
                            "(включая ?, !, .). Не сокращай и не перефразируй. Сохрани смысл и порядок. "
                            "Верни только исправленный текст."
                            if not is_vosk_local
                            else
                            "Ты корректируешь текст после офлайн-распознавания Vosk. "
                            "Исправь окончания слов, грамматику и расставь естественные знаки препинания "
                            "(обязательно корректно ? и !, а также запятые и точки). "
                            "Не меняй смысл, не сокращай, не добавляй нового. Верни только итоговый текст."
                        )
                    ),
                },
                {"role": "user", "content": src},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            # Turbo medium: strict refine budget to preserve end-to-end SLA.
            if is_medium_turbo:
                timeout_cfg = (0.6, 2.8)
            elif is_vosk_local:
                timeout_cfg = (1.0, 8.0)
            else:
                timeout_cfg = (1.0, 6.0)
            resp = self._refine_http.post(
                OpenRouterTranscriber.API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_cfg,
            )
            data = resp.json() if resp.content else {}
            if not resp.ok:
                raise RuntimeError(str(data.get("error", {}).get("message") or f"HTTP {resp.status_code}"))
            refined = OpenRouterTranscriber._extract_text(data)
            if not refined:
                return text, {"input_tokens": 0, "output_tokens": 0}, (time.perf_counter() - t0) * 1000.0
            refined = self.transcriber._postprocess_text(refined)
            usage = OpenRouterTranscriber._extract_usage(data)
            t1 = time.perf_counter()
            logging.info("Fast local refine timing: chars=%s total_ms=%.1f", len(src), (t1 - t0) * 1000.0)
            return refined, usage, (t1 - t0) * 1000.0
        except Exception as err:
            logging.debug("Fast local refine skipped: %s", err)
            return text, {"input_tokens": 0, "output_tokens": 0}, (time.perf_counter() - t0) * 1000.0

    def _fast_refine_local_text_for_context(
        self,
        text: str,
        duration_sec: float,
        cfg: AppConfig,
        transcriber_obj,
    ) -> tuple[str, dict, float]:
        if not isinstance(transcriber_obj, (LocalWhisperSmallTranscriber, LocalVoskTranscriber, LocalSherpaWhispeRuTranscriber)):
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        if not bool(getattr(cfg, "local_fast_refine", True)):
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        api_key = str(getattr(cfg, "openrouter_api_key", "") or "").strip()
        if not api_key:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0
        src = str(text or "").strip()
        if not src:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0

        is_vosk_local = isinstance(transcriber_obj, LocalVoskTranscriber)
        is_sherpa_local = isinstance(transcriber_obj, LocalSherpaWhispeRuTranscriber)
        is_medium_turbo = bool(
            isinstance(transcriber_obj, LocalWhisperSmallTranscriber)
            and str(getattr(transcriber_obj, "_selected_model", "")).strip().lower() == "medium"
            and bool(getattr(cfg, "whisper_medium_turbo", False))
        )
        max_chars = 2200 if (is_vosk_local or is_sherpa_local) else (1100 if is_medium_turbo else 900)
        max_duration = 45.0 if (is_vosk_local or is_sherpa_local) else (16.0 if is_medium_turbo else 20.0)
        if len(src) > max_chars or float(duration_sec or 0.0) > max_duration:
            return text, {"input_tokens": 0, "output_tokens": 0}, 0.0

        t0 = time.perf_counter()
        payload = {
            "model": str(getattr(cfg, "model", "google/gemini-2.5-flash") or "google/gemini-2.5-flash"),
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        (
                            "Исправь ошибки распознавания в русском тексте: орфография, окончания слов и пунктуация "
                            "(включая ?, !, .). Не сокращай и не перефразируй. Сохрани смысл и порядок. "
                            "Верни только исправленный текст."
                            if not is_vosk_local
                            else
                            "Ты корректируешь текст после офлайн-распознавания Vosk. "
                            "Исправь окончания слов, грамматику и расставь естественные знаки препинания "
                            "(обязательно корректно ? и !, а также запятые и точки). "
                            "Не меняй смысл, не сокращай, не добавляй нового. Верни только итоговый текст."
                        )
                    ),
                },
                {"role": "user", "content": src},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            if is_medium_turbo:
                timeout_cfg = (0.6, 2.8)
            elif is_vosk_local:
                timeout_cfg = (1.0, 8.0)
            else:
                timeout_cfg = (1.0, 6.0)
            resp = self._refine_http.post(
                OpenRouterTranscriber.API_URL,
                headers=headers,
                json=payload,
                timeout=timeout_cfg,
            )
            data = resp.json() if resp.content else {}
            if not resp.ok:
                raise RuntimeError(str(data.get("error", {}).get("message") or f"HTTP {resp.status_code}"))
            refined = OpenRouterTranscriber._extract_text(data)
            if not refined:
                return text, {"input_tokens": 0, "output_tokens": 0}, (time.perf_counter() - t0) * 1000.0
            if hasattr(transcriber_obj, "_postprocess_text"):
                refined = transcriber_obj._postprocess_text(refined)
            usage = OpenRouterTranscriber._extract_usage(data)
            t1 = time.perf_counter()
            return refined, usage, (t1 - t0) * 1000.0
        except Exception as err:
            logging.debug("Fast local refine (media) skipped: %s", err)
            return text, {"input_tokens": 0, "output_tokens": 0}, (time.perf_counter() - t0) * 1000.0

    def _apply_autoreplace_with_enabled(self, text: str, enabled: bool) -> str:
        if not bool(enabled):
            return text
        out = str(text or "")
        with self._rules_lock:
            rules = list(self._autoreplace_rules_cache)
        for rule in rules:
            try:
                if not bool(rule.get("enabled", True)):
                    continue
                pattern = str(rule.get("pattern", ""))
                if not pattern:
                    continue
                replacement = str(rule.get("replacement", ""))
                mode = str(rule.get("mode", "Text")).lower()
                if mode in {"regexp", "regex", "rx"}:
                    flags_value = 0
                    flags = str(rule.get("flags", "u")).lower()
                    if "i" in flags:
                        flags_value |= re.IGNORECASE
                    if "m" in flags:
                        flags_value |= re.MULTILINE
                    if "s" in flags:
                        flags_value |= re.DOTALL
                    if "u" in flags:
                        flags_value |= re.UNICODE
                    out = re.sub(pattern, replacement, out, flags=flags_value)
                else:
                    out = out.replace(pattern, replacement)
            except Exception:
                logging.exception("AutoReplace rule failed in media mode: %s", rule.get("id", "unknown"))
        return out

    def _next_media_item_id(self) -> str:
        self._media_id_counter += 1
        return f"media-{int(time.time() * 1000)}-{self._media_id_counter}"

    def _get_media_queue_snapshot(self) -> dict:
        with self._media_lock:
            items = [dict(item) for item in self._media_queue_items]
            active_id = str(self._media_active_id or "")
            processing = bool(self._media_is_processing)
        return {"items": items, "active_id": active_id, "is_processing": processing}

    @staticmethod
    def _safe_media_name(path: Path) -> str:
        name = str(path.name or "media").strip()
        return name or "media"

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        raw = str(name or "").strip()
        raw = re.sub(r"[\\/:*?\"<>|]+", "_", raw)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip(" ._") or "result"

    @staticmethod
    def _format_srt_timestamp(total_seconds: float) -> str:
        sec = max(0.0, float(total_seconds or 0.0))
        hours = int(sec // 3600)
        sec -= hours * 3600
        minutes = int(sec // 60)
        sec -= minutes * 60
        seconds = int(sec)
        millis = int(round((sec - seconds) * 1000.0))
        if millis >= 1000:
            seconds += 1
            millis = 0
        if seconds >= 60:
            minutes += 1
            seconds = 0
        if minutes >= 60:
            hours += 1
            minutes = 0
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

    @staticmethod
    def _build_srt_from_text(text: str, duration_sec: float) -> str:
        src = str(text or "").strip()
        if not src:
            return ""
        chunks = [part.strip() for part in re.split(r"(?<=[.!?…])\s+", src) if str(part).strip()]
        if not chunks:
            chunks = [src]
        total_duration = max(float(duration_sec or 0.0), 0.8)
        segment_duration = max(0.35, total_duration / max(1, len(chunks)))
        lines: list[str] = []
        cursor = 0.0
        for idx, chunk in enumerate(chunks, start=1):
            start = cursor
            end = min(total_duration, start + segment_duration)
            if idx == len(chunks):
                end = total_duration
            lines.append(str(idx))
            lines.append(f"{VoiceProApp._format_srt_timestamp(start)} --> {VoiceProApp._format_srt_timestamp(end)}")
            lines.append(chunk)
            lines.append("")
            cursor = end
        return "\n".join(lines).strip() + "\n"

    def _decode_media_to_audio(self, media_path: Path, target_sample_rate: int) -> tuple[np.ndarray, float]:
        sample_rate = max(8000, int(target_sample_rate))
        if av is None:
            if media_path.suffix.lower() == ".wav":
                with wave.open(str(media_path), "rb") as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    sr = max(8000, int(wav_file.getframerate()))
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    if wav_file.getnchannels() > 1 and audio.size:
                        audio = audio.reshape(-1, wav_file.getnchannels()).mean(axis=1).astype(np.float32)
                    if sr != sample_rate and audio.size:
                        src_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float32)
                        target_len = max(1, int(round(audio.size * (sample_rate / float(sr)))))
                        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float32)
                        audio = np.interp(dst_x, src_x, audio).astype(np.float32, copy=False)
            else:
                raise RuntimeError("Для обработки видео/аудио установите пакет PyAV.")
        else:
            container = av.open(str(media_path))
            try:
                stream = None
                for st in container.streams:
                    if getattr(st, "type", "") == "audio":
                        stream = st
                        break
                if stream is None:
                    raise RuntimeError("В файле не найдена аудио-дорожка.")
                resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
                chunks: list[np.ndarray] = []
                for frame in container.decode(stream):
                    converted = resampler.resample(frame)
                    if converted is None:
                        continue
                    frames = converted if isinstance(converted, list) else [converted]
                    for out_frame in frames:
                        arr = out_frame.to_ndarray()
                        if arr is None:
                            continue
                        if arr.ndim == 2:
                            arr = arr[0]
                        chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
                flushed = resampler.resample(None)
                if flushed is not None:
                    flush_frames = flushed if isinstance(flushed, list) else [flushed]
                    for out_frame in flush_frames:
                        arr = out_frame.to_ndarray()
                        if arr is None:
                            continue
                        if arr.ndim == 2:
                            arr = arr[0]
                        chunks.append(np.asarray(arr, dtype=np.float32).reshape(-1))
            finally:
                try:
                    container.close()
                except Exception:
                    pass
            if not chunks:
                raise RuntimeError("Аудио-дорожка пуста или не может быть декодирована.")
            audio = np.concatenate(chunks).astype(np.float32, copy=False)

        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size <= 0:
            raise RuntimeError("Файл не содержит пригодного звука.")
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32, copy=False)
        duration = float(audio.size) / float(sample_rate)
        return audio, max(0.0, duration)

    def _process_one_media_item(self, item: dict, cfg: AppConfig, transcriber_obj) -> tuple[str, dict, float]:
        path = Path(str(item.get("path", "")).strip())
        if not path.exists():
            raise RuntimeError("Файл не найден на диске.")
        audio, duration = self._decode_media_to_audio(path, int(getattr(cfg, "sample_rate", 24000)))
        if duration < 0.2 or not self._has_audio_signal(audio):
            raise RuntimeError("Не удалось обнаружить речь в файле.")

        if isinstance(transcriber_obj, LocalSherpaWhispeRuTranscriber):
            text, usage = self._transcribe_media_audio_sherpa_chunked(audio, duration, int(getattr(cfg, "sample_rate", 24000)), transcriber_obj)
        elif isinstance(transcriber_obj, (LocalWhisperSmallTranscriber, LocalVoskTranscriber)):
            text, usage = transcriber_obj.transcribe(audio, duration, int(getattr(cfg, "sample_rate", 24000)))
            text, refine_usage, _ = self._fast_refine_local_text_for_context(text, duration, cfg, transcriber_obj)
            usage = self._merge_usage(usage, refine_usage)
        else:
            wav_bytes = float_audio_to_wav_bytes(audio, int(getattr(cfg, "sample_rate", 24000)))
            text, usage = transcriber_obj.transcribe(wav_bytes, duration)
        normalized = self._apply_autoreplace_with_enabled(str(text or "").strip(), bool(getattr(cfg, "autoreplace_enabled", True)))
        return normalized, dict(usage or {}), duration

    def _transcribe_media_audio_sherpa_chunked(
        self,
        audio: np.ndarray,
        duration_sec: float,
        sample_rate: int,
        transcriber_obj: LocalSherpaWhispeRuTranscriber,
    ) -> tuple[str, dict]:
        # Sherpa offline RNNT can fail on long utterances in one pass (ONNX shape issues).
        # For media queue we transcribe safely in short overlapping chunks and merge.
        sr = max(8000, int(sample_rate))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            return "", {"input_tokens": 0, "output_tokens": 0}
        chunk_sec = 7.5
        overlap_sec = 0.35
        chunk_samples = max(1, int(chunk_sec * sr))
        overlap_samples = max(0, int(overlap_sec * sr))
        step = max(1, chunk_samples - overlap_samples)
        parts: list[str] = []
        last_err: Optional[Exception] = None
        start = 0
        loops = 0
        while start < audio.size:
            end = min(audio.size, start + chunk_samples)
            segment = audio[start:end]
            seg_duration = float(segment.size) / float(sr)
            if seg_duration < 0.15:
                break
            try:
                seg_text, _seg_usage = transcriber_obj.transcribe(segment, seg_duration, sr)
                seg_text = str(seg_text or "").strip()
                if seg_text:
                    parts.append(seg_text)
            except Exception as err:
                last_err = err
                logging.warning("Sherpa media chunk failed: start=%s end=%s err=%s", start, end, err)
            start += step
            loops += 1
            if loops > 10000:
                break
        if not parts:
            if last_err:
                raise last_err
            return "", {"input_tokens": 0, "output_tokens": 0}
        merged = " ".join(parts).strip()
        # Final normalization pass to smooth joins.
        merged = transcriber_obj._postprocess_text(merged) if merged else merged
        return merged, {"input_tokens": 0, "output_tokens": 0}

    def _media_processing_worker(self) -> None:
        transcriber_cache_key = None
        transcriber_cache = None
        while self._running:
            with self._media_lock:
                next_item = None
                for item in self._media_queue_items:
                    if str(item.get("status", "queued")) == "queued":
                        next_item = item
                        break
                if next_item is None:
                    self._media_active_id = ""
                    self._media_is_processing = False
                    return
                item_id = str(next_item.get("id", ""))
                next_item["status"] = "processing"
                next_item["progress"] = 5.0
                next_item["error"] = ""
                next_item["ts_start"] = time.time()
                self._media_active_id = item_id
                self._media_is_processing = True
                item_snapshot = dict(next_item)
            try:
                cfg = AppConfig(**asdict(self.config))
                cache_key = (
                    cfg.transcription_backend,
                    cfg.local_backend_engine,
                    cfg.local_whisper_model,
                    cfg.local_vosk_model,
                    cfg.local_sherpa_model,
                    cfg.model,
                    bool(cfg.whisper_medium_turbo),
                )
                if transcriber_cache is None or transcriber_cache_key != cache_key:
                    transcriber_cache = self._create_transcriber(cfg)
                    transcriber_cache_key = cache_key
                with self._media_lock:
                    for item in self._media_queue_items:
                        if str(item.get("id", "")) == item_id:
                            item["progress"] = 20.0
                            break

                text, usage, duration = self._process_one_media_item(item_snapshot, cfg, transcriber_cache)
                with self._media_lock:
                    for item in self._media_queue_items:
                        if str(item.get("id", "")) == item_id:
                            item["duration_sec"] = max(0.0, float(duration))
                            item["text"] = str(text or "")
                            item["usage"] = dict(usage or {})
                            item["status"] = "done"
                            item["progress"] = 100.0
                            item["error"] = ""
                            item["ts_end"] = time.time()
                            break
                self._update_stats(text, duration, usage)
                self._append_history_entry(text)
            except Exception as err:
                logging.exception("Media processing failed for item %s", item_id)
                with self._media_lock:
                    for item in self._media_queue_items:
                        if str(item.get("id", "")) == item_id:
                            item["status"] = "error"
                            item["progress"] = 100.0
                            item["error"] = str(err)
                            item["ts_end"] = time.time()
                            break
            finally:
                with self._media_lock:
                    self._media_active_id = ""
                    # Keep thread running for remaining queued items.
                    has_queued = any(str(it.get("status", "queued")) == "queued" for it in self._media_queue_items)
                    if not has_queued:
                        self._media_is_processing = False
                        self._media_worker_thread = None
                        return

    def _add_media_files_from_ui(self, paths: list[str]) -> tuple[bool, str, dict]:
        added = 0
        skipped = 0
        with self._media_lock:
            existing_paths = {str(item.get("path", "")) for item in self._media_queue_items}
            for raw in list(paths or []):
                try:
                    p = Path(str(raw)).expanduser().resolve()
                except Exception:
                    skipped += 1
                    continue
                if not p.exists() or not p.is_file():
                    skipped += 1
                    continue
                p_str = str(p)
                if p_str in existing_paths:
                    skipped += 1
                    continue
                existing_paths.add(p_str)
                item = {
                    "id": self._next_media_item_id(),
                    "name": self._safe_media_name(p),
                    "path": p_str,
                    "duration_sec": 0.0,
                    "status": "queued",
                    "progress": 0.0,
                    "error": "",
                    "text": "",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                    "ts_start": 0.0,
                    "ts_end": 0.0,
                }
                self._media_queue_items.append(item)
                added += 1
        snap = self._get_media_queue_snapshot()
        if added <= 0:
            return False, "Файлы не добавлены (возможно, уже в очереди или недоступны).", snap
        if skipped > 0:
            return True, f"Добавлено файлов: {added}. Пропущено: {skipped}.", snap
        return True, f"Добавлено файлов: {added}.", snap

    def _start_media_processing_from_ui(self) -> tuple[bool, str]:
        with self._media_lock:
            if self._media_is_processing and self._media_worker_thread and self._media_worker_thread.is_alive():
                return True, "Обработка уже выполняется."
            has_queued = any(str(item.get("status", "queued")) == "queued" for item in self._media_queue_items)
            if not has_queued:
                return False, "Нет файлов в статусе «В очереди»."
            self._media_is_processing = True
            worker = threading.Thread(target=self._media_processing_worker, daemon=True)
            self._media_worker_thread = worker
            worker.start()
        return True, "Обработка очереди запущена."

    def _clear_media_queue_from_ui(self) -> tuple[bool, str, dict]:
        processing = False
        with self._media_lock:
            processing = bool(self._media_is_processing)
            if not processing:
                self._media_queue_items = []
                self._media_active_id = ""
        if processing:
            return False, "Нельзя очистить очередь во время обработки.", self._get_media_queue_snapshot()
        return True, "Очередь очищена.", self._get_media_queue_snapshot()

    def _copy_media_result_from_ui(self, item_id: str) -> tuple[bool, str]:
        item = None
        with self._media_lock:
            for row in self._media_queue_items:
                if str(row.get("id", "")) == str(item_id):
                    item = dict(row)
                    break
        if not item:
            return False, "Элемент очереди не найден."
        text = str(item.get("text", "")).strip()
        if not text:
            return False, "Нет текста для копирования."
        try:
            self._copy_text(text)
            return True, "Текст скопирован."
        except Exception as err:
            return False, f"Ошибка: {err}"

    def _export_media_result_from_ui(self, item_id: str, export_format: str, target_path: Optional[str] = None) -> tuple[bool, str]:
        item = None
        with self._media_lock:
            for row in self._media_queue_items:
                if str(row.get("id", "")) == str(item_id):
                    item = dict(row)
                    break
        if not item:
            return False, "Элемент очереди не найден."
        text = str(item.get("text", "")).strip()
        if not text:
            return False, "Нет текста для экспорта."
        ext = str(export_format or "txt").strip().lower()
        if ext not in {"txt", "srt"}:
            ext = "txt"
        stem = self._sanitize_filename(Path(str(item.get("name", "result"))).stem)
        out_path: Path
        target_raw = str(target_path or "").strip()
        if target_raw:
            out_path = Path(target_raw).expanduser()
            if out_path.suffix.lower() != f".{ext}":
                out_path = out_path.with_suffix(f".{ext}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            MEDIA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = MEDIA_EXPORT_DIR / f"{stem}_{ts}.{ext}"
        try:
            if ext == "srt":
                payload = self._build_srt_from_text(text, float(item.get("duration_sec", 0.0)))
            else:
                payload = text + ("\n" if not text.endswith("\n") else "")
            out_path.write_text(payload, encoding="utf-8")
            return True, f"Экспортировано: {out_path}"
        except Exception as err:
            logging.exception("Failed to export media result")
            return False, f"Ошибка экспорта: {err}"

    def _get_media_queue_from_ui(self) -> dict:
        return self._get_media_queue_snapshot()

    def _meeting_backend_label(self, cfg: AppConfig) -> str:
        if str(getattr(cfg, "transcription_backend", "api") or "api").strip().lower() == "api":
            return "API"
        engine = str(getattr(cfg, "local_backend_engine", "whisper") or "whisper").strip().lower()
        if engine == "vosk":
            return "Локально · Vosk"
        if engine == "sherpa":
            return "Локально · Sherpa"
        model = str(getattr(cfg, "local_whisper_model", "small") or "small").strip().lower()
        return f"Локально · Whisper {model}"

    def _get_meeting_session_snapshot(self) -> dict:
        with self._meeting_lock:
            snap = dict(self._meeting_state)
            audio_levels = dict((snap.get("audio_levels") or {}))
            snap["audio_levels"] = {
                "mic": int(max(0, min(100, int(audio_levels.get("mic", 0) or 0)))),
                "system": int(max(0, min(100, int(audio_levels.get("system", 0) or 0)))),
            }
            snap["loopback_device_name"] = str(snap.get("loopback_device_name", "") or "")
            snap["loopback_channels"] = int(max(0, int(snap.get("loopback_channels", 0) or 0)))
            snap["loopback_error"] = str(snap.get("loopback_error", "") or "")
            snap["loopback_error_code"] = str(snap.get("loopback_error_code", "") or "")
            snap["loopback_error_message"] = str(snap.get("loopback_error_message", "") or "")
            snap["system_backend"] = str(snap.get("system_backend", "") or "none")
            if not str(snap.get("summary_backend", "")).strip():
                sb = str(getattr(self.config, "meeting_summary_backend", "api")).strip().lower()
                if sb == "local_gguf":
                    snap["summary_backend"] = "Локально (GGUF)"
                elif sb == "local_ollama":
                    snap["summary_backend"] = "Локально (Ollama)"
                else:
                    snap["summary_backend"] = "API"
            if not str(snap.get("summary_model", "")).strip():
                sb = str(getattr(self.config, "meeting_summary_backend", "api")).strip().lower()
                if sb == "local_gguf":
                    snap["summary_model"] = LocalMeetingSummaryGGUF.model_title(
                        str(getattr(self.config, "meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
                    )
                elif sb == "local_ollama":
                    snap["summary_model"] = str(getattr(self.config, "meeting_ollama_model", "") or "").strip()
                else:
                    snap["summary_model"] = str(getattr(self.config, "model", "") or "").strip()
            if bool(snap.get("is_running")) and float(snap.get("started_at", 0.0) or 0.0) > 0.0:
                snap["duration_sec"] = max(0.0, time.time() - float(snap.get("started_at", 0.0)))
            return snap

    def _meeting_update(self, **kwargs) -> None:
        if "error" in kwargs:
            raw_err = str(kwargs.get("error", "") or "").strip()
            if len(raw_err) > 420:
                kwargs["error"] = raw_err[:417].rstrip() + "..."
        with self._meeting_lock:
            self._meeting_state.update(kwargs)
            self._meeting_state["updated_at"] = time.time()

    @staticmethod
    def _meeting_audio_level(samples: np.ndarray) -> int:
        if samples is None or samples.size <= 0:
            return 0
        try:
            peak = float(np.max(np.abs(np.asarray(samples, dtype=np.float32))))
        except Exception:
            peak = 0.0
        return int(max(0, min(100, round(peak * 100.0))))

    def _meeting_transcribe_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        cfg: AppConfig,
        transcriber_obj,
    ) -> tuple[str, dict, float]:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = float(audio.size) / float(max(8000, int(sample_rate)))
        if duration < 0.15 or not self._has_audio_signal(audio):
            return "", {"input_tokens": 0, "output_tokens": 0}, duration
        if isinstance(transcriber_obj, (LocalWhisperSmallTranscriber, LocalVoskTranscriber, LocalSherpaWhispeRuTranscriber)):
            text, usage = transcriber_obj.transcribe(audio, duration, int(sample_rate))
            text, refine_usage, _ = self._fast_refine_local_text_for_context(text, duration, cfg, transcriber_obj)
            usage = self._merge_usage(usage, refine_usage)
        else:
            wav_bytes = float_audio_to_wav_bytes(audio, int(sample_rate))
            text, usage = transcriber_obj.transcribe(wav_bytes, duration)
        normalized = self._apply_autoreplace_with_enabled(str(text or "").strip(), bool(getattr(cfg, "autoreplace_enabled", True)))
        return normalized, dict(usage or {}), duration

    def _meeting_get_loopback_candidates(self) -> list[tuple[int, str]]:
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception:
            return []
        supports_wasapi = self._supports_sd_wasapi_loopback()
        candidates: list[tuple[int, str]] = []
        preferred: list[tuple[int, str]] = []

        # 1) last good device first
        if self._meeting_last_loopback_device is not None:
            try:
                idx = int(self._meeting_last_loopback_device)
                if 0 <= idx < len(devices):
                    dev = devices[idx]
                    max_in = int(dev.get("max_input_channels", 0) or 0)
                    max_out = int(dev.get("max_output_channels", 0) or 0)
                    if max_in > 0:
                        preferred.append((idx, "native_loopback"))
                    elif supports_wasapi and max_out > 0:
                        preferred.append((idx, "wasapi_output"))
            except Exception:
                pass

        # 2) explicit WASAPI loopback input devices (most stable way).
        for idx, dev in enumerate(devices):
            try:
                host_idx = int(dev.get("hostapi", -1))
                host_name = str((hostapis[host_idx] or {}).get("name", "")).lower() if 0 <= host_idx < len(hostapis) else ""
                if "wasapi" not in host_name:
                    continue
                name = str(dev.get("name", "")).lower()
                max_in = int(dev.get("max_input_channels", 0) or 0)
                if max_in > 0 and (
                    "loopback" in name
                    or "loop back" in name
                    or "stereo mix" in name
                    or "стерео микшер" in name
                    or "what u hear" in name
                ):
                    preferred.append((idx, "native_loopback"))
            except Exception:
                continue

        # 3) default/all WASAPI output devices only when loopback API is supported by sounddevice build.
        if supports_wasapi:
            try:
                default_dev = sd.default.device
                if isinstance(default_dev, (list, tuple)) and len(default_dev) >= 2:
                    out_idx = int(default_dev[1])
                    if 0 <= out_idx < len(devices):
                        dev = devices[out_idx]
                        host_idx = int(dev.get("hostapi", -1))
                        host_name = str((hostapis[host_idx] or {}).get("name", "")).lower() if 0 <= host_idx < len(hostapis) else ""
                        if int(dev.get("max_output_channels", 0) or 0) > 0 and "wasapi" in host_name:
                            preferred.append((out_idx, "wasapi_output"))
            except Exception:
                pass
            for idx, dev in enumerate(devices):
                try:
                    if int(dev.get("max_output_channels", 0) or 0) <= 0:
                        continue
                    host_idx = int(dev.get("hostapi", -1))
                    host_name = str((hostapis[host_idx] or {}).get("name", "")).lower() if 0 <= host_idx < len(hostapis) else ""
                    if "wasapi" in host_name:
                        candidates.append((idx, "wasapi_output"))
                except Exception:
                    continue

        ordered: list[tuple[int, str]] = []
        seen: set[tuple[int, str]] = set()
        for entry in preferred + candidates:
            if entry not in seen:
                seen.add(entry)
                ordered.append(entry)
        return ordered

    @staticmethod
    def _parse_pa_error(raw_error: str) -> tuple[str, str]:
        text = str(raw_error or "").strip()
        if not text:
            return "", ""
        m = re.search(r"PaErrorCode\s+(-?\d+)", text, flags=re.IGNORECASE)
        code = m.group(1) if m else ""
        return code, text

    @staticmethod
    def _parse_meeting_system_device_id(device_id: str) -> tuple[str, str]:
        raw = str(device_id or "").strip()
        if not raw or "::" not in raw:
            return "", ""
        kind, value = raw.split("::", 1)
        kind = str(kind or "").strip().lower()
        value = str(value or "").strip()
        if kind in {"sc", "si", "wa"} and value:
            return kind, value
        return "", ""

    @staticmethod
    def _supports_sd_wasapi_loopback() -> bool:
        if not hasattr(sd, "WasapiSettings"):
            return False
        try:
            import inspect
            sig = inspect.signature(sd.WasapiSettings)
            if "loopback" not in sig.parameters:
                return False
            sd.WasapiSettings(loopback=True)
            return True
        except Exception:
            return False

    def _start_soundcard_loopback_worker(
        self,
        sample_rate: int,
        blocksize: int,
        sys_q: "queue.Queue[np.ndarray]",
        preferred_device_id: str = "",
    ) -> tuple[Optional[threading.Event], Optional[threading.Thread], str, int, str]:
        if sc is None:
            return None, None, "", 0, "Библиотека soundcard не установлена."
        try:
            preferred_kind, preferred_value = self._parse_meeting_system_device_id(preferred_device_id)
            loop_mics: list = []
            preferred: list = []
            secondary: list = []
            default_loop = None
            try:
                default_loop = sc.default_microphone(include_loopback=True)
            except Exception:
                default_loop = None
            if default_loop is not None:
                did = str(getattr(default_loop, "id", "") or "")
                if "{0.0.0." in did.replace(" ", "").lower():
                    preferred.append(default_loop)
                else:
                    secondary.append(default_loop)
            try:
                for mic in list(sc.all_microphones(include_loopback=True) or []):
                    mid = str(getattr(mic, "id", "") or "")
                    if not mid:
                        continue
                    if all(str(getattr(existing, "id", "") or "") != mid for existing in preferred + secondary):
                        if "{0.0.0." in mid.replace(" ", "").lower():
                            preferred.append(mic)
                        else:
                            secondary.append(mic)
            except Exception:
                pass
            loop_mics.extend(preferred)
            loop_mics.extend(secondary)
            if preferred_kind == "sc" and preferred_value:
                loop_mics = [m for m in loop_mics if str(getattr(m, "id", "") or "").strip() == preferred_value]
            if not loop_mics:
                return None, None, "", 0, "Loopback-устройства не найдены."

            chosen_loop_mic = None
            chosen_channels = 0
            last_err = ""
            for loop_mic in loop_mics:
                base_channels = int(getattr(loop_mic, "channels", 0) or 0)
                candidates: list[int] = []
                for c in (2, base_channels, 1):
                    if int(c or 0) > 0 and int(c) not in candidates:
                        candidates.append(int(c))
                for channels in candidates:
                    try:
                        test_rec = loop_mic.recorder(samplerate=int(sample_rate), channels=int(channels), blocksize=int(blocksize))
                        test_rec.__enter__()
                        test_rec.__exit__(None, None, None)
                        chosen_loop_mic = loop_mic
                        chosen_channels = int(channels)
                        break
                    except Exception as err:
                        last_err = str(err)
                        continue
                if chosen_loop_mic is not None:
                    break

            if chosen_loop_mic is None or chosen_channels <= 0:
                return None, None, "", 0, (last_err or "Loopback-устройство не найдено.")

            loop_mic = chosen_loop_mic
            name = str(getattr(loop_mic, "name", "") or "System loopback").strip()
            channels = int(chosen_channels)
            logging.info("Meeting soundcard loopback initialized: device=%s channels=%s sr=%s", name, channels, sample_rate)
            stop_event = threading.Event()

            def _worker() -> None:
                rec = None
                try:
                    rec = loop_mic.recorder(samplerate=int(sample_rate), channels=channels, blocksize=int(blocksize))
                    rec.__enter__()
                    while not stop_event.is_set() and not self._meeting_stop_event.is_set():
                        data = rec.record(numframes=int(blocksize))
                        if data is None:
                            continue
                        arr = np.asarray(data, dtype=np.float32)
                        if arr.ndim == 2:
                            arr = np.mean(arr, axis=1, dtype=np.float32)
                        arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
                        if arr.size <= 0:
                            continue
                        if not self._meeting_state.get("is_paused", False):
                            sys_q.put(arr)
                        self._meeting_update(
                            audio_levels={
                                "mic": int(self._meeting_state.get("audio_levels", {}).get("mic", 0)),
                                "system": self._meeting_audio_level(arr),
                            }
                        )
                except Exception as err:
                    logging.warning("Soundcard loopback worker failed: %s", err)
                    self._meeting_update(
                        loopback_available=False,
                        loopback_device_name="",
                        loopback_channels=0,
                        loopback_error=str(err),
                        loopback_error_code="",
                        loopback_error_message=str(err),
                        error="Системный звук недоступен. Запись продолжена только с микрофона.",
                    )
                finally:
                    if rec is not None:
                        try:
                            rec.__exit__(None, None, None)
                        except Exception:
                            pass

            thr = threading.Thread(target=_worker, name="VoicePROAI-MeetingLoopbackSC", daemon=True)
            thr.start()
            return stop_event, thr, name, channels, ""
        except Exception as err:
            return None, None, "", 0, str(err)

    def _meeting_try_open_loopback_stream(
        self,
        sample_rate: int,
        blocksize: int,
        callback,
        preferred_device_id: str = "",
    ) -> tuple[Optional[sd.InputStream], str, int, str]:
        can_use_wasapi_loopback = self._supports_sd_wasapi_loopback()
        preferred_kind, preferred_value = self._parse_meeting_system_device_id(preferred_device_id)
        try:
            devices = sd.query_devices()
        except Exception:
            devices = []

        last_error = ""
        for device_idx, mode in self._meeting_get_loopback_candidates():
            if preferred_kind == "wa" and preferred_value:
                if str(device_idx) != str(preferred_value):
                    continue
            elif preferred_kind and preferred_kind != "wa":
                continue
            dev_name = ""
            max_in = 0
            max_out = 0
            try:
                dev = devices[device_idx] if 0 <= int(device_idx) < len(devices) else {}
                dev_name = str(dev.get("name", "")).strip()
                max_in = int(dev.get("max_input_channels", 0) or 0)
                max_out = int(dev.get("max_output_channels", 0) or 0)
            except Exception:
                max_in = 0
                max_out = 0

            mode_candidates: list[str] = []
            if mode == "native_loopback":
                mode_candidates.append("native_loopback")
                if can_use_wasapi_loopback:
                    mode_candidates.append("wasapi_output")
            elif mode == "wasapi_output":
                if can_use_wasapi_loopback:
                    mode_candidates.append("wasapi_output")
                if max_in > 0:
                    mode_candidates.append("native_loopback")
            else:
                if max_in > 0:
                    mode_candidates.append("native_loopback")
                if can_use_wasapi_loopback:
                    mode_candidates.append("wasapi_output")
            if not mode_candidates:
                mode_candidates = ["native_loopback"]

            for try_mode in mode_candidates:
                channel_candidates: list[int] = []
                if try_mode == "native_loopback":
                    first = min(2, max_in) if max_in > 0 else 1
                    for ch in [first, max_in, 1]:
                        try:
                            ch_i = int(ch)
                        except Exception:
                            continue
                        if ch_i > 0 and ch_i not in channel_candidates:
                            channel_candidates.append(ch_i)
                else:
                    for ch in [2, max_out, 1]:
                        try:
                            ch_i = int(ch)
                        except Exception:
                            continue
                        if ch_i > 0 and ch_i not in channel_candidates:
                            channel_candidates.append(ch_i)
                if not channel_candidates:
                    channel_candidates = [1]

                extra_settings = None
                if try_mode == "wasapi_output" and can_use_wasapi_loopback:
                    try:
                        extra_settings = sd.WasapiSettings(loopback=True)
                    except Exception:
                        extra_settings = None

                for channels in channel_candidates:
                    try:
                        stream = sd.InputStream(
                            samplerate=sample_rate,
                            channels=channels,
                            dtype="float32",
                            blocksize=blocksize,
                            latency="low",
                            device=int(device_idx),
                            callback=callback,
                            extra_settings=extra_settings,
                        )
                        stream.start()
                        self._meeting_last_loopback_device = int(device_idx)
                        return stream, (dev_name or f"device#{device_idx}"), int(channels), ""
                    except Exception as err:
                        last_error = str(err)
                        logging.warning(
                            "Loopback open failed: device=%s mode=%s name=%s channels=%s err=%s",
                            device_idx,
                            try_mode,
                            dev_name,
                            channels,
                            err,
                        )
                        continue

        # Final fallback: default output with explicit WASAPI loopback and safe channels.
        if can_use_wasapi_loopback and (not preferred_kind or preferred_kind == "wa"):
            try:
                default_dev = sd.default.device
                out_idx = int(default_dev[1]) if isinstance(default_dev, (list, tuple)) and len(default_dev) >= 2 else -1
                if preferred_kind == "wa" and preferred_value and str(out_idx) != str(preferred_value):
                    return None, "", 0, (last_error or "Выбранный WASAPI loopback недоступен")
                if 0 <= out_idx < len(devices):
                    dev = devices[out_idx]
                    max_out = int(dev.get("max_output_channels", 0) or 0)
                    dev_name = str(dev.get("name", "")).strip() or f"device#{out_idx}"
                    for channels in [min(2, max_out) if max_out > 0 else 1, max_out, 1]:
                        try:
                            ch_i = int(channels)
                        except Exception:
                            continue
                        if ch_i <= 0:
                            continue
                        try:
                            stream = sd.InputStream(
                                samplerate=sample_rate,
                                channels=ch_i,
                                dtype="float32",
                                blocksize=blocksize,
                                latency="low",
                                device=out_idx,
                                callback=callback,
                                extra_settings=sd.WasapiSettings(loopback=True),
                            )
                            stream.start()
                            self._meeting_last_loopback_device = int(out_idx)
                            return stream, dev_name, int(ch_i), ""
                        except Exception as err:
                            last_error = str(err)
                            logging.warning(
                                "Loopback default fallback failed: device=%s name=%s channels=%s err=%s",
                                out_idx,
                                dev_name,
                                ch_i,
                                err,
                            )
                            continue
            except Exception:
                pass

        return None, "", 0, last_error

    def _meeting_try_open_system_input_stream(
        self,
        sample_rate: int,
        blocksize: int,
        callback,
        preferred_device_id: str = "",
    ) -> tuple[Optional[sd.InputStream], str, int, str]:
        keywords = ("stereo mix", "what u hear", "wave out", "mix", "loopback")
        last_error = ""
        preferred_kind, preferred_value = self._parse_meeting_system_device_id(preferred_device_id)
        try:
            devices = list(sd.query_devices() or [])
        except Exception as err:
            return None, "", 0, str(err)
        candidates: list[tuple[int, str, int]] = []
        for idx, dev in enumerate(devices):
            try:
                max_in = int(dev.get("max_input_channels", 0) or 0)
            except Exception:
                max_in = 0
            if max_in <= 0:
                continue
            name = str(dev.get("name", "") or "").strip()
            low = name.lower()
            if any(k in low for k in keywords):
                candidates.append((int(idx), name, max_in))
        if preferred_kind == "si" and preferred_value:
            candidates = [entry for entry in candidates if str(entry[0]) == str(preferred_value)]
        elif preferred_kind and preferred_kind != "si":
            candidates = []
        for idx, name, max_in in candidates:
            channel_candidates: list[int] = []
            for ch in (2, max_in, 1):
                try:
                    ch_i = int(ch)
                except Exception:
                    continue
                if ch_i > 0 and ch_i not in channel_candidates:
                    channel_candidates.append(ch_i)
            if not channel_candidates:
                channel_candidates = [1]
            for channels in channel_candidates:
                try:
                    stream = sd.InputStream(
                        samplerate=sample_rate,
                        channels=channels,
                        dtype="float32",
                        blocksize=blocksize,
                        latency="low",
                        device=idx,
                        callback=callback,
                    )
                    stream.start()
                    logging.info(
                        "Meeting system-input initialized: device=%s idx=%s channels=%s sr=%s",
                        name,
                        idx,
                        channels,
                        sample_rate,
                    )
                    return stream, name, int(channels), ""
                except Exception as err:
                    last_error = str(err)
                    continue
        return None, "", 0, (last_error or "Стерео-микс не найден")

    def _meeting_open_system_audio_channel(
        self,
        cfg: AppConfig,
        sample_rate: int,
        blocksize: int,
        sys_q: "queue.Queue[np.ndarray]",
        callback,
    ) -> tuple[Optional[sd.InputStream], Optional[threading.Event], Optional[threading.Thread], str, int, str, str]:
        mode = str(getattr(cfg, "meeting_system_audio_mode", "auto") or "auto").strip().lower()
        selected_id = str(getattr(cfg, "meeting_system_audio_device_id", "") or "").strip()
        kind, _ = self._parse_meeting_system_device_id(selected_id)

        # mode=manual: only selected device/backend, no cross-device auto-fallback
        if mode == "manual":
            if not selected_id or not kind:
                return None, None, None, "", 0, "Выберите устройство системного звука в настройках.", "none"
            if kind == "sc":
                sc_stop, sc_thread, sc_name, sc_channels, sc_err = self._start_soundcard_loopback_worker(
                    sample_rate,
                    blocksize,
                    sys_q,
                    preferred_device_id=selected_id,
                )
                if sc_thread is not None:
                    return None, sc_stop, sc_thread, sc_name, sc_channels, "", "soundcard"
                return None, None, None, "", 0, (sc_err or "Выбранный loopback soundcard недоступен"), "none"
            if kind == "si":
                stream, name, channels, err = self._meeting_try_open_system_input_stream(
                    sample_rate,
                    blocksize,
                    callback,
                    preferred_device_id=selected_id,
                )
                if stream is not None:
                    return stream, None, None, name, channels, "", "system_input"
                return None, None, None, "", 0, (err or "Выбранный системный вход недоступен"), "none"
            if kind == "wa":
                stream, name, channels, err = self._meeting_try_open_loopback_stream(
                    sample_rate,
                    blocksize,
                    callback,
                    preferred_device_id=selected_id,
                )
                if stream is not None:
                    return stream, None, None, name, channels, "", "sounddevice"
                return None, None, None, "", 0, (err or "Выбранный WASAPI loopback недоступен"), "none"
            return None, None, None, "", 0, "Неизвестный тип устройства системного звука.", "none"

        # mode=auto: stable backend chain
        sc_stop, sc_thread, sc_name, sc_channels, sc_err = self._start_soundcard_loopback_worker(
            sample_rate,
            blocksize,
            sys_q,
        )
        if sc_thread is not None:
            return None, sc_stop, sc_thread, sc_name, sc_channels, "", "soundcard"

        stream, name, channels, err = self._meeting_try_open_system_input_stream(
            sample_rate,
            blocksize,
            callback,
        )
        if stream is not None:
            return stream, None, None, name, channels, "", "system_input"

        stream, name, channels, err2 = self._meeting_try_open_loopback_stream(
            sample_rate,
            blocksize,
            callback,
        )
        if stream is not None:
            return stream, None, None, name, channels, "", "sounddevice"

        final_err = str(err2 or err or sc_err or "Системный звук недоступен.")
        return None, None, None, "", 0, final_err, "none"

    def _meeting_capture_worker(self, cfg: AppConfig) -> None:
        sample_rate = max(16000, min(48000, int(getattr(cfg, "sample_rate", 24000) or 24000)))
        blocksize = 1024
        mic_q: queue.Queue[np.ndarray] = queue.Queue()
        sys_q: queue.Queue[np.ndarray] = queue.Queue()
        transcriber_obj = self._create_transcriber(cfg)
        aggregate_usage = {"input_tokens": 0, "output_tokens": 0}
        transcript_parts: list[str] = []
        total_duration = 0.0

        def _mic_cb(indata, _frames, _time_info, status):
            if status:
                logging.debug("Meeting mic callback status: %s", status)
            if indata is None:
                return
            arr = np.asarray(indata, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[:, 0]
            arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
            if arr.size <= 0:
                return
            if not self._meeting_state.get("is_paused", False):
                mic_q.put(arr)
            self._meeting_update(audio_levels={"mic": self._meeting_audio_level(arr), "system": int(self._meeting_state.get("audio_levels", {}).get("system", 0))})

        def _sys_cb(indata, _frames, _time_info, status):
            if status:
                logging.debug("Meeting loopback callback status: %s", status)
            if indata is None:
                return
            arr = np.asarray(indata, dtype=np.float32)
            if arr.ndim == 2:
                arr = np.mean(arr, axis=1, dtype=np.float32)
            arr = np.asarray(arr, dtype=np.float32).reshape(-1).copy()
            if arr.size <= 0:
                return
            if not self._meeting_state.get("is_paused", False):
                sys_q.put(arr)
            self._meeting_update(audio_levels={"mic": int(self._meeting_state.get("audio_levels", {}).get("mic", 0)), "system": self._meeting_audio_level(arr)})

        def _drain(qobj: "queue.Queue[np.ndarray]") -> list[np.ndarray]:
            out: list[np.ndarray] = []
            while True:
                try:
                    out.append(qobj.get_nowait())
                except queue.Empty:
                    break
            return out

        mic_stream = None
        system_stream = None
        system_sc_stop: Optional[threading.Event] = None
        system_sc_thread: Optional[threading.Thread] = None
        try:
            mic_device_idx = AudioRecorder.resolve_device_index(str(cfg.input_device or "").strip())
            mic_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                latency="low",
                device=mic_device_idx,
                callback=_mic_cb,
            )
            mic_stream.start()
            self._meeting_mic_stream = mic_stream

            system_stream, system_sc_stop, system_sc_thread, loopback_name, loopback_channels, loopback_err, system_stream_backend = (
                self._meeting_open_system_audio_channel(
                    cfg,
                    sample_rate,
                    blocksize,
                    sys_q,
                    _sys_cb,
                )
            )
            if system_stream is not None:
                logging.info(
                    "Meeting system audio connected: backend=%s device=%s channels=%s",
                    system_stream_backend,
                    loopback_name,
                    loopback_channels,
                )
                self._meeting_system_stream = system_stream
                self._meeting_loopback_info = {"name": loopback_name, "channels": int(loopback_channels), "error": ""}
                self._meeting_update(
                    loopback_available=True,
                    system_backend=(system_stream_backend or "sounddevice"),
                    loopback_device_name=loopback_name,
                    loopback_channels=int(loopback_channels),
                    loopback_error="",
                    loopback_error_code="",
                    loopback_error_message="",
                    error="",
                )
            elif system_sc_thread is not None:
                logging.info(
                    "Meeting system audio connected: backend=soundcard device=%s channels=%s",
                    loopback_name,
                    loopback_channels,
                )
                self._meeting_system_stream = None
                self._meeting_loopback_info = {"name": loopback_name, "channels": int(loopback_channels), "error": ""}
                self._meeting_update(
                    loopback_available=True,
                    system_backend="soundcard",
                    loopback_device_name=loopback_name,
                    loopback_channels=int(loopback_channels),
                    loopback_error="",
                    loopback_error_code="",
                    loopback_error_message="",
                    error="",
                )
            else:
                logging.warning("Meeting system audio unavailable: %s", loopback_err)
                lp_code, lp_msg = self._parse_pa_error(loopback_err)
                self._meeting_loopback_info = {"name": "", "channels": 0, "error": str(loopback_err or "")}
                self._meeting_update(
                    loopback_available=False,
                    system_backend="none",
                    loopback_device_name="",
                    loopback_channels=0,
                    loopback_error=str(loopback_err or ""),
                    loopback_error_code=lp_code,
                    loopback_error_message=lp_msg,
                    error="Системный звук недоступен. Запись продолжена только с микрофона.",
                )

            target_chunk_samples = int(sample_rate * 4.5)
            mic_buf: list[np.ndarray] = []
            sys_buf: list[np.ndarray] = []
            last_process = time.monotonic()

            while not self._meeting_stop_event.is_set():
                time.sleep(0.08)
                if self._meeting_reinit_loopback_event.is_set():
                    self._meeting_reinit_loopback_event.clear()
                    try:
                        if system_stream is not None:
                            system_stream.stop()
                            system_stream.close()
                    except Exception:
                        logging.exception("Failed to close loopback stream on reinit")
                    system_stream = None
                    if system_sc_stop is not None:
                        system_sc_stop.set()
                    if system_sc_thread is not None and system_sc_thread.is_alive():
                        system_sc_thread.join(timeout=1.2)
                    system_sc_stop = None
                    system_sc_thread = None
                    self._meeting_system_stream = None
                    system_stream, system_sc_stop, system_sc_thread, loopback_name, loopback_channels, loopback_err, system_stream_backend = (
                        self._meeting_open_system_audio_channel(
                            cfg,
                            sample_rate,
                            blocksize,
                            sys_q,
                            _sys_cb,
                        )
                    )
                    if system_stream is not None:
                        logging.info(
                            "Meeting system audio reinit success: backend=%s device=%s channels=%s",
                            system_stream_backend,
                            loopback_name,
                            loopback_channels,
                        )
                        self._meeting_system_stream = system_stream
                        self._meeting_loopback_info = {"name": loopback_name, "channels": int(loopback_channels), "error": ""}
                        self._meeting_update(
                            loopback_available=True,
                            system_backend=(system_stream_backend or "sounddevice"),
                            loopback_device_name=loopback_name,
                            loopback_channels=int(loopback_channels),
                            loopback_error="",
                            loopback_error_code="",
                            loopback_error_message="",
                            error="",
                        )
                    elif system_sc_thread is not None:
                        logging.info(
                            "Meeting system audio reinit success: backend=soundcard device=%s channels=%s",
                            loopback_name,
                            loopback_channels,
                        )
                        self._meeting_system_stream = None
                        self._meeting_loopback_info = {"name": loopback_name, "channels": int(loopback_channels), "error": ""}
                        self._meeting_update(
                            loopback_available=True,
                            system_backend="soundcard",
                            loopback_device_name=loopback_name,
                            loopback_channels=int(loopback_channels),
                            loopback_error="",
                            loopback_error_code="",
                            loopback_error_message="",
                            error="",
                        )
                    else:
                        logging.warning("Meeting system audio reinit failed: %s", loopback_err)
                        lp_code, lp_msg = self._parse_pa_error(loopback_err)
                        self._meeting_loopback_info = {"name": "", "channels": 0, "error": str(loopback_err or "")}
                        self._meeting_update(
                            loopback_available=False,
                            system_backend="none",
                            loopback_device_name="",
                            loopback_channels=0,
                            loopback_error=str(loopback_err or ""),
                            loopback_error_code=lp_code,
                            loopback_error_message=lp_msg,
                            error="Системный звук недоступен. Запись продолжена только с микрофона.",
                        )
                if bool(self._meeting_state.get("is_paused", False)):
                    continue
                mic_buf.extend(_drain(mic_q))
                sys_buf.extend(_drain(sys_q))
                mic_len = int(sum(int(x.size) for x in mic_buf))
                sys_len = int(sum(int(x.size) for x in sys_buf))
                buffered = max(mic_len, sys_len)
                if buffered < max(1, target_chunk_samples) and (time.monotonic() - last_process) < 5.0:
                    continue
                if buffered <= 0:
                    continue
                mic_audio = np.concatenate(mic_buf).astype(np.float32, copy=False) if mic_buf else np.zeros(0, dtype=np.float32)
                sys_audio = np.concatenate(sys_buf).astype(np.float32, copy=False) if sys_buf else np.zeros(0, dtype=np.float32)
                mic_buf = []
                sys_buf = []
                max_len = int(max(mic_audio.size, sys_audio.size))
                if max_len <= 0:
                    continue
                if mic_audio.size < max_len:
                    mic_audio = np.pad(mic_audio, (0, max_len - mic_audio.size), mode="constant")
                if sys_audio.size < max_len:
                    sys_audio = np.pad(sys_audio, (0, max_len - sys_audio.size), mode="constant")
                mixed = (0.62 * mic_audio + 0.62 * sys_audio).astype(np.float32, copy=False)
                mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)
                if not self._has_audio_signal(mixed):
                    continue
                self._meeting_update(state="transcribing")
                txt, usage, dur = self._meeting_transcribe_chunk(mixed, sample_rate, cfg, transcriber_obj)
                last_process = time.monotonic()
                total_duration += max(0.0, float(dur))
                if txt:
                    transcript_parts.append(str(txt).strip())
                    joined = " ".join(part for part in transcript_parts if str(part).strip()).strip()
                    self._meeting_update(transcript_text=joined)
                aggregate_usage = self._merge_usage(aggregate_usage, usage)
                self._meeting_update(state="recording")

            mic_buf.extend(_drain(mic_q))
            sys_buf.extend(_drain(sys_q))
            if mic_buf or sys_buf:
                mic_audio = np.concatenate(mic_buf).astype(np.float32, copy=False) if mic_buf else np.zeros(0, dtype=np.float32)
                sys_audio = np.concatenate(sys_buf).astype(np.float32, copy=False) if sys_buf else np.zeros(0, dtype=np.float32)
                max_len = int(max(mic_audio.size, sys_audio.size))
                if max_len > 0:
                    if mic_audio.size < max_len:
                        mic_audio = np.pad(mic_audio, (0, max_len - mic_audio.size), mode="constant")
                    if sys_audio.size < max_len:
                        sys_audio = np.pad(sys_audio, (0, max_len - sys_audio.size), mode="constant")
                    mixed = np.clip((0.62 * mic_audio + 0.62 * sys_audio), -1.0, 1.0).astype(np.float32, copy=False)
                    if self._has_audio_signal(mixed):
                        self._meeting_update(state="transcribing")
                        txt, usage, dur = self._meeting_transcribe_chunk(mixed, sample_rate, cfg, transcriber_obj)
                        total_duration += max(0.0, float(dur))
                        if txt:
                            transcript_parts.append(str(txt).strip())
                            self._meeting_update(transcript_text=" ".join(part for part in transcript_parts if str(part).strip()).strip())
                        aggregate_usage = self._merge_usage(aggregate_usage, usage)

            transcript = " ".join(part for part in transcript_parts if str(part).strip()).strip()
            if transcript:
                self._update_stats(transcript, max(0.0, float(total_duration)), aggregate_usage)
                self._append_history_entry(transcript)
            self._meeting_update(
                state="done",
                is_running=False,
                is_paused=False,
                duration_sec=max(0.0, time.time() - float(self._meeting_state.get("started_at", time.time()))),
                transcript_text=transcript,
                backend_label=self._meeting_backend_label(cfg),
            )
        except Exception as err:
            logging.exception("Meeting capture worker failed")
            self._meeting_update(state="error", error=str(err), is_running=False, is_paused=False)
        finally:
            try:
                if mic_stream is not None:
                    mic_stream.stop()
                    mic_stream.close()
            except Exception:
                logging.exception("Failed to close meeting mic stream")
            try:
                if system_stream is not None:
                    system_stream.stop()
                    system_stream.close()
            except Exception:
                logging.exception("Failed to close meeting loopback stream")
            try:
                if system_sc_stop is not None:
                    system_sc_stop.set()
                if system_sc_thread is not None and system_sc_thread.is_alive():
                    system_sc_thread.join(timeout=1.2)
            except Exception:
                logging.exception("Failed to close soundcard loopback worker")
            self._meeting_mic_stream = None
            self._meeting_system_stream = None
            self._meeting_stop_event.set()

    def _start_meeting_capture_from_ui(self) -> tuple[bool, str]:
        with self._meeting_lock:
            if self._meeting_thread and self._meeting_thread.is_alive() and bool(self._meeting_state.get("is_running", False)):
                return False, "Встреча уже запущена."
            self._meeting_stop_event = threading.Event()
            self._meeting_reinit_loopback_event = threading.Event()
            self._meeting_state = {
                "state": "recording",
                "started_at": time.time(),
                "duration_sec": 0.0,
                "transcript_text": "",
                "summary_text": "",
                "audio_levels": {"mic": 0, "system": 0},
                "error": "",
                "backend_label": self._meeting_backend_label(self.config),
                "loopback_available": True,
                "system_backend": "none",
                "loopback_device_name": "",
                "loopback_channels": 0,
                "loopback_error": "",
                "loopback_error_code": "",
                "loopback_error_message": "",
                "summary_backend": (
                    "Локально (GGUF)"
                    if str(getattr(self.config, "meeting_summary_backend", "api")).strip().lower() == "local_gguf"
                    else "Локально (Ollama)"
                    if str(getattr(self.config, "meeting_summary_backend", "api")).strip().lower() == "local_ollama"
                    else "API"
                ),
                "summary_model": (
                    LocalMeetingSummaryGGUF.model_title(
                        str(getattr(self.config, "meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
                    )
                    if str(getattr(self.config, "meeting_summary_backend", "api")).strip().lower() == "local_gguf"
                    else str(getattr(self.config, "meeting_ollama_model", "") or "").strip()
                ),
                "is_running": True,
                "is_paused": False,
                "updated_at": time.time(),
            }
            cfg = AppConfig(**asdict(self.config))
            worker = threading.Thread(target=self._meeting_capture_worker, args=(cfg,), daemon=True, name="MeetingCapture")
            self._meeting_thread = worker
            worker.start()
        return True, "Сессия встречи запущена."

    def _pause_meeting_capture_from_ui(self) -> tuple[bool, str]:
        with self._meeting_lock:
            if not bool(self._meeting_state.get("is_running", False)):
                return False, "Сессия встречи не запущена."
            paused = bool(self._meeting_state.get("is_paused", False))
            self._meeting_state["is_paused"] = not paused
            self._meeting_state["state"] = "paused" if not paused else "recording"
            self._meeting_state["updated_at"] = time.time()
            return True, ("Пауза включена." if not paused else "Запись продолжена.")

    def _stop_meeting_capture_from_ui(self) -> tuple[bool, str]:
        thread = None
        with self._meeting_lock:
            if not bool(self._meeting_state.get("is_running", False)):
                return False, "Сессия встречи не запущена."
            self._meeting_state["state"] = "transcribing"
            self._meeting_state["is_running"] = False
            self._meeting_state["is_paused"] = False
            self._meeting_state["updated_at"] = time.time()
            thread = self._meeting_thread
        self._meeting_stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=4.5)
        snap = self._get_meeting_session_snapshot()
        if str(snap.get("state", "")) == "error":
            return False, str(snap.get("error", "") or "Ошибка остановки сессии.")
        return True, "Сессия встречи остановлена."

    def _get_meeting_session_status_from_ui(self) -> dict:
        return self._get_meeting_session_snapshot()

    def _retry_meeting_loopback_from_ui(self) -> tuple[bool, str]:
        with self._meeting_lock:
            running = bool(self._meeting_state.get("is_running", False))
        if not running:
            self._meeting_last_loopback_device = None
            return True, "Loopback будет повторно инициализирован при следующем запуске встречи."
        self._meeting_reinit_loopback_event.set()
        return True, "Запрошена повторная инициализация системного звука."

    def _ollama_base_url(self, cfg: Optional[AppConfig] = None) -> str:
        target_cfg = cfg or self.config
        url = str(getattr(target_cfg, "meeting_ollama_url", "http://127.0.0.1:11434") or "").strip()
        if not url:
            url = "http://127.0.0.1:11434"
        return url.rstrip("/")

    @staticmethod
    def _find_ollama_exe() -> str:
        exe_name = "ollama.exe" if os.name == "nt" else "ollama"
        path = shutil.which(exe_name)
        if path:
            return path
        if os.name == "nt":
            local = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
            if local.exists():
                return str(local)
        return ""

    def _ensure_ollama_running(self, base_url: str) -> tuple[bool, str]:
        try:
            resp = self._ollama_http.get(f"{base_url}/api/tags", timeout=(1.0, 2.5))
            if resp.ok:
                return True, "Ollama подключена"
        except Exception:
            pass
        exe = self._find_ollama_exe()
        if not exe:
            return False, "Ollama не установлена на ПК."
        try:
            if os.name == "nt":
                subprocess.Popen(
                    [exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as err:
            return False, f"Не удалось запустить Ollama: {err}"
        deadline = time.time() + 8.0
        while time.time() < deadline:
            try:
                resp = self._ollama_http.get(f"{base_url}/api/tags", timeout=(1.0, 2.5))
                if resp.ok:
                    return True, "Ollama запущена и доступна"
            except Exception:
                pass
            time.sleep(0.35)
        return False, "Ollama не отвечает. Проверьте, что служба Ollama запущена."

    def _get_ollama_status_from_ui(self) -> dict:
        cfg = AppConfig(**asdict(self.config))
        base = self._ollama_base_url(cfg)
        ok_run, msg_run = self._ensure_ollama_running(base)
        if not ok_run:
            return {"ok": False, "message": msg_run, "models": []}
        try:
            resp = self._ollama_http.get(f"{base}/api/tags", timeout=(1.0, 3.0))
            if not resp.ok:
                return {"ok": False, "message": f"Ollama недоступна: HTTP {resp.status_code}", "models": []}
            data = resp.json() if resp.content else {}
            models = [str((row or {}).get("name", "")).strip() for row in list(data.get("models") or []) if str((row or {}).get("name", "")).strip()]
            return {"ok": True, "message": "Ollama подключена", "models": models}
        except Exception as err:
            return {"ok": False, "message": f"Ollama недоступна: {err}", "models": []}

    def _get_ollama_models_from_ui(self) -> tuple[bool, str, list[str]]:
        st = self._get_ollama_status_from_ui()
        return bool(st.get("ok", False)), str(st.get("message", "")), list(st.get("models") or [])

    def _download_ollama_model_from_ui(self, model_name: str) -> tuple[bool, str]:
        cfg = AppConfig(**asdict(self.config))
        base = self._ollama_base_url(cfg)
        model = str(model_name or "").strip() or str(getattr(cfg, "meeting_ollama_model", "")).strip()
        if not model:
            return False, "Укажите имя модели Ollama."
        ok_run, msg_run = self._ensure_ollama_running(base)
        if not ok_run:
            return False, msg_run
        payload = {"name": model, "stream": False}
        try:
            resp = self._ollama_http.post(f"{base}/api/pull", json=payload, timeout=(5.0, 1800.0))
            if not resp.ok:
                body = (resp.text or "").strip()
                return False, f"Не удалось скачать модель: HTTP {resp.status_code}" + (f" ({body[:220]})" if body else "")
            try:
                data = resp.json() if resp.content else {}
                err = str((data or {}).get("error", "")).strip()
                status = str((data or {}).get("status", "")).strip().lower()
                if err:
                    return False, err
                if "success" in status or "downloaded" in status:
                    return True, f"Модель {model} скачана."
            except Exception:
                pass
            return True, f"Запрос на скачивание модели {model} выполнен."
        except Exception as err:
            return False, f"Ошибка скачивания Ollama-модели: {err}"

    def _delete_ollama_model_from_ui(self, model_name: str) -> tuple[bool, str]:
        cfg = AppConfig(**asdict(self.config))
        base = self._ollama_base_url(cfg)
        model = str(model_name or "").strip() or str(getattr(cfg, "meeting_ollama_model", "")).strip()
        if not model:
            return False, "Укажите имя модели Ollama."
        ok_run, msg_run = self._ensure_ollama_running(base)
        if not ok_run:
            return False, msg_run
        payload = {"name": model}
        try:
            resp = self._ollama_http.delete(f"{base}/api/delete", json=payload, timeout=(2.0, 30.0))
            if not resp.ok:
                # Compatibility fallback for proxies/builds where DELETE is blocked.
                resp = self._ollama_http.post(f"{base}/api/delete", json=payload, timeout=(2.0, 30.0))
            if not resp.ok:
                return False, f"Не удалось удалить модель: HTTP {resp.status_code}"
            return True, f"Модель {model} удалена."
        except Exception as err:
            return False, f"Ошибка удаления Ollama-модели: {err}"

    def _summarize_meeting_with_ollama(self, transcript: str, cfg: AppConfig) -> tuple[bool, str]:
        base = self._ollama_base_url(cfg)
        model = str(getattr(cfg, "meeting_ollama_model", "") or "").strip() or "qwen2.5:7b-instruct-q4_K_M"
        prompt = (
            "Сделай резюме встречи на русском строго в формате:\n"
            "Кратко:\n- ...\n\n"
            "Ключевые решения:\n- ...\n\n"
            "Action items:\n"
            "- Задача | Ответственный | Срок\n\n"
            "Если ответственный/срок не указаны — пиши «не указан».\n"
            "Не добавляй ничего лишнего.\n\n"
            f"Транскрипт:\n{str(transcript or '').strip()}"
        )
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        try:
            resp = self._ollama_http.post(f"{base}/api/generate", json=payload, timeout=(2.0, 90.0))
            if not resp.ok:
                return False, f"Ollama HTTP {resp.status_code}"
            data = resp.json() if resp.content else {}
            result = str(data.get("response", "") or "").strip()
            if not result:
                return False, "Пустой ответ Ollama"
            return True, result
        except Exception as err:
            logging.exception("Meeting summary Ollama failed")
            return False, str(err)

    def _build_meeting_summary_fallback(self, transcript: str) -> str:
        text = str(transcript or "").strip()
        if not text:
            return "Кратко:\nНет данных для резюме.\n\nКлючевые решения:\n- Не определены.\n\nAction items:\n- Не определены."
        sentences = [s.strip() for s in re.split(r"(?<=[.!?…])\s+", text) if s.strip()]
        short = sentences[:3] if sentences else [text[:400].strip()]
        actions_src = [s for s in sentences if re.search(r"\b(нужно|надо|сделать|подготовить|отправить|проверить|согласовать|выполнить)\b", s, re.IGNORECASE)]
        actions = actions_src[:5] if actions_src else ["Уточнить задачи и назначить ответственных по пунктам встречи."]
        lines = ["Кратко:"]
        lines.extend([f"- {s}" for s in short])
        lines.append("")
        lines.append("Ключевые решения:")
        lines.extend([f"- {s}" for s in short[:2]])
        lines.append("")
        lines.append("Action items:")
        lines.extend([f"- {s}" for s in actions])
        lines.append("")
        lines.append("Примечание: упрощённый режим без LLM-резюме.")
        return "\n".join(lines).strip()

    def _summarize_meeting_with_api(self, transcript: str, cfg: AppConfig) -> tuple[bool, str]:
        api_key = str(getattr(cfg, "openrouter_api_key", "") or "").strip()
        if not api_key:
            return False, ""
        payload = {
            "model": str(getattr(cfg, "model", "google/gemini-2.5-flash") or "google/gemini-2.5-flash"),
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Сделай резюме встречи на русском в формате:\n"
                        "1) Кратко (3-5 пунктов)\n"
                        "2) Ключевые решения (список)\n"
                        "3) Action items (пункты: задача | ответственный | срок, если не указан — «не указан»).\n"
                        "Без лишних вступлений, только полезный результат."
                    ),
                },
                {"role": "user", "content": str(transcript or "").strip()},
            ],
            "max_tokens": 550,
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            resp = self._refine_http.post(OpenRouterTranscriber.API_URL, headers=headers, json=payload, timeout=(2.0, 18.0))
            data = resp.json() if resp.content else {}
            if not resp.ok:
                msg = str(data.get("error", {}).get("message", "")).strip() or f"HTTP {resp.status_code}"
                return False, msg
            result = OpenRouterTranscriber._extract_text(data).strip()
            if not result:
                return False, "Пустой ответ модели."
            return True, result
        except Exception as err:
            logging.exception("Meeting summary API failed")
            return False, str(err)

    def _get_local_summary_model_path(self, cfg: AppConfig) -> tuple[str, str]:
        model_key = LocalMeetingSummaryGGUF.normalize_model_name(
            str(getattr(cfg, "meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
        )
        if not LocalMeetingSummaryGGUF.is_model_present(model_key):
            installed = [k for k, v in (LocalMeetingSummaryGGUF.list_installed_models() or {}).items() if v]
            if installed:
                model_key = LocalMeetingSummaryGGUF.normalize_model_name(installed[0])
        model_path = str(LocalMeetingSummaryGGUF.model_path(model_key))
        return model_key, model_path

    def _ensure_local_summary_llm(self, cfg: AppConfig) -> tuple[bool, Optional[object], str]:
        if Llama is None and GPT4All is None and AutoModelForCausalLM is None:
            return False, None, "Локальный движок GGUF недоступен: установите llama-cpp-python, gpt4all или ctransformers."
        model_key, model_path = self._get_local_summary_model_path(cfg)
        if not LocalMeetingSummaryGGUF.is_model_present(model_key):
            return False, None, f"Локальная модель резюме не загружена: {LocalMeetingSummaryGGUF.model_title(model_key)}."
        with self._summary_llm_lock:
            if self._summary_llm is not None and self._summary_llm_model_path == model_path:
                return True, self._summary_llm, ""
            n_threads = max(2, min(8, int(os.cpu_count() or 4)))
            init_errors: list[str] = []

            if Llama is not None:
                try:
                    llm = Llama(
                        model_path=model_path,
                        n_ctx=4096,
                        n_threads=n_threads,
                        n_batch=512,
                        n_gpu_layers=0,
                        verbose=False,
                    )
                    self._summary_llm = llm
                    self._summary_llm_model_path = model_path
                    self._summary_llm_kind = "llama_cpp"
                    return True, llm, ""
                except Exception as err:
                    init_errors.append(f"llama.cpp: {err}")
                    logging.exception("Failed to init llama.cpp summary model: %s", model_path)

            if GPT4All is not None:
                try:
                    mp = Path(model_path)
                    llm = GPT4All(
                        model_name=mp.name,
                        model_path=str(mp.parent),
                        allow_download=False,
                        n_threads=n_threads,
                        device="cpu",
                        n_ctx=4096,
                        ngl=0,
                        verbose=False,
                    )
                    self._summary_llm = llm
                    self._summary_llm_model_path = model_path
                    self._summary_llm_kind = "gpt4all"
                    return True, llm, ""
                except Exception as err:
                    init_errors.append(f"gpt4all: {err}")
                    logging.exception("Failed to init gpt4all summary model: %s", model_path)

            # Optional last-resort fallback for environments where ctransformers works.
            if AutoModelForCausalLM is not None:
                try:
                    llm = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        gpu_layers=0,
                        threads=n_threads,
                        context_length=4096,
                    )
                    self._summary_llm = llm
                    self._summary_llm_model_path = model_path
                    self._summary_llm_kind = "ctransformers"
                    return True, llm, ""
                except Exception as err:
                    init_errors.append(f"ctransformers: {err}")
                    logging.exception("Failed to init ctransformers summary model: %s", model_path)

            self._summary_llm = None
            self._summary_llm_model_path = ""
            self._summary_llm_kind = ""
            err_text = "; ".join(init_errors).strip() or "Локальный движок GGUF не инициализирован."
            return False, None, err_text

    def _summarize_meeting_with_local_gguf(self, transcript: str, cfg: AppConfig) -> tuple[bool, str]:
        ok, llm, err = self._ensure_local_summary_llm(cfg)
        if not ok or llm is None:
            return False, err or "Локальная GGUF-модель недоступна."
        prompt = (
            "Ты помощник по резюме встреч. Ответ строго на русском и строго в формате:\n"
            "Кратко:\n- ...\n\n"
            "Ключевые решения:\n- ...\n\n"
            "Action items:\n- Задача | Ответственный | Срок\n\n"
            "Если ответственный или срок не указаны — пиши «не указан».\n"
            "Не добавляй комментарии вне формата.\n\n"
            f"Транскрипт встречи:\n{str(transcript or '').strip()}\n\n"
            "Результат:\n"
        )
        try:
            text = ""
            llm_kind = str(self._summary_llm_kind or "").strip().lower()
            if llm_kind == "llama_cpp":
                result = llm(
                    prompt,
                    max_tokens=560,
                    temperature=0.15,
                    top_p=0.9,
                    stop=["\n\n\n"],
                )
                if isinstance(result, dict):
                    choices = result.get("choices") or []
                    if choices:
                        text = str((choices[0] or {}).get("text", "") or "").strip()
            elif llm_kind == "gpt4all":
                text = str(
                    llm.generate(
                        prompt,
                        max_tokens=560,
                        temp=0.15,
                        top_p=0.9,
                    )
                    or ""
                ).strip()
            elif llm_kind == "ctransformers":
                text = str(
                    llm(
                        prompt,
                        max_new_tokens=560,
                        temperature=0.15,
                        top_p=0.9,
                    )
                    or ""
                ).strip()
            if not text:
                return False, "Пустой ответ локальной GGUF-модели."
            return True, text
        except Exception as err:
            logging.exception("Meeting summary local GGUF failed")
            return False, str(err)

    def _build_meeting_summary_from_ui(self) -> tuple[bool, str]:
        with self._meeting_lock:
            if self._meeting_summary_worker and self._meeting_summary_worker.is_alive():
                return False, "Резюме уже формируется."
            transcript = str(self._meeting_state.get("transcript_text", "")).strip()
            if not transcript:
                return False, "Нет транскрипта для формирования резюме."
            cfg = AppConfig(**asdict(self.config))
            self._meeting_state["state"] = "summarizing"
            self._meeting_state["error"] = ""
            self._meeting_state["updated_at"] = time.time()

        def _worker() -> None:
            mode = str(getattr(cfg, "meeting_summary_backend", "api") or "api").strip().lower()
            used_mode = "api"
            ok = False
            summary = ""
            if mode == "local_gguf":
                used_mode = "local_gguf"
                ok, summary = self._summarize_meeting_with_local_gguf(transcript, cfg)
                if not ok:
                    api_ok, api_summary = self._summarize_meeting_with_api(transcript, cfg)
                    if api_ok:
                        ok, summary = True, api_summary
                        used_mode = "api-fallback"
            elif mode == "local_ollama":
                used_mode = "local_ollama"
                ok, summary = self._summarize_meeting_with_ollama(transcript, cfg)
                if not ok:
                    api_ok, api_summary = self._summarize_meeting_with_api(transcript, cfg)
                    if api_ok:
                        ok, summary = True, api_summary
                        used_mode = "api-fallback"
            else:
                ok, summary = self._summarize_meeting_with_api(transcript, cfg)
            if not ok:
                summary = self._build_meeting_summary_fallback(transcript)
                used_mode = "fallback"
            self._meeting_update(
                summary_text=summary,
                state="done",
                summary_backend=(
                    "Локально (GGUF)" if used_mode == "local_gguf"
                    else "Локально (Ollama)" if used_mode == "local_ollama"
                    else "API" if used_mode == "api"
                    else "API (fallback)" if used_mode == "api-fallback"
                    else "Упрощённый fallback"
                ),
                summary_model=(
                    LocalMeetingSummaryGGUF.model_title(
                        str(getattr(cfg, "meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
                    )
                    if used_mode == "local_gguf"
                    else str(getattr(cfg, "meeting_ollama_model", "") or "").strip()
                    if used_mode == "local_ollama"
                    else str(getattr(cfg, "model", "") or "").strip()
                ),
                error=(
                    ""
                    if ok
                    else "Использован fallback режим резюме."
                ),
            )

        th = threading.Thread(target=_worker, daemon=True, name="MeetingSummary")
        with self._meeting_lock:
            self._meeting_summary_worker = th
        th.start()
        return True, "Формирование резюме запущено."

    def _copy_meeting_text_from_ui(self, kind: str) -> tuple[bool, str]:
        snap = self._get_meeting_session_snapshot()
        mode = str(kind or "transcript").strip().lower()
        if mode == "summary":
            text = str(snap.get("summary_text", "")).strip()
        else:
            text = str(snap.get("transcript_text", "")).strip()
        if not text:
            return False, "Нет текста для копирования."
        try:
            self._copy_text(text)
            return True, "Текст скопирован."
        except Exception as err:
            return False, f"Ошибка: {err}"

    def _export_meeting_txt_from_ui(self, target_path: Optional[str] = None) -> tuple[bool, str]:
        snap = self._get_meeting_session_snapshot()
        transcript = str(snap.get("transcript_text", "")).strip()
        summary = str(snap.get("summary_text", "")).strip()
        if not transcript and not summary:
            return False, "Нет данных для экспорта."
        started = float(snap.get("started_at", 0.0) or 0.0)
        ts = datetime.fromtimestamp(started if started > 0 else time.time()).strftime("%Y%m%d_%H%M%S")
        out_path: Path
        target_raw = str(target_path or "").strip()
        if target_raw:
            out_path = Path(target_raw).expanduser()
            if out_path.suffix.lower() != ".txt":
                out_path = out_path.with_suffix(".txt")
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = MEETINGS_DIR / f"meeting_{ts}.txt"
        try:
            payload = [
                f"{APP_NAME} В· Voice Meads AI",
                f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
                f"Режим: {str(snap.get('backend_label', '')).strip() or 'Не указан'}",
                "",
                "Транскрипт:",
                transcript or "—",
                "",
                "Резюме встречи:",
                summary or "—",
                "",
            ]
            out_path.write_text("\n".join(payload), encoding="utf-8")
            return True, f"Экспортировано: {out_path}"
        except Exception as err:
            logging.exception("Failed to export meeting txt")
            return False, f"Ошибка экспорта: {err}"

    @staticmethod
    def _count_words(text: str) -> int:
        if not text:
            return 0
        words = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+(?:[-'][0-9A-Za-zА-Яа-яЁё]+)?", text)
        return len(words)

    def _update_stats(self, text: str, duration_sec: float, usage: Optional[dict] = None) -> None:
        words = self._count_words(text)
        input_tokens = 0
        output_tokens = 0
        if isinstance(usage, dict):
            try:
                input_tokens = max(0, int(usage.get("input_tokens", 0) or 0))
            except Exception:
                input_tokens = 0
            try:
                output_tokens = max(0, int(usage.get("output_tokens", 0) or 0))
            except Exception:
                output_tokens = 0

        if words <= 0 and input_tokens <= 0 and output_tokens <= 0:
            return
        with self._stats_lock:
            stats = self.stats_store.load()
            stats.total_words = max(0, int(stats.total_words)) + words
            stats.today_words = max(0, int(stats.today_words)) + words
            stats.total_input_tokens = max(0, int(stats.total_input_tokens)) + input_tokens
            stats.total_output_tokens = max(0, int(stats.total_output_tokens)) + output_tokens
            stats.today_input_tokens = max(0, int(stats.today_input_tokens)) + input_tokens
            stats.today_output_tokens = max(0, int(stats.today_output_tokens)) + output_tokens
            stats.total_audio_seconds = max(0.0, float(stats.total_audio_seconds)) + max(0.0, float(duration_sec))
            self.stats_store.save(stats)

    def _get_stats_snapshot(self) -> dict:
        with self._stats_lock:
            stats = self.stats_store.load()
        total_words = max(0, int(stats.total_words))
        today_words = max(0, int(stats.today_words))
        total_input_tokens = max(0, int(stats.total_input_tokens))
        total_output_tokens = max(0, int(stats.total_output_tokens))
        today_input_tokens = max(0, int(stats.today_input_tokens))
        today_output_tokens = max(0, int(stats.today_output_tokens))
        total_audio_seconds = max(0.0, float(stats.total_audio_seconds))
        speed_wpm = int(round(total_words * 60.0 / total_audio_seconds)) if total_audio_seconds > 0.1 else 0
        return {
            "total_words": total_words,
            "today_words": today_words,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "today_input_tokens": today_input_tokens,
            "today_output_tokens": today_output_tokens,
            "total_audio_seconds": total_audio_seconds,
            "speed_wpm": speed_wpm,
            "today_iso_date": stats.today_iso_date or date.today().isoformat(),
        }

    def _reset_stats_from_ui(self) -> tuple[bool, str, dict]:
        try:
            with self._stats_lock:
                stats = self.stats_store.reset()
                total_words = max(0, int(stats.total_words))
                today_words = max(0, int(stats.today_words))
                total_input_tokens = max(0, int(stats.total_input_tokens))
                total_output_tokens = max(0, int(stats.total_output_tokens))
                today_input_tokens = max(0, int(stats.today_input_tokens))
                today_output_tokens = max(0, int(stats.today_output_tokens))
                total_audio_seconds = max(0.0, float(stats.total_audio_seconds))
                speed_wpm = int(round(total_words * 60.0 / total_audio_seconds)) if total_audio_seconds > 0.1 else 0
                snapshot = {
                    "total_words": total_words,
                    "today_words": today_words,
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "today_input_tokens": today_input_tokens,
                    "today_output_tokens": today_output_tokens,
                    "total_audio_seconds": total_audio_seconds,
                    "speed_wpm": speed_wpm,
                    "today_iso_date": stats.today_iso_date or date.today().isoformat(),
                }
            self._notify("Statistics reset")
            return True, "Статистика сброшена", snapshot
        except Exception as err:
            logging.exception("Failed to reset stats")
            return False, f"Ошибка: {err}", self._get_stats_snapshot()

    def _append_history_entry(self, text: str) -> None:
        if not self.config.history_enabled:
            return
        normalized = str(text or "").strip()
        if not normalized:
            return
        with self._history_lock:
            self.history_store.add(normalized)

    def _apply_autoreplace(self, text: str) -> str:
        if not self.config.autoreplace_enabled:
            return text
        out = str(text or "")
        with self._rules_lock:
            rules = list(self._autoreplace_rules_cache)
        for rule in rules:
            try:
                if not bool(rule.get("enabled", True)):
                    continue
                pattern = str(rule.get("pattern", ""))
                if not pattern:
                    continue
                replacement = str(rule.get("replacement", ""))
                mode = str(rule.get("mode", "Text")).lower()
                if mode in {"regexp", "regex", "rx"}:
                    flags_value = 0
                    flags = str(rule.get("flags", "u")).lower()
                    if "i" in flags:
                        flags_value |= re.IGNORECASE
                    if "m" in flags:
                        flags_value |= re.MULTILINE
                    if "s" in flags:
                        flags_value |= re.DOTALL
                    if "u" in flags:
                        flags_value |= re.UNICODE
                    out = re.sub(pattern, replacement, out, flags=flags_value)
                else:
                    out = out.replace(pattern, replacement)
            except Exception:
                logging.exception("AutoReplace rule failed: %s", rule.get("id", "unknown"))
        return out

    def _get_history_snapshot(self) -> dict:
        with self._history_lock:
            items = self.history_store.load()
        return {
            "enabled": bool(self.config.history_enabled),
            "items": items,
        }

    def _get_autoreplace_snapshot(self) -> dict:
        with self._rules_lock:
            rules = list(self._autoreplace_rules_cache)
        return {
            "enabled": bool(self.config.autoreplace_enabled),
            "rules": rules,
        }

    def _set_autoreplace_snapshot_from_ui(self, enabled: bool, rules: list[dict]) -> tuple[bool, str, dict]:
        try:
            normalized: list[dict] = []
            for row in list(rules):
                if not isinstance(row, dict):
                    continue
                normalized.append(
                    {
                        "id": str(row.get("id", "")).strip() or f"rule-{int(time.time() * 1000)}",
                        "enabled": bool(row.get("enabled", True)),
                        "pattern": str(row.get("pattern", "")),
                        "replacement": str(row.get("replacement", "")),
                        "mode": "RegExp" if str(row.get("mode", "Text")).lower() in {"regexp", "regex", "rx"} else "Text",
                        "flags": str(row.get("flags", "u")),
                    }
                )
            with self._rules_lock:
                self.autoreplace_store.save(normalized)
                self._autoreplace_rules_cache = self.autoreplace_store.load()
            self.config.autoreplace_enabled = bool(enabled)
            self.config_store.save(self.config)
            return True, "Автозамена обновлена", self._get_autoreplace_snapshot()
        except Exception as err:
            logging.exception("Failed to update autoreplace rules")
            return False, f"Ошибка: {err}", self._get_autoreplace_snapshot()

    def _reset_autoreplace_from_ui(self) -> tuple[bool, str, dict]:
        try:
            with self._rules_lock:
                self._autoreplace_rules_cache = self.autoreplace_store.reset()
            return True, "Правила автозамены сброшены", self._get_autoreplace_snapshot()
        except Exception as err:
            logging.exception("Failed to reset autoreplace rules")
            return False, f"Ошибка: {err}", self._get_autoreplace_snapshot()

    def _set_history_enabled_from_ui(self, enabled: bool) -> tuple[bool, str, dict]:
        try:
            self.config.history_enabled = bool(enabled)
            self.config_store.save(self.config)
            return True, "Настройка истории обновлена", self._get_history_snapshot()
        except Exception as err:
            logging.exception("Failed to set history enabled")
            return False, f"Ошибка: {err}", self._get_history_snapshot()

    def _delete_history_item_from_ui(self, item_id: str) -> tuple[bool, str, dict]:
        try:
            with self._history_lock:
                self.history_store.delete(item_id)
            return True, "Запись удалена", self._get_history_snapshot()
        except Exception as err:
            logging.exception("Failed to delete history item")
            return False, f"Ошибка: {err}", self._get_history_snapshot()

    def _clear_history_from_ui(self) -> tuple[bool, str, dict]:
        try:
            with self._history_lock:
                self.history_store.clear()
            return True, "История очищена", self._get_history_snapshot()
        except Exception as err:
            logging.exception("Failed to clear history")
            return False, f"Ошибка: {err}", self._get_history_snapshot()

    def _prepare_text_for_paste(self, text: str, target_hwnd: int) -> str:
        if not text:
            return text
        # For consecutive dictation inserts in the same window, prefix a separator space
        # so chunks don't merge as "...text.Text".
        if (
            self._has_successful_insert
            and self._last_insert_target_hwnd
            and self._last_insert_target_hwnd == target_hwnd
            and not text[:1].isspace()
        ):
            return " " + text
        return text

    def _copy_text(self, text: str) -> None:
        with self._clipboard_lock:
            pyperclip.copy(text)

    def _paste_text(self, text: str, target_hwnd: int) -> bool:
        with self._clipboard_lock:
            previous = None
            if self.config.restore_clipboard:
                try:
                    previous = pyperclip.paste()
                except Exception:
                    previous = None

            try:
                pyperclip.copy(text)
                time.sleep(0.05)
                self._focus_window(target_hwnd)
                time.sleep(0.06)
                self._ignore_hotkey_until = time.monotonic() + 0.8
                if not self._send_paste_shortcut():
                    return False
            except Exception:
                logging.exception("Paste simulation failed")
                return False
            finally:
                if self.config.restore_clipboard and previous is not None:
                    def restore() -> None:
                        time.sleep(0.35)
                        try:
                            pyperclip.copy(previous)
                        except Exception:
                            logging.exception("Clipboard restore failed")

                    threading.Thread(target=restore, daemon=True).start()

        return True

    @staticmethod
    def _send_paste_shortcut() -> bool:
        sent_any = False
        for shortcut in ("ctrl+v", "shift+insert"):
            try:
                keyboard.send(shortcut)
                sent_any = True
                time.sleep(0.07)
            except Exception:
                logging.exception("Failed paste shortcut: %s", shortcut)

        return sent_any

    def _set_last_transcript(self, text: str) -> None:
        cache_path = CONFIG_DIR / "last_transcript.txt"
        try:
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            logging.exception("Failed to save last transcript")

    def _get_last_transcript(self) -> str:
        cache_path = CONFIG_DIR / "last_transcript.txt"
        if not cache_path.exists():
            return ""
        try:
            return cache_path.read_text(encoding="utf-8")
        except Exception:
            logging.exception("Failed to read last transcript")
            return ""

    def _on_copy_last(self, _icon: Icon, _item: MenuItem) -> None:
        text = self._get_last_transcript()
        if not text:
            self._notify("No transcript yet")
            return
        self._copy_text(text)
        self._notify("Last transcript copied")

    @staticmethod
    def _enumerate_input_devices_for_ui() -> list[str]:
        return AudioRecorder.list_input_devices()

    def _list_input_devices_for_ui(self, force_refresh: bool = False) -> list[str]:
        now = time.monotonic()
        ttl_sec = 20.0
        with self._ui_devices_cache_lock:
            cached = self._ui_devices_cache["input"]
            if not force_refresh and (now - float(cached.get("ts", 0.0) or 0.0)) <= ttl_sec:
                return list(cached.get("data") or [])
        data = self._enumerate_input_devices_for_ui()
        with self._ui_devices_cache_lock:
            self._ui_devices_cache["input"] = {"ts": now, "data": list(data or [])}
        return list(data or [])

    @staticmethod
    def _enumerate_system_audio_devices_for_ui() -> list[dict]:
        items: list[dict] = []
        seen: set[str] = set()
        try:
            if sc is not None:
                for mic in list(sc.all_microphones(include_loopback=True) or []):
                    dev_id = str(getattr(mic, "id", "") or "").strip()
                    name = str(getattr(mic, "name", "") or "").strip()
                    if not dev_id or not name:
                        continue
                    key = f"sc::{dev_id}"
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append({"id": key, "label": f"Loopback (soundcard): {name}"})
        except Exception:
            logging.exception("Failed to enumerate soundcard loopback devices for UI")
        try:
            devices = list(sd.query_devices() or [])
            hostapis = list(sd.query_hostapis() or [])
            for idx, dev in enumerate(devices):
                try:
                    name = str(dev.get("name", "") or "").strip()
                    if not name:
                        continue
                    max_in = int(dev.get("max_input_channels", 0) or 0)
                    max_out = int(dev.get("max_output_channels", 0) or 0)
                    host_idx = int(dev.get("hostapi", -1))
                    host_name = str((hostapis[host_idx] or {}).get("name", "")).lower() if 0 <= host_idx < len(hostapis) else ""
                    low = name.lower()
                    if max_in > 0 and any(k in low for k in ("stereo mix", "what u hear", "wave out", "loopback", "стерео микшер")):
                        key = f"si::{idx}"
                        if key not in seen:
                            seen.add(key)
                            items.append({"id": key, "label": f"Системный вход: {name}"})
                    if max_out > 0 and "wasapi" in host_name:
                        key = f"wa::{idx}"
                        if key not in seen:
                            seen.add(key)
                            items.append({"id": key, "label": f"WASAPI loopback: {name}"})
                except Exception:
                    continue
        except Exception:
            logging.exception("Failed to enumerate sounddevice system devices for UI")
        return items

    def _list_system_audio_devices_for_ui(self, force_refresh: bool = False) -> list[dict]:
        now = time.monotonic()
        ttl_sec = 20.0
        with self._ui_devices_cache_lock:
            cached = self._ui_devices_cache["system"]
            if not force_refresh and (now - float(cached.get("ts", 0.0) or 0.0)) <= ttl_sec:
                return list(cached.get("data") or [])
        data = self._enumerate_system_audio_devices_for_ui()
        with self._ui_devices_cache_lock:
            self._ui_devices_cache["system"] = {"ts": now, "data": list(data or [])}
        return list(data or [])

    def _get_system_audio_devices_from_ui(self) -> list[dict]:
        try:
            return self._list_system_audio_devices_for_ui(force_refresh=True)
        except Exception as err:
            logging.exception("Failed to get system audio devices from UI")
            return [{"id": "", "label": f"Ошибка списка устройств: {err}"}]

    def _open_windows_sound_settings_from_ui(self) -> tuple[bool, str]:
        try:
            subprocess.Popen(["cmd", "/c", "start", "ms-settings:sound"], shell=False)
            return True, "Открыты параметры звука Windows"
        except Exception as err:
            logging.exception("Failed to open Windows sound settings")
            return False, f"Ошибка: {err}"

    def _test_microphone_from_ui(self, device_name: str) -> tuple[bool, str]:
        try:
            sample_rate = int(self.config.sample_rate)
            frames = int(sample_rate * 1.8)
            device_index = AudioRecorder.resolve_device_index(device_name)
            recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32", device=device_index)
            sd.wait()
            audio = np.asarray(recording).reshape(-1)
            if audio.size == 0:
                return False, "Нет данных с микрофона"
            peak = float(np.max(np.abs(audio)))
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if peak < 0.003 and rms < 0.0007:
                return False, "Сигнал слишком тихий или микрофон выключен"
            return True, "Микрофон работает корректно"
        except Exception as err:
            logging.exception("Microphone test failed")
            return False, f"Ошибка: {err}"

    def _open_logs_from_ui(self) -> tuple[bool, str]:
        try:
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            if not LOG_PATH.exists():
                LOG_PATH.write_text("", encoding="utf-8")
            subprocess.Popen(["explorer", "/select,", str(LOG_PATH)], shell=False)
            return True, "Открыт файл логов"
        except Exception as err:
            logging.exception("Open logs failed")
            return False, f"Ошибка: {err}"

    def _start_local_model_download_from_ui(self, engine: str, model_name: str) -> tuple[bool, str]:
        try:
            return self.start_local_model_download(engine, model_name)
        except Exception as err:
            logging.exception("Failed to start local model download")
            return False, f"Ошибка запуска загрузки: {err}"

    def _get_local_model_download_status_from_ui(self, engine: str) -> dict:
        try:
            return self.get_local_model_download_status(engine)
        except Exception as err:
            logging.exception("Failed to read local model download status")
            return {
                "state": "error",
                "engine": str(engine or "whisper"),
                "model": "",
                "progress_percent": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "message": "Ошибка чтения статуса загрузки",
                "started_at": 0.0,
                "finished_at": 0.0,
                "error": str(err),
            }

    def _delete_local_model_from_ui(self, engine: str, model_name: str) -> tuple[bool, str, dict]:
        engine_name, model = self._normalize_local_download_target(engine, model_name)
        model_title = self._local_model_title(engine_name, model)
        model_cls = (
            LocalVoskTranscriber
            if engine_name == "vosk"
            else LocalSherpaWhispeRuTranscriber
            if engine_name == "sherpa"
            else LocalWhisperSmallTranscriber
        )
        try:
            status = self.get_local_model_download_status(engine_name)
            if (
                status.get("state") in {"preparing", "downloading"}
                and status.get("model") == model
                and str(status.get("engine", "whisper")).strip().lower() == engine_name
            ):
                return False, f"Нельзя удалить модель {model_title}: сейчас идёт скачивание", self._settings_snapshot()
            removed_count = model_cls.remove_cached_model(model)
            current_dl = self.get_local_model_download_status(engine_name)
            if str(current_dl.get("model", "")).strip() == model:
                self._set_model_download_status(
                    state="idle",
                    engine=engine_name,
                    model=model,
                    progress_percent=0.0,
                    downloaded_bytes=0,
                    total_bytes=0,
                    message="Ожидание",
                    finished_at=time.time(),
                    error="",
                )
            if isinstance(self.transcriber, (LocalWhisperSmallTranscriber, LocalVoskTranscriber, LocalSherpaWhispeRuTranscriber)):
                current_model = str(getattr(self.transcriber, "_selected_model", "") or "").strip().lower()
                current_engine = (
                    "vosk"
                    if isinstance(self.transcriber, LocalVoskTranscriber)
                    else "sherpa"
                    if isinstance(self.transcriber, LocalSherpaWhispeRuTranscriber)
                    else "whisper"
                )
                if current_model == model and current_engine == engine_name:
                    try:
                        if hasattr(self.transcriber, "_model"):
                            self.transcriber._model = None
                        if hasattr(self.transcriber, "_recognizer"):
                            self.transcriber._recognizer = None
                        self.transcriber._set_runtime("init", "Модель удалена с диска. Будет загружена заново при старте записи.")
                    except Exception:
                        logging.exception("Failed to reset local transcriber runtime after model deletion")
            if removed_count <= 0:
                return True, f"Кэш модели {model_title} не найден", self._settings_snapshot()
            return True, f"Модель {model_title} удалена (элементов: {removed_count})", self._settings_snapshot()
        except Exception as err:
            logging.exception("Failed to delete local model: %s", model)
            return False, f"Ошибка удаления модели {model_title}: {err}", self._settings_snapshot()

    def _start_summary_model_download_from_ui(self, model_name: str) -> tuple[bool, str]:
        try:
            return self.start_summary_model_download(model_name)
        except Exception as err:
            logging.exception("Failed to start summary model download")
            return False, f"Ошибка запуска загрузки: {err}"

    def _get_summary_model_download_status_from_ui(self) -> dict:
        try:
            return self.get_summary_model_download_status()
        except Exception as err:
            logging.exception("Failed to read summary model download status")
            return {
                "state": "error",
                "model": LocalMeetingSummaryGGUF.DEFAULT_MODEL,
                "progress_percent": 0.0,
                "downloaded_bytes": 0,
                "total_bytes": 0,
                "message": "Ошибка чтения статуса загрузки модели резюме",
                "started_at": 0.0,
                "finished_at": 0.0,
                "error": str(err),
                "installed_models": LocalMeetingSummaryGGUF.list_installed_models(),
                "installed_sizes": LocalMeetingSummaryGGUF.list_installed_sizes(),
            }

    def _delete_summary_model_from_ui(self, model_name: str) -> tuple[bool, str]:
        return self.delete_summary_model(model_name)

    def _startup_command(self) -> str:
        exe = str(Path(sys.executable).resolve())
        if getattr(sys, "frozen", False):
            return f"\"{exe}\""
        launcher = str((self.base_dir / "launcher.py").resolve())
        return f"\"{exe}\" \"{launcher}\""

    @staticmethod
    def _startup_cmd_path() -> Path:
        startup_dir = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        return startup_dir / "Voice PRO AI.cmd"

    def _apply_windows_schtask_autostart(self, enabled: bool) -> None:
        if not enabled:
            try:
                subprocess.run(
                    ["schtasks", "/Delete", "/TN", AUTOSTART_TASK_NAME, "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=False,
                )
            except Exception:
                logging.exception("Failed to delete scheduled task autostart")
            return

        command = self._startup_command()
        # /RL LIMITED avoids elevation prompts on user logon.
        args = [
            "schtasks",
            "/Create",
            "/TN",
            AUTOSTART_TASK_NAME,
            "/SC",
            "ONLOGON",
            "/TR",
            command,
            "/RL",
            "LIMITED",
            "/F",
        ]
        result = subprocess.run(args, check=False, capture_output=True, text=True, shell=False)
        if int(result.returncode) != 0:
            stderr = (result.stderr or "").strip()
            stdout = (result.stdout or "").strip()
            details = stderr or stdout or "unknown error"
            raise RuntimeError(f"Task Scheduler: {details}")

    def _apply_windows_startup(self, enabled: bool) -> None:
        ok_any = False
        errors: list[str] = []

        if winreg is not None:
            try:
                key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
                with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    if enabled:
                        winreg.SetValueEx(key, AUTOSTART_VALUE_NAME, 0, winreg.REG_SZ, self._startup_command())
                    else:
                        for value_name in (AUTOSTART_VALUE_NAME, APP_NAME):
                            try:
                                winreg.DeleteValue(key, value_name)
                            except FileNotFoundError:
                                pass
                ok_any = True
            except Exception as err:
                logging.exception("Failed to update Run registry autostart")
                errors.append(str(err))

        try:
            cmd_path = self._startup_cmd_path()
            cmd_path.parent.mkdir(parents=True, exist_ok=True)
            if enabled:
                cmd_path.write_text(f"@echo off\r\nstart \"\" {self._startup_command()}\r\n", encoding="utf-8")
            else:
                if cmd_path.exists():
                    cmd_path.unlink()
            ok_any = True
        except Exception as err:
            logging.exception("Failed to update Startup-folder autostart")
            errors.append(str(err))

        try:
            self._apply_windows_schtask_autostart(enabled)
            ok_any = True
        except Exception as err:
            logging.exception("Failed to update Task Scheduler autostart")
            errors.append(str(err))

        if enabled and not ok_any:
            raise RuntimeError("; ".join(errors) if errors else "Autostart setup failed")

    def _apply_run_as_admin(self, enabled: bool) -> None:
        if winreg is None:
            return
        key_path = r"Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers"
        exe = str(Path(sys.executable).resolve())
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            if enabled:
                winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, "~ RUNASADMIN")
            else:
                try:
                    winreg.DeleteValue(key, exe)
                except FileNotFoundError:
                    pass

    def _check_updates_from_ui(self) -> tuple[bool, str, dict, Optional[dict]]:
        return self._check_for_updates(notify_if_latest=False)

    def _start_update_install_from_ui(self, update_payload: dict) -> tuple[bool, str, dict]:
        return self._start_update_install(dict(update_payload or {}))

    def _get_updater_status_from_ui(self) -> dict:
        return self._get_updater_status()

    def _schedule_auto_update_check(self) -> None:
        if not bool(getattr(self.config, "auto_download_updates", False)):
            return
        if not self._effective_update_feed_url():
            return

        def _worker() -> None:
            try:
                time.sleep(6.0)
                ok, message, _snap, update_info = self._check_for_updates(notify_if_latest=False)
                if ok and update_info:
                    ver = str(update_info.get("version", "") or "").strip()
                    if ver:
                        self._notify(f"Доступно обновление {ver}. Откройте настройки для установки.")
                elif not ok:
                    logging.info("Auto update check failed: %s", message)
            except Exception:
                logging.exception("Auto update check worker failed")

        threading.Thread(target=_worker, name="VoiceProAutoUpdateCheck", daemon=True).start()

    def _settings_snapshot(self) -> dict:
        snap = asdict(self.config)
        snap["_app_version"] = APP_VERSION
        snap["_audio_devices"] = self._list_input_devices_for_ui()
        snap["_system_audio_devices"] = self._list_system_audio_devices_for_ui()
        snap["_transcriber_status"] = self._get_transcriber_runtime_status()
        snap["_updater_status"] = self._get_updater_status()
        snap["_model_download_status"] = self.get_local_model_download_status(
            str(getattr(self.config, "local_backend_engine", "whisper") or "whisper")
        )
        snap["_summary_model_download_status"] = self.get_summary_model_download_status()
        snap["_media_queue"] = self._get_media_queue_snapshot()
        snap["_meeting_session"] = self._get_meeting_session_snapshot()
        # Keep settings opening fast: Ollama status can block for several seconds
        # when local service is unavailable. UI fetches it on-demand.
        snap["_ollama_status"] = {"ok": False, "message": "Проверка по запросу", "models": []}
        return snap

    def _reset_app_settings_from_ui(self) -> tuple[bool, str, dict]:
        try:
            old_api_key = self.config.openrouter_api_key
            self.config = AppConfig()
            # Keep API key on reset for user convenience.
            self.config.openrouter_api_key = old_api_key
            self.config_store.save(self.config)
            self.transcriber = self._create_transcriber(self.config)
            self.recorder.sample_rate = self.config.sample_rate
            self.recorder.set_input_device(self.config.input_device)
            self._bind_hotkey()
            self._apply_windows_startup(self.config.autostart_windows)
            self._apply_run_as_admin(self.config.run_as_admin)
            supports_preview = False
            try:
                supports_preview = bool(self.transcriber.supports_live_preview())
            except Exception:
                supports_preview = False
            self.overlay.set_live_preview_enabled(bool(self.config.show_live_preview_text and self.config.show_overlay_widget and supports_preview))
            self.overlay.set_mic_quality_enabled(bool(self.config.show_mic_quality_indicator and self.config.show_overlay_widget))
            self.overlay.clear_mic_quality_text()
            return True, "Настройки сброшены", self._settings_snapshot()
        except Exception as err:
            logging.exception("Reset settings failed")
            return False, f"Ошибка: {err}", self._settings_snapshot()

    def _on_open_settings(self, _icon: Optional[Icon], _item: Optional[MenuItem]) -> None:
        try:
            data = self._settings_snapshot()
            data["_stats"] = self._get_stats_snapshot()
            data["_history"] = self._get_history_snapshot()
            data["_autoreplace"] = self._get_autoreplace_snapshot()
            self.overlay.open_settings(
                data,
                self._apply_settings_from_ui,
                self._notify,
                on_open_cb=self._on_settings_opened,
                on_close_cb=self._on_settings_closed,
                on_reset_stats_cb=self._reset_stats_from_ui,
                on_set_history_enabled_cb=self._set_history_enabled_from_ui,
                on_delete_history_item_cb=self._delete_history_item_from_ui,
                on_clear_history_cb=self._clear_history_from_ui,
                on_set_autoreplace_cb=self._set_autoreplace_snapshot_from_ui,
                on_reset_autoreplace_cb=self._reset_autoreplace_from_ui,
                on_open_windows_sound_cb=self._open_windows_sound_settings_from_ui,
                on_test_microphone_cb=self._test_microphone_from_ui,
                on_open_logs_cb=self._open_logs_from_ui,
                on_start_local_model_download_cb=self._start_local_model_download_from_ui,
                on_get_local_model_download_status_cb=self._get_local_model_download_status_from_ui,
                on_delete_local_model_cb=self._delete_local_model_from_ui,
                on_reset_app_settings_cb=self._reset_app_settings_from_ui,
                on_add_media_files_cb=self._add_media_files_from_ui,
                on_get_media_queue_cb=self._get_media_queue_from_ui,
                on_start_media_processing_cb=self._start_media_processing_from_ui,
                on_clear_media_queue_cb=self._clear_media_queue_from_ui,
                on_copy_media_result_cb=self._copy_media_result_from_ui,
                on_export_media_result_cb=self._export_media_result_from_ui,
                on_start_meeting_capture_cb=self._start_meeting_capture_from_ui,
                on_pause_meeting_capture_cb=self._pause_meeting_capture_from_ui,
                on_stop_meeting_capture_cb=self._stop_meeting_capture_from_ui,
                on_get_meeting_session_status_cb=self._get_meeting_session_status_from_ui,
                on_build_meeting_summary_cb=self._build_meeting_summary_from_ui,
                on_copy_meeting_text_cb=self._copy_meeting_text_from_ui,
                on_export_meeting_txt_cb=self._export_meeting_txt_from_ui,
                on_retry_meeting_loopback_cb=self._retry_meeting_loopback_from_ui,
                on_get_system_audio_devices_cb=self._get_system_audio_devices_from_ui,
                on_get_ollama_status_cb=self._get_ollama_status_from_ui,
                on_get_ollama_models_cb=self._get_ollama_models_from_ui,
                on_download_ollama_model_cb=self._download_ollama_model_from_ui,
                on_delete_ollama_model_cb=self._delete_ollama_model_from_ui,
                on_start_summary_model_download_cb=self._start_summary_model_download_from_ui,
                on_get_summary_model_download_status_cb=self._get_summary_model_download_status_from_ui,
                on_delete_summary_model_cb=self._delete_summary_model_from_ui,
                on_check_updates_cb=self._check_updates_from_ui,
                on_start_update_install_cb=self._start_update_install_from_ui,
                on_get_updater_status_cb=self._get_updater_status_from_ui,
            )
        except Exception:
            logging.exception("Failed to open settings window")
            self._notify("Не удалось открыть настройки. Проверьте app.log")

    def _on_settings_opened(self) -> None:
        with self._hotkey_lock:
            if self._settings_open:
                return
            self._settings_open = True
            self._pressed_keys = set()
            self._ptt_armed = False
        logging.info("Settings opened: hotkey temporarily gated")
        # Settings window should never leave floating "listening" overlays visible.
        self._stop_live_preview_loop()
        self._stop_mic_quality_loop()
        self.overlay.clear_live_preview_text()
        self.overlay.clear_mic_quality_text()
        self.overlay.hide()

    def _on_settings_closed(self) -> None:
        with self._hotkey_lock:
            self._settings_open = False
            self._pressed_keys = set()
            self._ptt_armed = False
        # Small guard to avoid accidental trigger during close animation.
        self._ignore_hotkey_until = time.monotonic() + 0.15
        self._bind_hotkey()
        logging.info("Settings closed: hotkey rebound")

    def _apply_settings_from_ui(self, values: dict) -> tuple[bool, str, dict]:
        old_config = self.config
        old_transcriber = self.transcriber
        committed = False
        try:
            # Apply settings transactionally to avoid half-applied state on mode switch errors.
            next_config = AppConfig(**asdict(self.config))
            next_config.openrouter_api_key = str(values.get("openrouter_api_key", "")).strip()
            next_config.model = str(values.get("model", "")).strip() or "google/gemini-2.5-flash"
            backend = str(values.get("transcription_backend", "api")).strip().lower()
            next_config.transcription_backend = "local" if backend == "local" else "api"
            local_engine = str(values.get("local_backend_engine", "whisper")).strip().lower()
            next_config.local_backend_engine = local_engine if local_engine in {"whisper", "vosk", "sherpa"} else "whisper"
            local_model = str(values.get("local_whisper_model", "small")).strip().lower()
            next_config.local_whisper_model = local_model if local_model in {"small", "medium", "large-v3"} else "small"
            local_vosk_model = str(values.get("local_vosk_model", LocalVoskTranscriber.MODEL_NAME)).strip().lower()
            if local_vosk_model not in LocalVoskTranscriber.MODEL_URLS:
                local_vosk_model = LocalVoskTranscriber.MODEL_NAME
            next_config.local_vosk_model = local_vosk_model
            local_sherpa_model = str(values.get("local_sherpa_model", LocalSherpaWhispeRuTranscriber.MODEL_NAME)).strip().lower()
            next_config.local_sherpa_model = LocalSherpaWhispeRuTranscriber.normalize_model_name(local_sherpa_model)
            raw_ptt = str(values.get("ptt_key", "")).strip().lower() or "f8"
            parsed_ptt = self._parse_hotkey_keys(raw_ptt)
            if not parsed_ptt:
                raise ValueError("Не удалось распознать сочетание клавиш. Нажмите «Захват клавиши» и повторите.")
            next_config.ptt_key = self._format_hotkey(parsed_ptt)
            mode = str(values.get("activation_mode", "hold")).strip().lower()
            next_config.activation_mode = "toggle" if mode == "toggle" else "hold"
            next_config.input_device = str(values.get("input_device", "")).strip()
            next_config.auto_paste = bool(values.get("auto_paste", True))
            next_config.copy_result_to_clipboard = bool(values.get("copy_result_to_clipboard", False))
            next_config.restore_clipboard = bool(values.get("restore_clipboard", True))
            next_config.history_enabled = bool(values.get("history_enabled", True))
            next_config.autoreplace_enabled = bool(values.get("autoreplace_enabled", True))
            next_config.punctuate_text = bool(values.get("punctuate_text", True))
            next_config.capitalize_sentences = bool(values.get("capitalize_sentences", True))
            next_config.terminal_period = bool(values.get("terminal_period", True))
            next_config.convert_numbers = bool(values.get("convert_numbers", False))
            next_config.normalize_quotes_dashes = bool(values.get("normalize_quotes_dashes", True))
            raw_clean_mode = str(values.get("clean_text_mode", "")).strip().lower()
            if raw_clean_mode not in {"off", "soft", "strong"}:
                raw_clean_mode = "soft" if bool(values.get("clean_text", False)) else "off"
            next_config.clean_text_mode = raw_clean_mode
            next_config.clean_text = raw_clean_mode != "off"
            next_config.max_text_quality = bool(values.get("max_text_quality", False))
            next_config.whisper_medium_turbo = bool(values.get("whisper_medium_turbo", False))
            next_config.local_fast_refine = bool(values.get("local_fast_refine", True))
            next_config.show_overlay_widget = bool(values.get("show_overlay_widget", True))
            next_config.show_live_preview_text = bool(values.get("show_live_preview_text", True))
            next_config.show_mic_quality_indicator = bool(values.get("show_mic_quality_indicator", True))
            next_config.play_recording_sounds = bool(values.get("play_recording_sounds", True))
            next_config.play_notification_sounds = bool(values.get("play_notification_sounds", True))
            next_config.autostart_windows = bool(values.get("autostart_windows", False))
            next_config.auto_download_updates = bool(values.get("auto_download_updates", False))
            next_config.update_feed_url = DEFAULT_UPDATE_FEED_URL
            next_config.update_channel = DEFAULT_UPDATE_CHANNEL
            next_config.run_as_admin = bool(values.get("run_as_admin", False))
            next_config.hardware_acceleration = bool(values.get("hardware_acceleration", True))
            next_config.notify_on_no_sound = bool(values.get("notify_on_no_sound", True))
            summary_backend = str(values.get("meeting_summary_backend", "api")).strip().lower()
            next_config.meeting_summary_backend = (
                "local_gguf"
                if summary_backend == "local_gguf"
                else "api"
            )
            next_config.meeting_local_summary_model = LocalMeetingSummaryGGUF.normalize_model_name(
                str(values.get("meeting_local_summary_model", LocalMeetingSummaryGGUF.DEFAULT_MODEL))
            )
            meeting_audio_mode = str(values.get("meeting_system_audio_mode", "auto")).strip().lower()
            next_config.meeting_system_audio_mode = "manual" if meeting_audio_mode == "manual" else "auto"
            next_config.meeting_system_audio_device_id = str(values.get("meeting_system_audio_device_id", "") or "").strip()
            next_config.meeting_ollama_model = str(values.get("meeting_ollama_model", "")).strip() or next_config.meeting_ollama_model
            next_config.meeting_ollama_url = str(values.get("meeting_ollama_url", "")).strip() or next_config.meeting_ollama_url
            try:
                parsed_sr = int(str(values.get("sample_rate", next_config.sample_rate)).strip() or next_config.sample_rate)
            except Exception:
                parsed_sr = next_config.sample_rate
            next_config.sample_rate = min(96000, max(8000, parsed_sr))

            old_config = self.config
            old_transcriber = self.transcriber
            new_transcriber = self._create_transcriber(next_config)

            # Commit config/transcriber only after successful creation.
            self.config = next_config
            self.transcriber = new_transcriber
            committed = True
            self.config_store.save(self.config)
            self.recorder.sample_rate = self.config.sample_rate
            self.recorder.set_input_device(self.config.input_device)
            self._apply_windows_startup(self.config.autostart_windows)
            self._apply_run_as_admin(self.config.run_as_admin)
            supports_preview = False
            try:
                supports_preview = bool(self.transcriber.supports_live_preview())
            except Exception:
                supports_preview = False
            self.overlay.set_live_preview_enabled(bool(self.config.show_live_preview_text and self.config.show_overlay_widget and supports_preview))
            self.overlay.set_mic_quality_enabled(bool(self.config.show_mic_quality_indicator and self.config.show_overlay_widget))
            if not (self.config.show_mic_quality_indicator and self.config.show_overlay_widget):
                self.overlay.clear_mic_quality_text()
            self._bind_hotkey()
            self._notify("Настройки сохранены")
            return True, "Сохранено", self._settings_snapshot()
        except Exception as err:
            logging.exception("Failed to apply settings")
            if committed:
                try:
                    self.config = old_config
                    self.transcriber = old_transcriber
                    self.recorder.sample_rate = self.config.sample_rate
                    self.recorder.set_input_device(self.config.input_device)
                except Exception:
                    logging.exception("Failed to rollback settings after apply error")
            try:
                # On failure keep previously working state and hotkey.
                if self._settings_open:
                    pass
                else:
                    self._bind_hotkey()
            except Exception:
                logging.exception("Failed to restore hotkey after settings error")
            return False, f"Ошибка: {err}", self._settings_snapshot()

    def _on_quit(self, _icon: Icon, _item: MenuItem) -> None:
        self._running = False
        if self._icon:
            self._icon.stop()

    def shutdown(self) -> None:
        try:
            self._stop_meeting_capture_from_ui()
        except Exception:
            pass
        self._stop_live_preview_loop()
        self._stop_mic_quality_loop()
        self._ipc_stop_event.set()
        try:
            if self._ipc_thread and self._ipc_thread.is_alive():
                self._ipc_thread.join(timeout=0.6)
        except Exception:
            logging.exception("Failed to stop IPC listener")
        self._model_dl_stop_event.set()
        try:
            self._unbind_hotkey()
        except Exception:
            logging.exception("Failed to unhook keyboard")
        try:
            if self._mutex_handle:
                ctypes.windll.kernel32.CloseHandle(self._mutex_handle)
                self._mutex_handle = None
        except Exception:
            logging.exception("Failed to release mutex")
        self.recorder.close()
        try:
            self._refine_http.close()
        except Exception:
            logging.exception("Failed to close refine HTTP session")
        try:
            self._ollama_http.close()
        except Exception:
            logging.exception("Failed to close ollama HTTP session")
        self.overlay.shutdown()

    @staticmethod
    def _get_foreground_window() -> int:
        try:
            return int(ctypes.windll.user32.GetForegroundWindow())
        except Exception:
            return 0

    @staticmethod
    def _get_window_text(hwnd: int) -> str:
        try:
            if not hwnd:
                return ""
            user32 = ctypes.windll.user32
            length = int(user32.GetWindowTextLengthW(int(hwnd)))
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(int(hwnd), buf, length + 1)
            return str(buf.value or "").strip()
        except Exception:
            return ""

    def _is_settings_window_foreground(self) -> bool:
        hwnd = self._get_foreground_window()
        title = self._get_window_text(hwnd).lower()
        if not title:
            return False
        return (APP_NAME.lower() in title) and ("настройк" in title)

    @staticmethod
    def _focus_window(hwnd: int) -> None:
        if not hwnd:
            return
        try:
            user32 = ctypes.windll.user32
            if user32.IsWindow(hwnd):
                # SW_RESTORE = 9
                user32.ShowWindow(hwnd, 9)
                user32.SetForegroundWindow(hwnd)
        except Exception:
            logging.exception("Failed to focus target window")


def run() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    app = VoiceProApp(base_dir)
    app.run()


