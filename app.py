from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "generated_audio"
DEFAULT_VOICE = "de_DE-thorsten-medium"
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
PROJECT_VENV_PYTHON = BASE_DIR / ".venv" / "bin" / "python"
VERSE_REFERENCE = "Jesaja 43,1-4 (personalisiert)"
PIPER_LENGTH_SCALE = float(os.environ.get("PIPER_LENGTH_SCALE", "1.0"))
PIPER_SENTENCE_SILENCE = float(os.environ.get("PIPER_SENTENCE_SILENCE", "0.2"))
PIPER_MODEL_PATH = Path(
    os.environ.get(
        "PIPER_MODEL_PATH",
        str(BASE_DIR / "models" / f"{DEFAULT_VOICE}.onnx"),
    )
)
PIPER_CONFIG_PATH = Path(f"{PIPER_MODEL_PATH}.json")
_PIPER_VOICE = None
_PIPER_SYN_CONFIG = None
_PIPER_LOAD_LOCK = threading.Lock()
_PIPER_SYNTHESIS_LOCK = threading.RLock()
_AUDIO_JOB_LOCK = threading.Lock()
_PENDING_AUDIO_JOBS: set[str] = set()
_AUDIO_JOB_ERRORS: dict[str, str] = {}


def personalize_verse(first_name: str) -> str:
    clean_name = " ".join(first_name.split())
    if not clean_name:
        return ""

    return (
        f"{clean_name}, ich bin der Herr, dein Gott.\n"
        "Ich habe dich bei deinem Namen gerufen.\n"
        f"{clean_name}, du bist mein.\n"
        "Fürchte dich nicht, denn ich habe dich erlöst.\n"
        f"{clean_name}, du bist wertvoll in meinen Augen und ich liebe dich."
    )


def find_piper_runner() -> tuple[list[str] | None, str]:
    if importlib.util.find_spec("piper") is not None:
        return ([sys.executable, "-m", "piper"], Path(sys.executable).name)

    if PROJECT_VENV_PYTHON.exists():
        result = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON),
                "-c",
                "import importlib.util; print(importlib.util.find_spec('piper') is not None)",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            return ([str(PROJECT_VENV_PYTHON), "-m", "piper"], str(PROJECT_VENV_PYTHON))

    return (None, "")


def current_process_has_piper() -> bool:
    return importlib.util.find_spec("piper") is not None


def load_piper_voice():
    global _PIPER_VOICE, _PIPER_SYN_CONFIG

    if not current_process_has_piper():
        return None, None

    with _PIPER_LOAD_LOCK:
        if (_PIPER_VOICE is None) or (_PIPER_SYN_CONFIG is None):
            from piper import PiperVoice, SynthesisConfig

            _PIPER_VOICE = PiperVoice.load(
                PIPER_MODEL_PATH,
                config_path=PIPER_CONFIG_PATH,
            )
            _PIPER_SYN_CONFIG = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)

    return _PIPER_VOICE, _PIPER_SYN_CONFIG


def warm_piper_voice() -> None:
    try:
        load_piper_voice()
    except Exception:
        # If preloading fails, the request path will show the actual error later.
        return


def get_piper_status() -> tuple[bool, str]:
    runner, runner_label = find_piper_runner()
    if runner is None:
        return (
            False,
            "Piper ist noch nicht gefunden worden. Installiere es am besten in .venv.",
        )

    if not PIPER_MODEL_PATH.exists():
        return (
            False,
            f"Das Piper-Stimmenmodell fehlt noch: {PIPER_MODEL_PATH}",
        )

    if not PIPER_CONFIG_PATH.exists():
        return (
            False,
            f"Die Piper-Konfigurationsdatei fehlt noch: {PIPER_CONFIG_PATH}",
        )

    return (
        True,
        (
            "Piper ist aktiv mit dem Modell "
            f"{PIPER_MODEL_PATH.name} ueber "
            f"{'eingebettete Piper-API' if current_process_has_piper() else runner_label}. "
            f"Tempo={PIPER_LENGTH_SCALE}, Satzpause={PIPER_SENTENCE_SILENCE}s."
        ),
    )


def cleanup_audio_cache(max_files: int = 12) -> None:
    AUDIO_DIR.mkdir(exist_ok=True)
    wav_files = sorted(AUDIO_DIR.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    for stale_file in wav_files[max_files:]:
        stale_file.unlink(missing_ok=True)


def build_audio_name(text: str) -> str:
    cache_key = "|".join(
        [
            PIPER_MODEL_PATH.name,
            str(PIPER_LENGTH_SCALE),
            str(PIPER_SENTENCE_SILENCE),
            text,
        ]
    )
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.wav"


def get_audio_path(audio_name: str) -> Path:
    return AUDIO_DIR / audio_name


def is_audio_ready(audio_name: str) -> bool:
    audio_path = get_audio_path(audio_name)
    return audio_path.exists() and audio_path.stat().st_size > 0


def mark_audio_error(audio_name: str, message: str) -> None:
    with _AUDIO_JOB_LOCK:
        _AUDIO_JOB_ERRORS[audio_name] = message


def clear_audio_error(audio_name: str) -> None:
    with _AUDIO_JOB_LOCK:
        _AUDIO_JOB_ERRORS.pop(audio_name, None)


def get_audio_job_state(audio_name: str) -> tuple[str, str]:
    if is_audio_ready(audio_name):
        return ("ready", "")

    with _AUDIO_JOB_LOCK:
        if audio_name in _PENDING_AUDIO_JOBS:
            return ("pending", "")

        error_message = _AUDIO_JOB_ERRORS.get(audio_name, "")

    if error_message:
        return ("error", error_message)

    return ("pending", "")


def synthesize_with_embedded_piper(text: str, output_path: Path) -> None:
    voice, syn_config = load_piper_voice()
    if (voice is None) or (syn_config is None):
        raise RuntimeError("Die eingebettete Piper-API ist nicht verfuegbar.")

    silence_bytes = bytes(int(voice.config.sample_rate * PIPER_SENTENCE_SILENCE * 2))

    with _PIPER_SYNTHESIS_LOCK:
        wav_file = wave.open(str(output_path), "wb")
        with wav_file:
            wav_params_set = False
            for i, audio_chunk in enumerate(voice.synthesize(text, syn_config=syn_config)):
                if not wav_params_set:
                    wav_file.setframerate(audio_chunk.sample_rate)
                    wav_file.setsampwidth(audio_chunk.sample_width)
                    wav_file.setnchannels(audio_chunk.sample_channels)
                    wav_params_set = True

                if i > 0 and silence_bytes:
                    wav_file.writeframes(silence_bytes)

                wav_file.writeframes(audio_chunk.audio_int16_bytes)


def synthesize_with_piper(text: str) -> str:
    runner, _runner_label = find_piper_runner()
    if runner is None:
        raise RuntimeError("Piper ist nicht installiert oder nicht auffindbar.")

    AUDIO_DIR.mkdir(exist_ok=True)
    output_name = build_audio_name(text)
    output_path = get_audio_path(output_name)
    temp_output_path = output_path.with_name(f"{output_path.stem}.partial.wav")

    with _PIPER_SYNTHESIS_LOCK:
        if output_path.exists() and output_path.stat().st_size > 0:
            output_path.touch()
            cleanup_audio_cache()
            return output_name

        cleanup_audio_cache()
        temp_output_path.unlink(missing_ok=True)

        try:
            if current_process_has_piper():
                synthesize_with_embedded_piper(text, temp_output_path)
            else:
                result = subprocess.run(
                    [
                        *runner,
                        "--model",
                        str(PIPER_MODEL_PATH),
                        "--output_file",
                        str(temp_output_path),
                        "--length-scale",
                        str(PIPER_LENGTH_SCALE),
                        "--sentence-silence",
                        str(PIPER_SENTENCE_SILENCE),
                    ],
                    input=text,
                    text=True,
                    capture_output=True,
                    check=False,
                )

                if result.returncode != 0:
                    details = (result.stderr or result.stdout or "Unbekannter Piper-Fehler").strip()
                    raise RuntimeError(details)

            if (not temp_output_path.exists()) or temp_output_path.stat().st_size == 0:
                raise RuntimeError("Piper hat keine gueltige Audiodatei erzeugt.")

            temp_output_path.replace(output_path)
        except Exception as error:
            temp_output_path.unlink(missing_ok=True)
            if isinstance(error, RuntimeError):
                raise
            raise RuntimeError(str(error)) from error

    clear_audio_error(output_name)
    return output_name


def build_setup_hint() -> str:
    return """python3 -m venv .venv
source .venv/bin/activate
python -m pip install piper-tts
python -m piper.download_voices de_DE-thorsten-medium --data-dir models
python app.py"""


def build_page(
    first_name: str = "",
    personalized_text: str = "",
    audio_name: str = "",
    audio_pending: bool = False,
    piper_ready: bool = False,
    piper_status: str = "",
    synthesis_error: str = "",
) -> str:
    safe_name = html.escape(first_name)
    safe_personalized_text = html.escape(personalized_text)
    safe_piper_status = html.escape(piper_status)
    safe_synthesis_error = html.escape(synthesis_error)
    safe_setup_hint = html.escape(build_setup_hint())

    status_class = "ok" if piper_ready else "warn"
    audio_block = ""
    audio_status_block = ""
    if audio_name and not audio_pending:
        safe_audio_name = html.escape(audio_name)
        audio_block = f"""
        <div class="audio-card">
          <p class="audio-label">Piper-Audio</p>
          <audio id="piper-player" controls preload="auto" src="/audio/{safe_audio_name}"></audio>
          <div class="actions">
            <button type="button" onclick="playAudio()">Noch einmal abspielen</button>
            <button type="button" class="secondary" onclick="stopAudio()">Wiedergabe stoppen</button>
          </div>
        </div>
        """
    elif audio_name and audio_pending:
        audio_status_block = """
        <div id="audio-status" class="status warn audio-status">
          <strong>Audio wird vorbereitet</strong>
          <p>Dein Text ist schon da. Die Stimme wird jetzt im Hintergrund erzeugt.</p>
        </div>
        """

    error_block = ""
    if synthesis_error:
        error_block = f"""
        <div class="status error">
          <strong>Piper konnte das Audio nicht erzeugen.</strong>
          <p>{safe_synthesis_error}</p>
        </div>
        """

    setup_block = ""
    if not piper_ready:
        setup_block = f"""
        <div class="setup-box">
          <p class="setup-title">Piper schnell einrichten</p>
          <pre>{safe_setup_hint}</pre>
        </div>
        """

    result_block = ""
    if personalized_text:
        result_block = f"""
        <section class="result" aria-live="polite">
          <span class="reference">{VERSE_REFERENCE}</span>
          <p id="personalized-verse" class="verse">{safe_personalized_text}</p>
          {audio_status_block}
          <div id="audio-container">{audio_block}</div>
        </section>
        """

    return f"""<!doctype html>
<html lang="de">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Personalisierter Bibelvers mit Piper</title>
    <style>
      :root {{
        --bg-top: #f7efe2;
        --bg-bottom: #e8dfc8;
        --card: rgba(255, 252, 246, 0.94);
        --text: #2f2419;
        --muted: #675849;
        --accent: #7b4b2a;
        --accent-soft: #d9b382;
        --border: rgba(123, 75, 42, 0.16);
        --ok-bg: rgba(75, 122, 54, 0.12);
        --ok-text: #355225;
        --warn-bg: rgba(156, 102, 65, 0.12);
        --warn-text: #76482f;
        --error-bg: rgba(139, 36, 36, 0.1);
        --error-text: #7f1d1d;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
        background:
          radial-gradient(circle at top, rgba(255, 255, 255, 0.75), transparent 35%),
          linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
        color: var(--text);
        font-family: Georgia, "Times New Roman", serif;
      }}

      .card {{
        width: min(100%, 820px);
        padding: 32px;
        border-radius: 24px;
        background: var(--card);
        border: 1px solid var(--border);
        box-shadow: 0 24px 70px rgba(68, 44, 17, 0.12);
      }}

      h1 {{
        margin: 0 0 12px;
        font-size: clamp(2rem, 5vw, 3.4rem);
        line-height: 1;
      }}

      p {{
        line-height: 1.6;
        margin: 0 0 16px;
      }}

      .intro {{
        color: var(--muted);
      }}

      form {{
        display: grid;
        gap: 14px;
        margin: 28px 0 24px;
      }}

      label {{
        font-weight: 600;
      }}

      input {{
        width: 100%;
        padding: 14px 16px;
        border-radius: 14px;
        border: 1px solid rgba(47, 36, 25, 0.16);
        font-size: 1rem;
        font-family: inherit;
        background: rgba(255, 255, 255, 0.96);
      }}

      button {{
        width: fit-content;
        border: 0;
        border-radius: 999px;
        padding: 14px 22px;
        background: linear-gradient(135deg, var(--accent), #9c6641);
        color: white;
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
      }}

      button:hover {{
        filter: brightness(1.05);
      }}

      .secondary {{
        background: transparent;
        color: var(--accent);
        border: 1px solid rgba(123, 75, 42, 0.22);
      }}

      .status {{
        padding: 16px 18px;
        border-radius: 16px;
        margin-bottom: 20px;
      }}

      .status.ok {{
        background: var(--ok-bg);
        color: var(--ok-text);
      }}

      .status.warn {{
        background: var(--warn-bg);
        color: var(--warn-text);
      }}

      .status.error {{
        background: var(--error-bg);
        color: var(--error-text);
      }}

      .status strong {{
        display: block;
        margin-bottom: 6px;
      }}

      .result {{
        margin-top: 18px;
        padding: 22px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(217, 179, 130, 0.18), rgba(255, 255, 255, 0.86));
        border: 1px solid rgba(123, 75, 42, 0.18);
      }}

      .reference {{
        display: inline-block;
        margin-bottom: 10px;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(123, 75, 42, 0.12);
        color: var(--accent);
        font-size: 0.9rem;
        font-weight: 700;
      }}

      .verse {{
        margin: 0;
        font-size: clamp(1.2rem, 2.5vw, 1.7rem);
        line-height: 1.55;
        white-space: pre-line;
      }}

      .audio-card {{
        margin-top: 22px;
        padding-top: 18px;
        border-top: 1px solid rgba(123, 75, 42, 0.16);
      }}

      .audio-status {{
        margin-top: 22px;
        margin-bottom: 0;
      }}

      .audio-label {{
        margin-bottom: 10px;
        color: var(--muted);
        font-size: 0.95rem;
      }}

      audio {{
        width: 100%;
      }}

      .actions {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 18px;
      }}

      .setup-box {{
        margin-top: 18px;
        padding: 18px;
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(123, 75, 42, 0.16);
      }}

      .setup-title {{
        margin-bottom: 12px;
        font-weight: 700;
      }}

      pre {{
        margin: 0;
        padding: 14px;
        overflow-x: auto;
        border-radius: 14px;
        background: #2a211b;
        color: #f5ead6;
        font-size: 0.95rem;
      }}

      @media (max-width: 640px) {{
        .card {{
          padding: 24px;
        }}

        button {{
          width: 100%;
        }}
      }}
    </style>
  </head>
  <body>
    <main class="card">
      <h1>Ein Name. Ein Zuspruch.</h1>
      <p class="intro">
        Gib einen Vornamen ein. Danach wird dein personalisierter Text angezeigt
        und mit Piper als Audiodatei erzeugt.
      </p>

      <div class="status {status_class}">
        <strong>Piper-Status</strong>
        <p>{safe_piper_status}</p>
      </div>

      {error_block}
      {setup_block}

      <form method="post">
        <div>
          <label for="first_name">Vorname</label>
          <input
            id="first_name"
            name="first_name"
            type="text"
            placeholder="Zum Beispiel Anna"
            value="{safe_name}"
            autocomplete="given-name"
            required
          >
        </div>
        <button type="submit">Text personalisieren</button>
      </form>

      {result_block}
    </main>

    <script>
      const initialAudioName = {json.dumps(audio_name)};
      const initialAudioReady = {str(bool(audio_name and not audio_pending)).lower()};
      const shouldAutoplay = initialAudioReady;
      const shouldPollForAudio = {str(bool(audio_name and audio_pending)).lower()};
      const audioStatusUrl = initialAudioName ? `/audio-status/${{encodeURIComponent(initialAudioName)}}` : "";

      function getPlayer() {{
        return document.getElementById("piper-player");
      }}

      function getAudioContainer() {{
        return document.getElementById("audio-container");
      }}

      function getAudioStatus() {{
        return document.getElementById("audio-status");
      }}

      function buildAudioCard(audioName) {{
        return `
          <div class="audio-card">
            <p class="audio-label">Piper-Audio</p>
            <audio id="piper-player" controls preload="auto" src="/audio/${{encodeURIComponent(audioName)}}"></audio>
            <div class="actions">
              <button type="button" onclick="playAudio()">Noch einmal abspielen</button>
              <button type="button" class="secondary" onclick="stopAudio()">Wiedergabe stoppen</button>
            </div>
          </div>
        `;
      }}

      function setAudioStatus(kind, title, message) {{
        const status = getAudioStatus();
        if (!status) {{
          return;
        }}

        status.className = `status audio-status ${{kind}}`;
        status.innerHTML = `<strong>${{title}}</strong><p>${{message}}</p>`;
      }}

      function mountAudio(audioName) {{
        const container = getAudioContainer();
        if (!container) {{
          return;
        }}

        container.innerHTML = buildAudioCard(audioName);
        const status = getAudioStatus();
        if (status) {{
          status.remove();
        }}

        setTimeout(playAudio, 250);
      }}

      function playAudio() {{
        const player = getPlayer();
        if (!player) {{
          return;
        }}

        player.currentTime = 0;
        player.play().catch(() => {{
          return;
        }});
      }}

      function stopAudio() {{
        const player = getPlayer();
        if (!player) {{
          return;
        }}

        player.pause();
        player.currentTime = 0;
      }}

      async function pollAudioStatus() {{
        if (!audioStatusUrl) {{
          return;
        }}

        try {{
          const response = await fetch(audioStatusUrl, {{ cache: "no-store" }});
          if (!response.ok) {{
            throw new Error("Status konnte nicht geladen werden.");
          }}

          const data = await response.json();
          if (data.status === "ready") {{
            mountAudio(data.audio_name || initialAudioName);
            return;
          }}

          if (data.status === "error") {{
            setAudioStatus(
              "error",
              "Audio konnte nicht erzeugt werden",
              data.message || "Bitte versuche es gleich noch einmal."
            );
            return;
          }}
        }} catch (_error) {{
          // Der naechste Poll versucht es einfach noch einmal.
        }}

        window.setTimeout(pollAudioStatus, 1200);
      }}

      window.addEventListener("load", () => {{
        if (shouldAutoplay) {{
          setTimeout(playAudio, 250);
        }}

        if (shouldPollForAudio) {{
          window.setTimeout(pollAudioStatus, 400);
        }}
      }});
    </script>
  </body>
</html>
"""


def run_audio_job(text: str, audio_name: str) -> None:
    try:
        synthesize_with_piper(text)
    except RuntimeError as error:
        mark_audio_error(audio_name, str(error))
    finally:
        with _AUDIO_JOB_LOCK:
            _PENDING_AUDIO_JOBS.discard(audio_name)


def queue_audio_synthesis(text: str) -> tuple[str, bool]:
    audio_name = build_audio_name(text)
    audio_path = get_audio_path(audio_name)

    if audio_path.exists() and audio_path.stat().st_size > 0:
        audio_path.touch()
        cleanup_audio_cache()
        clear_audio_error(audio_name)
        return (audio_name, True)

    with _AUDIO_JOB_LOCK:
        _AUDIO_JOB_ERRORS.pop(audio_name, None)
        if audio_name in _PENDING_AUDIO_JOBS:
            return (audio_name, False)

        _PENDING_AUDIO_JOBS.add(audio_name)

    threading.Thread(target=run_audio_job, args=(text, audio_name), daemon=True).start()
    return (audio_name, False)


class VerseRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.respond_home()
            return

        if parsed.path == "/healthz":
            self.respond(body="ok", status_code=200, content_type="text/plain; charset=utf-8")
            return

        if parsed.path.startswith("/audio/"):
            self.respond_audio(parsed.path.removeprefix("/audio/"))
            return

        if parsed.path.startswith("/audio-status/"):
            self.respond_audio_status(parsed.path.removeprefix("/audio-status/"))
            return

        self.respond_not_found()

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            body = build_page(
                piper_ready=get_piper_status()[0],
                piper_status=get_piper_status()[1],
            )
            self.respond(body="", status_code=200, content_type="text/html; charset=utf-8", content_length=len(body.encode("utf-8")))
            return

        if parsed.path == "/healthz":
            self.respond(body="", status_code=200, content_type="text/plain; charset=utf-8", content_length=2)
            return

        if parsed.path.startswith("/audio/"):
            self.respond_audio(parsed.path.removeprefix("/audio/"), head_only=True)
            return

        if parsed.path.startswith("/audio-status/"):
            self.respond_audio_status(parsed.path.removeprefix("/audio-status/"), head_only=True)
            return

        self.respond_not_found(head_only=True)

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        form_data = parse_qs(raw_body)
        first_name = form_data.get("first_name", [""])[0].strip()
        personalized_text = personalize_verse(first_name)
        piper_ready, piper_status = get_piper_status()
        audio_name = ""
        audio_pending = False
        synthesis_error = ""

        if personalized_text and piper_ready:
            try:
                audio_name, audio_ready = queue_audio_synthesis(personalized_text)
                audio_pending = not audio_ready
            except RuntimeError as error:
                synthesis_error = str(error)

        self.respond(
            build_page(
                first_name=first_name,
                personalized_text=personalized_text,
                audio_name=audio_name,
                audio_pending=audio_pending,
                piper_ready=piper_ready,
                piper_status=piper_status,
                synthesis_error=synthesis_error,
            )
        )

    def respond_home(self) -> None:
        piper_ready, piper_status = get_piper_status()
        self.respond(build_page(piper_ready=piper_ready, piper_status=piper_status))

    def respond_audio(self, audio_name: str, head_only: bool = False) -> None:
        if "/" in audio_name or "\\" in audio_name or not audio_name.endswith(".wav"):
            self.respond_not_found(head_only=head_only)
            return

        audio_path = get_audio_path(audio_name)
        if not audio_path.exists():
            self.respond_not_found(head_only=head_only)
            return

        content = audio_path.read_bytes()
        self.respond(
            body=b"" if head_only else content,
            status_code=200,
            content_type="audio/wav",
            content_length=len(content),
        )

    def respond_audio_status(self, audio_name: str, head_only: bool = False) -> None:
        if "/" in audio_name or "\\" in audio_name or not audio_name.endswith(".wav"):
            self.respond_not_found(head_only=head_only)
            return

        status, message = get_audio_job_state(audio_name)
        payload = {
            "audio_name": audio_name,
            "status": status,
        }
        if message:
            payload["message"] = message

        body = json.dumps(payload, ensure_ascii=False)
        self.respond(
            body="" if head_only else body,
            status_code=200,
            content_type="application/json; charset=utf-8",
            content_length=len(body.encode("utf-8")),
        )

    def respond_not_found(self, head_only: bool = False) -> None:
        content = b"" if head_only else b"Nicht gefunden"
        self.respond(
            body=content,
            status_code=404,
            content_type="text/plain; charset=utf-8",
            content_length=len(content) if not head_only else 0,
        )

    def respond(
        self,
        body: str | bytes,
        status_code: int = 200,
        content_type: str = "text/html; charset=utf-8",
        content_length: int | None = None,
    ) -> None:
        encoded = body.encode("utf-8") if isinstance(body, str) else body
        final_length = len(encoded) if content_length is None else content_length
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(final_length))
        self.end_headers()
        if encoded:
            self.wfile.write(encoded)

    def log_message(self, format: str, *args) -> None:
        return


def run() -> None:
    AUDIO_DIR.mkdir(exist_ok=True)
    if current_process_has_piper():
        threading.Thread(target=warm_piper_voice, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), VerseRequestHandler)
    print(f"Web-App gestartet: http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
