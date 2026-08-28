# setup_models.py

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

MODELS_DIR = PROJECT_ROOT / "models"

VOSK_DIR = MODELS_DIR / "vosk_models"
WHISPER_DIR = MODELS_DIR / "whisper_models"
VAD_DIR = MODELS_DIR / "vad_models"
PIPER_DIR = MODELS_DIR / "piper_models"
LLM_DIR = MODELS_DIR / "gguf_models"


# ==========================================================
# MODELS
# ==========================================================

VOSK_MODEL_NAME = "vosk-model-small-en-us-0.15"

VOSK_URL = (
    "https://alphacephei.com/vosk/models/"
    "vosk-model-small-en-us-0.15.zip"
)


WHISPER_MODEL_NAME = "faster-whisper-small"

WHISPER_HF_REPO = (
    "Systran/faster-whisper-small"
)


PIPER_VOICE = "en_US-lessac-medium"

PIPER_VOICE_DIR = (
    PIPER_DIR / "lessac"
)


LLM_FILENAME = (
    "Qwen3.5-2B-Q4_K_M.gguf"
)

LLM_URL = (
    "https://huggingface.co/bartowski/"
    "Qwen_Qwen3.5-2B-GGUF/resolve/main/"
    "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
)

# ==========================================================
# KOKORO
# ==========================================================

KOKORO_REPO = "hexgrad/Kokoro-82M"

KOKORO_DIR = (
    MODELS_DIR / "kokoro"
)

KOKORO_MODEL_FILE = (
    "kokoro-v1_0.pth"
)

KOKORO_CONFIG_FILE = (
    "config.json"
)

# Voice used by Buddy.
KOKORO_VOICE = "af_heart"

KOKORO_VOICE_DIR = (
    KOKORO_DIR / "voices"
)

KOKORO_VOICE_FILE = (
    f"{KOKORO_VOICE}.pt"
)

# ONNX models for kokoro-onnx (much faster CPU inference)
KOKORO_ONNX_MODEL = "kokoro-v1.0.onnx"
KOKORO_ONNX_VOICES = "voices-v1.0.bin"
# ==========================================================
# HELPERS
# ==========================================================

def print_header(text: str):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def print_status(
    state: str,
    message: str,
):
    print(
        f"[{state}] {message}"
    )


def ensure_directories():

    directories = [
        MODELS_DIR,
        VOSK_DIR,
        WHISPER_DIR,
        VAD_DIR,
        PIPER_DIR,
        PIPER_VOICE_DIR,
        LLM_DIR,
        KOKORO_DIR,
        KOKORO_VOICE_DIR,
    ]

    for directory in directories:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def download_file(
    url: str,
    destination: Path,
    chunk_size: int = 1024 * 1024,
):

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    print(
        f"Downloading:\n"
        f"  {url}\n"
        f"-> {destination}"
    )

    try:

        with urllib.request.urlopen(
            url,
            timeout=60,
        ) as response:

            total = response.headers.get(
                "Content-Length"
            )

            if total is not None:
                total = int(total)

            downloaded = 0

            with open(
                temp_path,
                "wb",
            ) as file:

                while True:

                    chunk = response.read(
                        chunk_size
                    )

                    if not chunk:
                        break

                    file.write(chunk)

                    downloaded += len(chunk)

                    if total:

                        percent = (
                            downloaded
                            / total
                            * 100
                        )

                        print(
                            f"\r  {percent:6.2f}%",
                            end="",
                            flush=True,
                        )

        print()

        temp_path.replace(
            destination
        )

    except (
        HTTPError,
        URLError,
        TimeoutError,
    ) as exc:

        if temp_path.exists():
            temp_path.unlink()

        raise RuntimeError(
            f"Download failed: {url}\n"
            f"{exc}"
        ) from exc


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as file:

        while True:

            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def command_exists(
    command: str,
) -> bool:

    return (
        shutil.which(command)
        is not None
    )


def require_package(
    package_name: str,
    import_name: str | None = None,
):

    if import_name is None:
        import_name = package_name

    try:

        __import__(
            import_name
        )

        return True

    except ImportError:

        print_status(
            "MISS",
            f"{package_name} is not installed.",
        )

        print(
            f"Install it with:\n"
            f"  {sys.executable} -m pip install "
            f"{package_name}"
        )

        return False


# ==========================================================
# VOSK
# ==========================================================

def install_vosk_model():

    print_header(
        "VOSK"
    )

    model_path = (
        VOSK_DIR
        / VOSK_MODEL_NAME
    )

    marker = (
        model_path
        / "am"
    )

    if model_path.is_dir():

        print_status(
            "OK",
            f"Vosk model already exists: "
            f"{model_path}"
        )

        return

    archive = (
        VOSK_DIR
        / f"{VOSK_MODEL_NAME}.zip"
    )

    download_file(
        VOSK_URL,
        archive,
    )

    print(
        "Extracting Vosk model..."
    )

    with zipfile.ZipFile(
        archive,
        "r",
    ) as zip_file:

        zip_file.extractall(
            VOSK_DIR
        )

    archive.unlink()

    if not model_path.is_dir():

        raise RuntimeError(
            "Vosk extraction completed but "
            f"model was not found at {model_path}"
        )

    print_status(
        "OK",
        f"Installed Vosk: {model_path}"
    )


# ==========================================================
# FASTER-WHISPER
# ==========================================================

def install_whisper_model():

    print_header(
        "FASTER-WHISPER"
    )

    target = (
        WHISPER_DIR
        / WHISPER_MODEL_NAME
    )

    required_files = [
        "config.json",
        "tokenizer.json",
        "vocabulary.txt",
    ]

    if all(
        (target / filename).is_file()
        for filename in required_files
    ):

        print_status(
            "OK",
            f"Whisper model already exists: "
            f"{target}"
        )

        return

    if not require_package(
        "huggingface_hub",
        "huggingface_hub",
    ):

        raise RuntimeError(
            "huggingface_hub is required to "
            "download the Faster-Whisper model."
        )

    from huggingface_hub import snapshot_download

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Downloading {WHISPER_HF_REPO}"
    )

    snapshot_download(
        repo_id=WHISPER_HF_REPO,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )

    missing = [
        filename
        for filename in required_files
        if not (
            target / filename
        ).is_file()
    ]

    if missing:

        raise RuntimeError(
            "Whisper model download completed "
            "but required files are missing:\n"
            + "\n".join(missing)
        )

    print_status(
        "OK",
        f"Installed Whisper model: {target}"
    )







# ==========================================================
# KOKORO
# ==========================================================

def install_kokoro():

    print_header(
        "KOKORO 82M"
    )

    KOKORO_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    KOKORO_VOICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        KOKORO_DIR
        / KOKORO_MODEL_FILE
    )

    config_path = (
        KOKORO_DIR
        / KOKORO_CONFIG_FILE
    )

    voice_path = (
        KOKORO_VOICE_DIR
        / KOKORO_VOICE_FILE
    )

    if not require_package(
        "huggingface_hub",
        "huggingface_hub",
    ):

        raise RuntimeError(
            "huggingface_hub is required "
            "for Kokoro model installation."
        )

    from huggingface_hub import hf_hub_download

    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    if model_path.is_file():

        print_status(
            "OK",
            f"Kokoro model already exists: "
            f"{model_path}"
        )

    else:

        print(
            "Downloading Kokoro model..."
        )

        hf_hub_download(
            repo_id=KOKORO_REPO,
            filename=KOKORO_MODEL_FILE,
            local_dir=str(KOKORO_DIR),
            local_dir_use_symlinks=False,
        )

    # ------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------

    if config_path.is_file():

        print_status(
            "OK",
            f"Kokoro config already exists: "
            f"{config_path}"
        )

    else:

        print(
            "Downloading Kokoro config..."
        )

        hf_hub_download(
            repo_id=KOKORO_REPO,
            filename=KOKORO_CONFIG_FILE,
            local_dir=str(KOKORO_DIR),
            local_dir_use_symlinks=False,
        )

    # ------------------------------------------------------
    # VOICE
    # ------------------------------------------------------

    if voice_path.is_file():

        print_status(
            "OK",
            f"Kokoro voice already exists: "
            f"{voice_path}"
        )

    else:

        print(
            f"Downloading Kokoro voice: "
            f"{KOKORO_VOICE}"
        )

        hf_hub_download(
            repo_id=KOKORO_REPO,
            filename=(
                f"voices/{KOKORO_VOICE_FILE}"
            ),
            local_dir=str(KOKORO_DIR),
            local_dir_use_symlinks=False,
        )

    # ------------------------------------------------------
    # VERIFY
    # ------------------------------------------------------

    missing = []

    if not model_path.is_file():
        missing.append(
            str(model_path)
        )

    if not config_path.is_file():
        missing.append(
            str(config_path)
        )

    if not voice_path.is_file():
        missing.append(
            str(voice_path)
        )

    if missing:

        raise RuntimeError(
            "Kokoro installation is incomplete:\n"
            + "\n".join(missing)
        )

    print_status(
        "OK",
        "Kokoro 82M installed successfully."
    )

# ==========================================================
# PIPER
# ==========================================================

def install_piper_voice():

    print_header(
        "PIPER"
    )

    PIPER_VOICE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        PIPER_VOICE_DIR
        / f"{PIPER_VOICE}.onnx"
    )

    config_path = (
        PIPER_VOICE_DIR
        / f"{PIPER_VOICE}.onnx.json"
    )

    if (
        model_path.is_file()
        and config_path.is_file()
    ):

        print_status(
            "OK",
            f"Piper voice already exists: "
            f"{PIPER_VOICE_DIR}"
        )

        return

    if not require_package(
        "piper-tts",
        "piper",
    ):

        raise RuntimeError(
            "piper-tts is required to download "
            "the Piper voice."
        )

    print(
        f"Downloading Piper voice: "
        f"{PIPER_VOICE}"
    )

    command = [
        sys.executable,
        "-m",
        "piper.download_voices",
        PIPER_VOICE,
        "--data-dir",
        str(PIPER_DIR),
    ]

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Piper voice download failed."
        )

    if not model_path.is_file():

        raise RuntimeError(
            f"Piper model was not found: "
            f"{model_path}"
        )

    if not config_path.is_file():

        raise RuntimeError(
            f"Piper config was not found: "
            f"{config_path}"
        )

    print_status(
        "OK",
        f"Installed Piper voice: "
        f"{PIPER_VOICE}"
    )


# ==========================================================
# GGUF LLM
# ==========================================================

def install_llm():

    print_header(
        "GGUF LLM"
    )

    target = (
        LLM_DIR
        / LLM_FILENAME
    )

    if target.is_file():

        print_status(
            "OK",
            f"LLM already exists: {target}"
        )

        return

    download_file(
        LLM_URL,
        target,
    )

    if not target.is_file():

        raise RuntimeError(
            "LLM download failed."
        )

    print_status(
        "OK",
        f"Installed LLM: {target}"
    )


# ==========================================================
# SUMMARY
# ==========================================================

def print_summary():

    print_header(
        "MODEL INSTALLATION COMPLETE"
    )

    print(
        f"Project:\n"
        f"  {PROJECT_ROOT}\n"
    )

    print(
        "Models:"
    )

    print(
        f"  Vosk:\n"
        f"    {VOSK_DIR / VOSK_MODEL_NAME}\n"
    )

    print(
        f"  Faster-Whisper:\n"
        f"    {WHISPER_DIR / WHISPER_MODEL_NAME}\n"
    )

    print(
        f"  Piper:\n"
        f"    {PIPER_VOICE_DIR / (PIPER_VOICE + '.onnx')}\n"
    )

    print(
        f"  Kokoro:\n"
        f"    {KOKORO_DIR / KOKORO_MODEL_FILE}\n"
    )

    print(
        f"  LLM:\n"
        f"    {LLM_DIR / LLM_FILENAME}\n"
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print_header(
        "BUDDY MODEL SETUP"
    )

    print(
        "Project root:"
    )

    print(
        f"  {PROJECT_ROOT}"
    )

    ensure_directories()

    try:

        install_vosk_model()

        install_whisper_model()

        install_piper_voice()

        install_kokoro()

        install_llm()

    except Exception as exc:

        print()
        print_status(
            "ERROR",
            str(exc),
        )

        sys.exit(1)

    print_summary()


# ==========================================================
# PATH HELPER — single source of truth
# ==========================================================

def get_model_paths() -> dict:
    """Return all resolved model paths as a dict.

    Every other module should import this instead of
    hardcoding paths.
    """

    return {
        "vosk": str(
            VOSK_DIR / VOSK_MODEL_NAME
        ),
        "whisper": str(
            WHISPER_DIR / WHISPER_MODEL_NAME
        ),
        "piper_model": str(
            PIPER_VOICE_DIR
            / f"{PIPER_VOICE}.onnx"
        ),
        "piper_config": str(
            PIPER_VOICE_DIR
            / f"{PIPER_VOICE}.onnx.json"
        ),
        "kokoro_model": str(
            KOKORO_DIR / KOKORO_MODEL_FILE
        ),
        "kokoro_config": str(
            KOKORO_DIR / KOKORO_CONFIG_FILE
        ),
        "kokoro_voice": str(
            KOKORO_VOICE_DIR / KOKORO_VOICE_FILE
        ),
        "kokoro_onnx_model": str(
            KOKORO_DIR / KOKORO_ONNX_MODEL
        ),
        "kokoro_onnx_voices": str(
            KOKORO_DIR / KOKORO_ONNX_VOICES
        ),
        "llm": str(
            LLM_DIR / LLM_FILENAME
        ),
    }


if __name__ == "__main__":
    main()