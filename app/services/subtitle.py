import importlib.util
import json
import os
import os.path
from pathlib import Path
import tempfile
import threading
from timeit import default_timer as timer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None
from loguru import logger

from app.config import config
from app.models import const
from app.utils import utils
from app.utils.subtitle_reader import file_to_subtitles as file_to_subtitles
from app.utils.subtitle_writer import write_subtitle_file

model_size = config.whisper.get("model_size", "large-v3-turbo")
device = config.whisper.get("device", "auto")
compute_type = config.whisper.get("compute_type", "auto")
initial_prompt = config.whisper.get("initial_prompt", "") or None
model = None
_loaded_device = None
_model_init_lock = threading.Lock()
_CUDA_ERROR_MARKERS = ("cuda", "cublas", "cudnn", "cudart", "nvrtc")
_CUDA_DLL_DIR_HANDLES = []
_CUDA_DLL_DIRS_REGISTERED: set[str] = set()
_CUDA_WINDOWS_REQUIRED_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
)
_CUDA_LINUX_PRELOAD_LIBS = (
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
)


def _register_cuda_dll_directories() -> list[str]:
    """Expose wheel-bundled NVIDIA CUDA 12 / cuDNN 9 libraries to the process loader."""
    added: list[str] = []
    sub_dirs = ("bin", "lib") if os.name == "nt" else ("lib",)
    candidate_dirs: list[Path] = []
    for package_name in ("nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc"):
        try:
            spec = importlib.util.find_spec(package_name)
        except (ImportError, ValueError, OSError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        for root_location in spec.submodule_search_locations:
            base_path = Path(root_location)
            for sub_dir in sub_dirs:
                candidate = base_path / sub_dir
                if candidate.is_dir():
                    candidate_dirs.append(candidate)

    if os.name == "nt":
        current_path = os.environ.get("PATH", "")
        path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
        normalized_existing = {os.path.normcase(entry) for entry in path_entries}
        prepend_entries: list[str] = []
        for directory in candidate_dirs:
            resolved = str(directory.resolve())
            key = os.path.normcase(resolved)
            if key in _CUDA_DLL_DIRS_REGISTERED:
                continue
            if hasattr(os, "add_dll_directory"):
                try:
                    _CUDA_DLL_DIR_HANDLES.append(os.add_dll_directory(resolved))
                except OSError:
                    pass
            if key not in normalized_existing:
                prepend_entries.append(resolved)
                normalized_existing.add(key)
            _CUDA_DLL_DIRS_REGISTERED.add(key)
            added.append(resolved)
        if prepend_entries:
            os.environ["PATH"] = os.pathsep.join(prepend_entries + path_entries)
    else:
        import ctypes

        rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
        for directory in candidate_dirs:
            resolved = str(directory.resolve())
            if resolved in _CUDA_DLL_DIRS_REGISTERED:
                continue
            for lib_name in _CUDA_LINUX_PRELOAD_LIBS:
                lib_path = directory / lib_name
                if lib_path.is_file():
                    try:
                        ctypes.CDLL(str(lib_path), mode=rtld_global)
                    except OSError:
                        pass
            _CUDA_DLL_DIRS_REGISTERED.add(resolved)
            added.append(resolved)
    return added


def _is_auto(value: str | None) -> bool:
    return not value or str(value).strip().lower() == "auto"


def _is_cuda_runtime_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CUDA_ERROR_MARKERS)


def _has_cuda() -> bool:
    _register_cuda_dll_directories()
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() <= 0:
            logger.debug("whisper CUDA unavailable: no CUDA device detected")
            return False
    except Exception as exc:
        logger.debug(f"whisper CUDA unavailable: ctranslate2 probe failed ({exc})")
        return False
    if os.name == "nt":
        import ctypes

        for dll_name in _CUDA_WINDOWS_REQUIRED_DLLS:
            try:
                ctypes.WinDLL(dll_name)
            except OSError as exc:
                logger.debug(f"whisper CUDA unavailable: failed to load {dll_name} ({exc})")
                return False
    return True


def resolve_whisper_runtime(
    configured_device: str | None = None,
    configured_compute_type: str | None = None,
) -> tuple[str, str]:
    raw_device = device if configured_device is None else configured_device
    raw_compute = compute_type if configured_compute_type is None else configured_compute_type
    if _is_auto(raw_device):
        resolved_device = "cuda" if _has_cuda() else "cpu"
    else:
        resolved_device = str(raw_device).strip().lower()
    if _is_auto(raw_compute):
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"
    else:
        resolved_compute = str(raw_compute).strip().lower()
        if (
            _is_auto(raw_device)
            and resolved_device == "cpu"
            and resolved_compute in {"float16", "int8_float16"}
        ):
            logger.warning(
                f"whisper compute_type '{resolved_compute}' is unsupported on CPU; "
                "falling back to 'int8'"
            )
            resolved_compute = "int8"
    return resolved_device, resolved_compute


def find_local_model_path(
    target_size: str | None = None, *, check_hf_cache: bool = True
) -> str | None:
    size = str(model_size if target_size is None else target_size).strip()
    if not size:
        return None
    candidates = [Path(utils.root_dir()) / "models" / f"whisper-{size}"]
    direct = Path(size).expanduser()
    if direct.is_dir():
        candidates.append(direct)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model.bin").is_file():
            return str(candidate)
    if check_hf_cache and WhisperModel is not None:
        try:
            from faster_whisper.utils import download_model

            cached = download_model(size, local_files_only=True)
            if cached and (Path(cached) / "model.bin").is_file():
                return str(cached)
        except Exception:
            pass
    return None


def _ensure_model_loaded(*, local_files_only: bool = False, force_cpu: bool = False) -> bool:
    """Load the large Whisper model once, even when jobs start concurrently."""
    global model, _loaded_device
    if model is not None and (not force_cpu or _loaded_device == "cpu"):
        return True

    with _model_init_lock:
        if model is not None and (not force_cpu or _loaded_device == "cpu"):
            return True
        if WhisperModel is None:
            return False
        local_path = find_local_model_path(model_size, check_hf_cache=local_files_only)
        if local_files_only and not local_path:
            logger.error(
                f"local whisper model '{model_size}' not found in "
                f"{Path(utils.root_dir()) / 'models' / f'whisper-{model_size}'}"
            )
            return False
        model_path = local_path or model_size
        if force_cpu:
            resolved_device, resolved_compute = "cpu", "int8"
        else:
            resolved_device, resolved_compute = resolve_whisper_runtime()
        if resolved_device == "cuda":
            _register_cuda_dll_directories()

        logger.info(
            f"loading model: {model_path}, device: {resolved_device}, compute_type: {resolved_compute}"
        )
        try:
            model = WhisperModel(
                model_size_or_path=model_path,
                device=resolved_device,
                compute_type=resolved_compute,
            )
            _loaded_device = resolved_device
        except Exception as e:
            if (
                not force_cpu
                and resolved_device == "cuda"
                and _is_auto(device)
                and _is_cuda_runtime_error(e)
            ):
                logger.warning(
                    f"CUDA whisper initialization failed ({e}); falling back to cpu/int8"
                )
                try:
                    model = WhisperModel(
                        model_size_or_path=model_path,
                        device="cpu",
                        compute_type="int8",
                    )
                    _loaded_device = "cpu"
                    return True
                except Exception as fallback_exc:
                    e = fallback_exc
            logger.error(
                f"failed to load model: {e} \n\n"
                f"********************************************\n"
                f"this may be caused by network issue. \n"
                f"please download the model manually and put it in the 'models' folder. \n"
                f"see [README.md FAQ](https://github.com/harry0703/MoneyPrinterTurbo) for more details.\n"
                f"********************************************\n\n"
            )
            return False
    return True


def transcribe_audio(
    audio_file: str,
    *,
    initial_prompt: str | None = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = True,
    vad_parameters: dict | None = None,
    local_files_only: bool = False,
):
    if WhisperModel is None:
        raise RuntimeError("faster_whisper is not installed")
    if not _ensure_model_loaded(local_files_only=local_files_only):
        raise RuntimeError(f"failed to load local whisper model '{model_size}'")
    effective_prompt = (
        globals()["initial_prompt"] if initial_prompt is None else (initial_prompt or None)
    )
    kwargs = {
        "beam_size": beam_size,
        "word_timestamps": word_timestamps,
        "vad_filter": vad_filter,
        "vad_parameters": vad_parameters or dict(min_silence_duration_ms=500),
    }
    if effective_prompt:
        kwargs["initial_prompt"] = effective_prompt
    try:
        active_model = model
        segments, info = active_model.transcribe(audio_file, **kwargs)
        return list(segments), info
    except Exception as exc:
        if _loaded_device == "cuda" and _is_auto(device) and _is_cuda_runtime_error(exc):
            logger.warning(
                f"CUDA whisper transcription failed ({exc}); retrying on cpu/int8"
            )
            if not _ensure_model_loaded(local_files_only=local_files_only, force_cpu=True):
                raise RuntimeError(
                    f"failed to reload whisper model '{model_size}' on cpu"
                ) from exc
            active_model = model
            segments, info = active_model.transcribe(audio_file, **kwargs)
            return list(segments), info
        raise


def create(
    audio_file,
    subtitle_file: str = "",
    word_level: bool = False,
    log_details: bool = True,
    initial_prompt: str | None = None,
):
    if WhisperModel is None:
        logger.warning("faster_whisper not available, skipping whisper subtitle generation")
        return ""
    if not _ensure_model_loaded():
        return None

    if log_details:
        logger.info(f"start, output file: {subtitle_file}")
    if not subtitle_file:
        subtitle_file = f"{audio_file}.srt"

    segments, info = transcribe_audio(
        audio_file,
        initial_prompt=initial_prompt,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    logger.info(
        f"detected language: '{info.language}', probability: {info.language_probability:.2f}"
    )

    start = timer()
    subtitles = []

    def recognized(seg_text, seg_start, seg_end):
        seg_text = seg_text.strip()
        if not seg_text:
            return

        if log_details:
            msg = "[%.2fs -> %.2fs] %s" % (seg_start, seg_end, seg_text)
            logger.debug(msg)

        subtitles.append(
            {"msg": seg_text, "start_time": seg_start, "end_time": seg_end}
        )

    for segment in segments:
        if not segment.words:
            # Faster Whisper exposes words as Optional[List[Word]]. A segment
            # can still have usable text and timestamps when alignment yields
            # no words; keep that subtitle in both sentence and word modes.
            recognized(segment.text, segment.start, segment.end)
            continue

        if word_level:
            for word in segment.words:
                cleaned_word = word.word.strip()
                if cleaned_word:
                    recognized(cleaned_word, word.start, word.end)
            continue

        words_idx = 0
        words_len = len(segment.words)

        seg_start = 0
        seg_end = 0
        seg_text = ""

        if segment.words:
            is_segmented = False
            for word in segment.words:
                if not is_segmented:
                    seg_start = word.start
                    is_segmented = True

                seg_end = word.end
                # Accumulate words; only trailing punctuation ends a sentence.
                seg_text += word.word

                if word.word.rstrip().endswith(tuple(const.PUNCTUATIONS)):
                    # Punctuation inside a word (3.14, 1,000, 12:30) is not a
                    # sentence boundary. Remove only actual trailing delimiters.
                    seg_text = seg_text.rstrip().rstrip("".join(const.PUNCTUATIONS))
                    recognized(seg_text, seg_start, seg_end)

                    is_segmented = False
                    seg_text = ""

                if words_idx == 0 and segment.start < word.start:
                    seg_start = word.start
                if words_idx == (words_len - 1) and segment.end > word.end:
                    seg_end = word.end
                words_idx += 1

        if not seg_text:
            continue

        recognized(seg_text, seg_start, seg_end)

    end = timer()

    diff = end - start
    logger.info(f"complete, elapsed: {diff:.2f} s")

    idx = 1
    lines = []
    for subtitle in subtitles:
        text = subtitle.get("msg")
        if text:
            lines.append(
                utils.text_to_srt(
                    idx, text, subtitle.get("start_time"), subtitle.get("end_time")
                )
            )
            idx += 1

    sub = "\n".join(lines) + "\n"
    if not lines:
        logger.warning("transcription produced no subtitle cues")
        return
    write_subtitle_file(subtitle_file, sub)
    if log_details:
        logger.info(f"subtitle file created: {subtitle_file}")


def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    """Transcribe in-memory WAV audio and remove every temporary artifact."""
    if not isinstance(audio_bytes, bytes) or not audio_bytes:
        return ""

    with tempfile.TemporaryDirectory(prefix="whisper-transcript-") as temp_dir:
        audio_file = os.path.join(temp_dir, "reference.wav")
        subtitle_file = os.path.join(temp_dir, "reference.srt")
        with open(audio_file, "wb") as output:
            output.write(audio_bytes)

        create_result = create(audio_file, subtitle_file, log_details=False)
        if create_result in (None, "") and not os.path.isfile(subtitle_file):
            return ""
        subtitle_items = file_to_subtitles(subtitle_file)
        return " ".join(item[2].strip() for item in subtitle_items if item[2].strip())


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity(a, b):
    distance = levenshtein_distance(a.lower(), b.lower())
    max_length = max(len(a), len(b))
    return 1 - (distance / max_length)


def correct(subtitle_file, video_script):
    subtitle_items = file_to_subtitles(subtitle_file)
    normalized_script = utils.normalize_script_for_subtitle_matching(video_script)
    # create() splits Whisper segments at punctuation but not at the space that ends
    # a Thai or Lao sentence, so a segment can hold several of those sentences.
    script_lines = utils.split_string_by_punctuations(
        normalized_script, split_thai_and_lao_sentences=False
    )

    corrected = False
    new_subtitle_items = []
    script_index = 0
    subtitle_index = 0

    while script_index < len(script_lines) and subtitle_index < len(subtitle_items):
        script_line = script_lines[script_index].strip()
        subtitle_line = subtitle_items[subtitle_index][2].strip()

        if script_line == subtitle_line:
            new_subtitle_items.append(subtitle_items[subtitle_index])
            script_index += 1
            subtitle_index += 1
        else:
            combined_subtitle = subtitle_line
            start_time = subtitle_items[subtitle_index][1].split(" --> ")[0]
            end_time = subtitle_items[subtitle_index][1].split(" --> ")[1]
            next_subtitle_index = subtitle_index + 1

            while next_subtitle_index < len(subtitle_items):
                next_subtitle = subtitle_items[next_subtitle_index][2].strip()
                if similarity(
                    script_line, combined_subtitle + " " + next_subtitle
                ) > similarity(script_line, combined_subtitle):
                    combined_subtitle += " " + next_subtitle
                    end_time = subtitle_items[next_subtitle_index][1].split(" --> ")[1]
                    next_subtitle_index += 1
                else:
                    break

            if similarity(script_line, combined_subtitle) > 0.8:
                logger.warning(
                    f"Merged/Corrected - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True
            else:
                logger.warning(
                    f"Mismatch - Script: {script_line}, Subtitle: {combined_subtitle}"
                )
                new_subtitle_items.append(
                    (
                        len(new_subtitle_items) + 1,
                        f"{start_time} --> {end_time}",
                        script_line,
                    )
                )
                corrected = True

            script_index += 1
            subtitle_index = next_subtitle_index

    # Process the remaining lines of the script.
    while script_index < len(script_lines):
        logger.warning(f"Extra script line: {script_lines[script_index]}")
        if subtitle_index < len(subtitle_items):
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    subtitle_items[subtitle_index][1],
                    script_lines[script_index],
                )
            )
            subtitle_index += 1
        else:
            new_subtitle_items.append(
                (
                    len(new_subtitle_items) + 1,
                    "00:00:00,000 --> 00:00:00,000",
                    script_lines[script_index],
                )
            )
        script_index += 1
        corrected = True

    if subtitle_index < len(subtitle_items):
        logger.warning(
            f"Dropping {len(subtitle_items) - subtitle_index} transcription cue(s) "
            "after the script ends"
        )
        corrected = True

    if corrected:
        content = "".join(
            f"{i + 1}\n{item[1]}\n{item[2]}\n\n"
            for i, item in enumerate(new_subtitle_items)
        )
        write_subtitle_file(subtitle_file, content)
        logger.info("Subtitle corrected")
    else:
        logger.success("Subtitle is correct")


if __name__ == "__main__":
    task_id = "c12fd1e6-4b0a-4d65-a075-c87abe35a072"
    task_dir = utils.task_dir(task_id)
    subtitle_file = f"{task_dir}/subtitle.srt"
    audio_file = f"{task_dir}/audio.mp3"

    subtitles = file_to_subtitles(subtitle_file)
    print(subtitles)

    script_file = f"{task_dir}/script.json"
    with open(script_file, "r") as f:
        script_content = f.read()
    s = json.loads(script_content)
    script = s.get("script")

    correct(subtitle_file, script)

    subtitle_file = f"{task_dir}/subtitle-test.srt"
    create(audio_file, subtitle_file)
