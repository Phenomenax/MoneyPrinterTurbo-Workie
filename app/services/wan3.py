"""Bailian Wan3 asynchronous video generation, without paid-submit retries."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils.request_logging import trace_http_call


DEFAULT_MODEL_NAME = "wan3.0-video"
MIN_CLIP_SECONDS = 2
MAX_CLIP_SECONDS = 30
DEFAULT_POLL_INTERVAL_SECONDS = 15.0
DEFAULT_RUN_TIMEOUT_SECONDS = 1200.0
MAX_POLL_RETRIES = 5
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class Wan3Error(RuntimeError):
    """A Wan3 configuration, rejected request, or completed-task failure."""

    def __init__(self, message: str, task_id: str = ""):
        super().__init__(message)
        self.task_id = task_id


class Wan3UnconfirmedTaskError(Wan3Error):
    """A paid job may exist remotely and must not be submitted again."""


class Wan3DownloadError(Wan3Error):
    """A paid output could not be downloaded or validated locally."""


def get_api_key(settings: Mapping[str, Any] | None = None) -> str:
    settings = config.app if settings is None else settings
    return (
        os.getenv("MPT_WAN3_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or str(settings.get("wan3_api_key", "") or "").strip()
    )


def _setting(name: str, default: str = "", settings: Mapping[str, Any] | None = None) -> str:
    settings = config.app if settings is None else settings
    return (
        os.getenv(f"MPT_{name.upper()}", "").strip()
        or str(settings.get(name, "") or "").strip()
        or default
    )


def _base_url(settings: Mapping[str, Any] | None = None) -> str:
    value = _setting("wan3_base_url", settings=settings).rstrip("/")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in {"http", "https"}
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise Wan3Error(
            "Wan3 requires wan3_base_url (or MPT_WAN3_BASE_URL) as the "
            "absolute HTTP(S) base URL for your Bailian workspace and region"
        )
    prefix = parts.path.rstrip("/")
    if prefix.endswith("/compatible-mode/v1"):
        raise Wan3Error(
            "wan3_base_url must use the native DashScope video API, not "
            "/compatible-mode/v1; use the same regional host or its /api/v1 base"
        )
    if prefix.endswith("/api/v1"):
        prefix = prefix[:-len("/api/v1")]
    return value[:len(value) - len(parts.path)] + prefix if parts.path else value


def validate_config(settings: Mapping[str, Any] | None = None) -> None:
    if not get_api_key(settings):
        raise Wan3Error(
            "Wan3 requires MPT_WAN3_API_KEY, DASHSCOPE_API_KEY, or wan3_api_key"
        )
    _base_url(settings)
    _resolution(settings)


def _resolution(settings: Mapping[str, Any] | None = None) -> str:
    value = _setting("wan3_resolution", "1080P", settings).upper()
    if value not in {"480P", "720P", "1080P"}:
        raise Wan3Error("wan3_resolution must be 480P, 720P, or 1080P")
    return value


def _tls_verify() -> bool:
    value = config.app.get("tls_verify", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(config.app.get(name, default))
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(value, minimum), maximum) if math.isfinite(value) else default


def _redact(value: Any, api_key: str) -> str:
    text = str(value or "")
    for secret in [api_key, *config.proxy.values()]:
        if secret:
            text = text.replace(str(secret), "***")
            text = text.replace(quote_plus(str(secret)), "***")
    return text[:500]


def _status_code(response: Any) -> int:
    try:
        return int(response.status_code)
    except (TypeError, ValueError, AttributeError):
        return 0


def _response_error(body: Any, api_key: str) -> str:
    if not isinstance(body, dict):
        return "unreadable provider response"
    output = body.get("output")
    output = output if isinstance(output, dict) else {}
    return _redact(
        ": ".join(str(value) for value in (
            output.get("code") or body.get("code"),
            output.get("message") or body.get("message"),
        ) if value), api_key,
    ) or "provider did not return error details"


def generate_videos(
    search_term: str,
    minimum_duration: int = 5,
    video_aspect: VideoAspect = VideoAspect.portrait,
    *,
    on_task_created: Callable[[str], None] | None = None,
) -> list[MaterialInfo]:
    """Submit one silent clip, save its remote ID, then poll the same task."""
    validate_config()
    term = str(search_term or "").strip()
    if not term:
        raise Wan3Error("Wan3 visual prompt must not be empty")
    try:
        aspect = VideoAspect(video_aspect)
    except (TypeError, ValueError) as exc:
        raise Wan3Error("Wan3 video aspect is invalid") from exc
    if (
        isinstance(minimum_duration, bool)
        or not isinstance(minimum_duration, int)
        or not MIN_CLIP_SECONDS <= minimum_duration <= MAX_CLIP_SECONDS
    ):
        raise Wan3Error("Wan3 clip duration must be an integer from 2 to 30 seconds")

    api_key = get_api_key()
    base_url = _base_url()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-DashScope-Async": "enable",
    }
    payload = {
        "model": _setting("wan3_model_name", DEFAULT_MODEL_NAME),
        "input": {"prompt": term},
        "parameters": {
            "resolution": _resolution(),
            "ratio": aspect.value,
            "duration": minimum_duration,
            "audio": False,
        },
    }
    try:
        response = trace_http_call(
            "POST", requests.post,
            f"{base_url}/api/v1/services/aigc/video-generation/video-synthesis",
            headers=headers, json=payload, proxies=config.proxy,
            verify=_tls_verify(), timeout=(30, 60), allow_redirects=False,
        )
    except Exception as exc:
        raise Wan3UnconfirmedTaskError(
            "Wan3 submission returned no response; a paid task may exist remotely: "
            f"error={type(exc).__name__}, detail={_redact(exc, api_key)}"
        ) from exc

    status = _status_code(response)
    if 300 <= status < 400 or status >= 500 or status == 0:
        raise Wan3UnconfirmedTaskError(
            f"Wan3 submission returned HTTP {status}; a paid task may exist remotely"
        )
    try:
        body = response.json()
    except Exception as exc:
        error_type = Wan3UnconfirmedTaskError if 200 <= status < 300 else Wan3Error
        raise error_type(f"Wan3 submission returned unreadable JSON: HTTP {status}") from exc
    if not 200 <= status < 300:
        raise Wan3Error(f"Wan3 submission rejected: HTTP {status}, {_response_error(body, api_key)}")
    output = body.get("output") if isinstance(body, dict) else None
    task_id = str(output.get("task_id") or "").strip() if isinstance(output, dict) else ""
    if not task_id:
        raise Wan3UnconfirmedTaskError(
            "Wan3 accepted the submission without returning a task ID"
        )
    logger.info(f"Wan3 video task created: id={task_id}")
    if on_task_created is not None:
        try:
            on_task_created(task_id)
        except Exception as exc:
            raise Wan3UnconfirmedTaskError(
                f"Wan3 task was submitted but its recovery record could not be saved: id={task_id}",
                task_id=task_id,
            ) from exc

    task = _wait_for_task(task_id=task_id, base_url=base_url, headers=headers, api_key=api_key)
    video_url = task["output"].get("video_url")
    try:
        parsed = urlsplit(video_url) if isinstance(video_url, str) else None
        valid_url = parsed and parsed.scheme in {"http", "https"} and parsed.hostname and parsed.username is None and parsed.password is None
    except ValueError:
        valid_url = False
    if not valid_url:
        raise Wan3Error(f"Wan3 completed without a downloadable video: id={task_id}", task_id=task_id)
    usage = task.get("usage")
    duration = usage.get("output_video_duration") if isinstance(usage, dict) else None
    try:
        measured_duration = float(duration)
        if not math.isfinite(measured_duration) or measured_duration <= 0:
            raise ValueError("invalid duration")
    except (TypeError, ValueError, OverflowError):
        measured_duration = float(minimum_duration)
    return [MaterialInfo(
        provider="wan3", url=video_url, duration=math.ceil(measured_duration),
        source_info={
            "provider": "wan3", "search_term": term, "asset_id": task_id,
            "rendition": {"id": task_id},
        },
    )]


def _wait_for_task(*, task_id: str, base_url: str, headers: dict[str, str], api_key: str) -> dict[str, Any]:
    deadline = time.monotonic() + _bounded_float("wan3_run_timeout", DEFAULT_RUN_TIMEOUT_SECONDS, 1.0, 7200.0)
    interval = _bounded_float("wan3_poll_interval", DEFAULT_POLL_INTERVAL_SECONDS, 0.5, 60.0)
    task_url = f"{base_url}/api/v1/tasks/{quote(task_id, safe='')}"
    failures = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Wan3UnconfirmedTaskError(f"Wan3 local wait timed out; remote task may still be running: id={task_id}", task_id=task_id)
        phase_timeout = max(min(remaining / 2.0, 30.0), 0.001)
        try:
            response = trace_http_call(
                "GET", requests.get, task_url, headers=headers,
                proxies=config.proxy, verify=_tls_verify(),
                timeout=(phase_timeout, phase_timeout), allow_redirects=False,
            )
            status_code = _status_code(response)
            if status_code in RETRYABLE_STATUS_CODES:
                raise requests.exceptions.HTTPError(f"HTTP {status_code}", response=response)
            if not 200 <= status_code < 300:
                raise Wan3UnconfirmedTaskError(f"Wan3 task status is unknown: id={task_id}, HTTP {status_code}", task_id=task_id)
            body = response.json()
            output = body.get("output") if isinstance(body, dict) else None
            if not isinstance(output, dict):
                raise Wan3UnconfirmedTaskError(f"Wan3 task status response is malformed: id={task_id}", task_id=task_id)
        except Wan3UnconfirmedTaskError:
            raise
        except Exception as exc:
            retryable = isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError))
            response = getattr(exc, "response", None)
            retryable = retryable or (response is not None and _status_code(response) in RETRYABLE_STATUS_CODES)
            failures += 1
            remaining = deadline - time.monotonic()
            if not retryable or failures > MAX_POLL_RETRIES or remaining <= 0:
                raise Wan3UnconfirmedTaskError(
                    f"Wan3 polling failed; remote task may still be running: id={task_id}, error={type(exc).__name__}, detail={_redact(exc, api_key)}",
                    task_id=task_id,
                ) from exc
            time.sleep(min(float(failures), remaining))
            continue
        failures = 0
        status = str(output.get("task_status") or "").strip().upper()
        if status == "SUCCEEDED":
            return body
        if status in {"FAILED", "CANCELED", "CANCELLED"}:
            raise Wan3Error(f"Wan3 task failed: id={task_id}, status={status}, {_response_error(body, api_key)}", task_id=task_id)
        if status not in {"PENDING", "RUNNING"}:
            raise Wan3UnconfirmedTaskError(f"Wan3 returned an unknown task status: id={task_id}, status={status!r}", task_id=task_id)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Wan3UnconfirmedTaskError(f"Wan3 local wait timed out; remote task may still be running: id={task_id}", task_id=task_id)
        time.sleep(min(interval, remaining))
