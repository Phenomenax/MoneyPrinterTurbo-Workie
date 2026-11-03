"""Per-run video provider selection that reads environment variables only.

Standard library only: the Agent Skill helpers load this file by path before the
application and its dependencies are importable. Do not import ``app.config``,
network libraries, or application start-up modules here.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from urllib.parse import urlsplit, urlunsplit

VIDEO_RUN_CONFIG_API_VERSION = 1

WAN3 = "wan3"
SEEDANCE = "volcengine_seedance"
SOURCES = (WAN3, SEEDANCE)
SOURCE_LABELS = {WAN3: "Wan3", SEEDANCE: "Seedance兼容网关"}

ENVIRONMENT_NAMES = {
    WAN3: {
        "key": ("MPT_WAN3_API_KEY", "DASHSCOPE_API_KEY"),
        "base_url": "MPT_WAN3_BASE_URL",
        "model": "MPT_WAN3_MODEL_NAME",
        "resolution": "MPT_WAN3_RESOLUTION",
        "concurrency": "MPT_WAN3_CONCURRENCY",
    },
    SEEDANCE: {
        "key": (
            "MPT_SEEDANCE_API_KEY",
            "MPT_VOLCENGINE_ARK_API_KEY",
            "VOLCENGINE_ARK_API_KEY",
        ),
        "base_url": "MPT_SEEDANCE_BASE_URL",
        "model": "MPT_SEEDANCE_MODEL_NAME",
        "resolution": "MPT_SEEDANCE_RESOLUTION",
        "concurrency": "MPT_SEEDANCE_CONCURRENCY",
        "family": "MPT_SEEDANCE_MODEL_FAMILY",
    },
}

REASON_NO_VIDEO_KEY = "no_video_key"
REASON_MULTIPLE_VIDEO_KEYS = "multiple_video_keys"
REASON_MISSING_VIDEO_SETTINGS = "missing_video_settings"
REASON_INVALID_VIDEO_SETTINGS = "invalid_video_settings"

MAX_CONCURRENCY = 5
MAX_MODEL_NAME_LENGTH = 200
SEEDANCE_TASKS_PATH = "/contents/generations/tasks"
SEEDANCE_DEFAULT_API_PREFIX = "/api/v3"
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class ModelFamily:
    name: str
    min_duration: int
    max_duration: int
    resolutions: tuple[str, ...]
    default_resolution: str
    # Native-audio models get an explicit ``generate_audio=false``; older
    # models do not document the field and must not depend on it.
    sends_generate_audio: bool


WAN3_FAMILY = ModelFamily("wan3", 2, 30, ("480P", "720P", "1080P"), "1080P", False)
SEEDANCE_FAMILIES = {
    family.name: family
    for family in (
        ModelFamily("seedance-2.0", 4, 15, ("480p", "720p", "1080p"), "1080p", True),
        ModelFamily("seedance-2.0-fast", 4, 15, ("480p", "720p"), "720p", True),
        ModelFamily("seedance-2.0-mini", 4, 15, ("480p", "720p"), "720p", True),
        ModelFamily("seedance-2.5", 4, 30, ("480p", "720p", "1080p"), "1080p", True),
        ModelFamily("seedance-1.0-pro", 2, 12, ("480p", "720p", "1080p"), "1080p", False),
        ModelFamily("seedance-1.0-pro-fast", 2, 12, ("480p", "720p", "1080p"), "1080p", False),
    )
}
_SEEDANCE_RESOLUTIONS = frozenset({"480p", "720p", "1080p"})


@dataclass(frozen=True)
class VideoRunConfig:
    """One immutable provider configuration shared by every call of a task.

    Never serialize this object with ``dataclasses.asdict``; use ``summary``.
    """

    source: str
    model: str
    model_family: str
    api_key: str = field(repr=False)
    endpoint: str
    resolution: str
    min_duration: int
    max_duration: int
    concurrency: int
    send_generate_audio: bool = False
    generate_audio: bool = False
    poll_interval: float | None = None
    run_timeout: float | None = None
    # ``None`` means the entry point did not capture network settings.
    proxies: tuple[tuple[str, str], ...] | None = field(default=None, repr=False)
    tls_verify: bool | None = None

    @property
    def display_name(self) -> str:
        return SOURCE_LABELS[self.source]

    def proxy_map(self) -> dict[str, str] | None:
        return None if self.proxies is None else dict(self.proxies)

    def credential_fingerprint(self) -> str:
        """Internal record identity; never display it or store the key itself."""
        return hashlib.sha256(
            f"{self.source}\n{self.endpoint}\n{self.api_key}".encode("utf-8")
        ).hexdigest()

    def secrets(self) -> tuple[str, ...]:
        values = [self.api_key, *(value for _name, value in self.proxies or ())]
        return tuple(value for value in values if value)

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "model": self.model,
            "model_family": self.model_family,
            "endpoint": self.endpoint,
            "resolution": self.resolution,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "concurrency": self.concurrency,
        }

    def with_runtime(
        self,
        *,
        proxies: Mapping[str, object] | None,
        tls_verify: bool,
        poll_interval: float,
        run_timeout: float,
    ) -> VideoRunConfig:
        proxy_items = tuple(
            sorted((str(name), str(value)) for name, value in (proxies or {}).items() if value)
        )
        return replace(
            self,
            proxies=proxy_items,
            tls_verify=bool(tls_verify),
            poll_interval=float(poll_interval),
            run_timeout=float(run_timeout),
        )

    def child_environment(self, environ: Mapping[str, str]) -> dict[str, str]:
        """Copy ``environ`` so a child resolves exactly this selection."""
        names = ENVIRONMENT_NAMES[self.source]
        child = dict(environ)
        primary, *aliases = names["key"]
        child[primary] = self.api_key
        for alias in aliases:
            if str(child.get(alias, "") or "").strip():
                child[alias] = self.api_key
        child[names["base_url"]] = self.endpoint
        child[names["model"]] = self.model
        child[names["resolution"]] = self.resolution
        child[names["concurrency"]] = str(self.concurrency)
        if "family" in names:
            child[names["family"]] = self.model_family
        return child


@dataclass(frozen=True)
class VideoInputRequired:
    """Non-sensitive description of what the user must configure or choose."""

    reason: str
    source: str | None = None
    options: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    invalid: tuple[str, ...] = ()

    def lines(self) -> list[str]:
        result = [f"INPUT_REASON={self.reason}"]
        if self.source:
            result.append(f"VIDEO_SOURCE={self.source}")
        for option in self.options:
            result.append(f"VIDEO_SOURCE_OPTION={option}|{SOURCE_LABELS[option]}")
            if self.reason == REASON_NO_VIDEO_KEY:
                result.append(
                    f"VIDEO_SOURCE_ENV={option}|" + ",".join(required_environment(option))
                )
        result.extend(f"MISSING={name}" for name in self.missing)
        result.extend(f"INVALID={name}" for name in self.invalid)
        return result


def required_environment(source: str) -> tuple[str, str, str]:
    names = ENVIRONMENT_NAMES[source]
    return names["key"][0], names["base_url"], names["model"]


def _value(environ: Mapping[str, str], name: str) -> str:
    return str(environ.get(name, "") or "").strip()


def first_api_key(environ: Mapping[str, str], source: str) -> str:
    for name in ENVIRONMENT_NAMES[source]["key"]:
        value = _value(environ, name)
        if value:
            return value
    return ""


def detected_sources(environ: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(source for source in SOURCES if first_api_key(environ, source))


def _http_url_parts(value: str):
    if _CONTROL_CHARACTERS.search(value) or any(char.isspace() for char in value):
        raise ValueError("invalid URL")
    parts = urlsplit(value)
    _ = parts.port  # urlsplit validates the port lazily.
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or "?" in value
        or "#" in value
    ):
        raise ValueError("invalid URL")
    return parts


def normalize_wan3_base_url(value: str) -> str:
    """Return the native DashScope host/prefix without ``/api/v1``."""
    parts = _http_url_parts(str(value or "").strip())
    prefix = parts.path.rstrip("/")
    if prefix.endswith("/compatible-mode/v1"):
        raise ValueError("Wan3 requires the native video API, not /compatible-mode/v1")
    if prefix.endswith("/api/v1"):
        prefix = prefix[: -len("/api/v1")]
    return urlunsplit((parts.scheme, parts.netloc, prefix, "", ""))


def normalize_seedance_tasks_url(value: str) -> str:
    """Accept a host, an API prefix, or the full task URL; append the path once."""
    parts = _http_url_parts(str(value or "").strip())
    path = parts.path.rstrip("/")
    if not path:
        path = SEEDANCE_DEFAULT_API_PREFIX + SEEDANCE_TASKS_PATH
    elif not path.endswith(SEEDANCE_TASKS_PATH):
        path += SEEDANCE_TASKS_PATH
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def detect_seedance_family(model: str) -> str | None:
    """Recognize a family from a model name; ``None`` when not reliable.

    Specific variants are matched before the standard model, and an unknown
    variant word is never treated as the standard model.
    """
    tokens = _tokens(str(model or ""))
    for index, token in enumerate(tokens):
        if token != "seedance":
            continue
        rest = tokens[index + 1:]
        if len(rest) < 2 or not (rest[0].isdigit() and rest[1].isdigit()):
            return None
        version, variants = (rest[0], rest[1]), rest[2:]
        following = variants[0] if variants else None

        def plain(item: str | None) -> bool:
            return item is None or item.isdigit()

        if version == ("1", "0"):
            if following != "pro":
                return None
            after = variants[1] if len(variants) > 1 else None
            if after == "fast":
                return "seedance-1.0-pro-fast"
            return "seedance-1.0-pro" if plain(after) else None
        if version == ("2", "0"):
            if following in {"fast", "mini"}:
                return f"seedance-2.0-{following}"
            return "seedance-2.0" if plain(following) else None
        if version == ("2", "5"):
            return "seedance-2.5" if plain(following) else None
        return None
    return None


def canonical_seedance_family(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if "seedance" not in text.lower():
        text = "seedance-" + text
    return detect_seedance_family(text)


def _valid_text(value: str, maximum: int = MAX_MODEL_NAME_LENGTH) -> bool:
    return bool(value) and len(value) <= maximum and not _CONTROL_CHARACTERS.search(value)


def _concurrency(raw: str) -> int | None:
    if not raw:
        return MAX_CONCURRENCY
    if not re.fullmatch(r"\d+", raw):
        return None
    value = int(raw)
    return value if 1 <= value <= MAX_CONCURRENCY else None


def _resolve_source(environ: Mapping[str, str], source: str) -> VideoRunConfig | VideoInputRequired:
    names = ENVIRONMENT_NAMES[source]
    missing: list[str] = []
    invalid: list[str] = []

    api_key = first_api_key(environ, source)
    if not api_key:
        missing.append(names["key"][0])
    elif _CONTROL_CHARACTERS.search(api_key):
        invalid.append(names["key"][0])

    endpoint = ""
    raw_url = _value(environ, names["base_url"])
    if not raw_url:
        missing.append(names["base_url"])
    else:
        try:
            endpoint = (
                normalize_wan3_base_url(raw_url)
                if source == WAN3
                else normalize_seedance_tasks_url(raw_url)
            )
        except ValueError:
            invalid.append(names["base_url"])

    # Send the platform-assigned name exactly; only outer whitespace is removed.
    model = _value(environ, names["model"])
    if not model:
        missing.append(names["model"])
    elif not _valid_text(model):
        invalid.append(names["model"])

    family = None
    if source == WAN3:
        family = WAN3_FAMILY
    else:
        explicit_raw = _value(environ, names["family"])
        explicit = canonical_seedance_family(explicit_raw) if explicit_raw else None
        detected = detect_seedance_family(model) if model else None
        if explicit_raw and explicit is None:
            invalid.append(names["family"])
        elif explicit and detected and explicit != detected:
            invalid.append(names["family"])
        elif explicit or detected:
            family = SEEDANCE_FAMILIES[explicit or detected]
        elif model:
            missing.append(names["family"])

    raw_resolution = _value(environ, names["resolution"])
    resolution = ""
    if source == WAN3:
        resolution = raw_resolution.upper() or WAN3_FAMILY.default_resolution
        if resolution not in WAN3_FAMILY.resolutions:
            invalid.append(names["resolution"])
    else:
        resolution = raw_resolution.lower()
        if family is not None:
            resolution = resolution or family.default_resolution
            if resolution not in family.resolutions:
                invalid.append(names["resolution"])
        elif resolution and resolution not in _SEEDANCE_RESOLUTIONS:
            invalid.append(names["resolution"])

    concurrency = _concurrency(_value(environ, names["concurrency"]))
    if concurrency is None:
        invalid.append(names["concurrency"])

    if missing or invalid:
        return VideoInputRequired(
            REASON_MISSING_VIDEO_SETTINGS if missing else REASON_INVALID_VIDEO_SETTINGS,
            source=source,
            missing=tuple(missing),
            invalid=tuple(invalid),
        )
    return VideoRunConfig(
        source=source,
        model=model,
        model_family=family.name,
        api_key=api_key,
        endpoint=endpoint,
        resolution=resolution,
        min_duration=family.min_duration,
        max_duration=family.max_duration,
        concurrency=concurrency,
        send_generate_audio=family.sends_generate_audio,
        generate_audio=False,
    )


def resolve_video_run_config(
    environ: Mapping[str, str],
    requested_source: str | None = None,
) -> VideoRunConfig | VideoInputRequired:
    """Select a provider by non-empty keys, then validate only the selection.

    An explicit choice never switches to another provider, and a candidate with
    an incomplete address or model is never silently dropped from the choice.
    """
    if requested_source is not None:
        requested_source = str(requested_source).strip().lower()
        if requested_source not in SOURCES:
            raise ValueError(f"unsupported environment video source: {requested_source!r}")
        return _resolve_source(environ, requested_source)
    candidates = detected_sources(environ)
    if not candidates:
        return VideoInputRequired(REASON_NO_VIDEO_KEY, options=SOURCES)
    if len(candidates) > 1:
        return VideoInputRequired(REASON_MULTIPLE_VIDEO_KEYS, options=candidates)
    return _resolve_source(environ, candidates[0])
