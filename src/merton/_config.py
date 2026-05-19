"""Runtime configuration for merton.

Configuration sources are layered (last wins):

1. Hard-coded defaults below.
2. ``~/.merton/config.toml`` (or platform-equivalent).
3. ``./merton.toml`` in the current working directory.
4. Environment variables prefixed with ``MERTON_``.
5. Explicit kwargs passed to :func:`configure`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from platformdirs import user_cache_dir, user_config_dir
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

BackendName = Literal["auto", "numpy", "numba", "cupy", "jax", "mlx"]


class MertonSettings(BaseSettings):
    """Process-wide settings.

    Field names map to environment variables as ``MERTON_<UPPER>``.
    Example: ``MERTON_BACKEND=numba``, ``MERTON_LOG_LEVEL=INFO``.

    Layered resolution (last wins):

    1. Hard-coded defaults below.
    2. ``~/.config/merton/config.toml`` (platform-dependent).
    3. ``./merton.toml`` in the current working directory.
    4. Environment variables prefixed with ``MERTON_``.
    5. Explicit kwargs passed to :func:`configure`.
    """

    model_config = SettingsConfigDict(
        env_prefix="MERTON_",
        env_nested_delimiter="__",
        toml_file=[
            str(Path(user_config_dir("merton")) / "config.toml"),
            "merton.toml",
        ],
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(  # type: ignore[override]
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            TomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
        )

    backend: BackendName = "auto"
    numba_threshold: int = 256
    cache_dir: Path = Path(user_cache_dir("merton"))
    log_level: str = "WARNING"

    default_horizon: float = 1.0
    default_calibration_method: str = "vassalou_xing"
    default_default_point: str = "kmv"
    default_lgd: float = 0.6
    default_max_iter: int = 200
    default_tol: float = 1e-8

    risk_free_source: Literal["fred", "ecb", "manual"] = "fred"
    risk_free_series: str = "DGS1"
    fred_api_key: str | None = None

    excel_server_host: str = "127.0.0.1"
    excel_server_port: int = 8000

    progress_bars: bool = True


_settings: MertonSettings | None = None


def get_config() -> MertonSettings:
    """Return the current process-wide :class:`MertonSettings`."""
    global _settings
    if _settings is None:
        _settings = MertonSettings()
    return _settings


def configure(**overrides: object) -> MertonSettings:
    """Update settings in place. Unknown keys raise ``TypeError``."""
    current = get_config()
    new = current.model_copy(update=overrides)
    globals()["_settings"] = new
    return new


def reset_config() -> MertonSettings:
    """Clear in-process overrides and re-read from disk + env."""
    globals()["_settings"] = None
    return get_config()


__all__ = ["MertonSettings", "configure", "get_config", "reset_config"]
