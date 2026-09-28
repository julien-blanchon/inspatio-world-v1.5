"""Hub plumbing shared by every weight-bearing module: config round trips and repo paths.

`HubModule` is `huggingface_hub.PyTorchModelHubMixin` with one addition: the constructor's
`config` argument is a frozen dataclass, and this class turns the `config.json` dictionary back
into that dataclass before construction. The stock mixin decodes through the constructor's
annotation, which is a string under `from __future__ import annotations`, so it would hand the
constructor a plain dict. Every model component (DiT, VAE, text encoder, TAEHV decoder, depth
estimator) therefore gets `save_pretrained`, `from_pretrained` and `push_to_hub` with a
`config.json` next to `model.safetensors`, one folder per component of the weights repository.
Modules are built without their random initialisation (`no_init`): every parameter is loaded
right after, and initialising the 5.7B-parameter text encoder on the CPU otherwise costs longer
than loading it. `component_dir` and `repo_file` locate one folder or file of the weights
repository, which may be a local staging directory instead of a Hub repo id.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, cast, get_origin

from huggingface_hub import PyTorchModelHubMixin, snapshot_download
from torch import nn

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


INITIALISED_LAYERS = (
    nn.Linear,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose2d,
    nn.LayerNorm,
    nn.Embedding,
)


def _skip_reset(_layer: nn.Module) -> None:
    return None


class no_init:
    """Context: freshly built layers skip their random initialisation (weights get loaded)."""

    def __enter__(self) -> None:
        self.saved: list[tuple[type[nn.Module], Callable[..., None]]] = []
        for layer in INITIALISED_LAYERS:
            self.saved.append((layer, layer.reset_parameters))
            setattr(layer, "reset_parameters", _skip_reset)  # noqa: B010

    def __exit__(self, *exception: object) -> None:
        for layer, reset in self.saved:
            setattr(layer, "reset_parameters", reset)  # noqa: B010


def config_from_json[ConfigT: DataclassInstance](
    config_class: type[ConfigT], data: Mapping[str, object]
) -> ConfigT:
    """Rebuild a frozen dataclass from its JSON form, restoring tuples the JSON turned into lists."""

    kwargs: dict[str, object] = {}
    for field in fields(config_class):
        if field.name not in data:
            continue
        value = data[field.name]
        origin = get_origin(field.type) if not isinstance(field.type, str) else None
        if isinstance(value, list) and (origin is tuple or str(field.type).startswith("tuple")):
            value = tuple(cast(list[object], value))
        kwargs[field.name] = value
    return config_class(**kwargs)


class HubModule(PyTorchModelHubMixin):
    """`PyTorchModelHubMixin` whose `config` constructor argument is a frozen dataclass."""

    config_class: ClassVar[type]

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | Path | None,
        force_download: bool,
        local_files_only: bool,
        token: str | bool | None,
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs: object,
    ) -> Self:
        config = model_kwargs.get("config")
        if isinstance(config, dict):
            config_class = cast("type[DataclassInstance]", cls.config_class)
            model_kwargs["config"] = config_from_json(
                config_class, cast("Mapping[str, object]", config)
            )
        elif config is not None and not is_dataclass(config):
            raise TypeError(f"config must be a {cls.config_class.__name__} or a JSON object")
        with no_init():
            return super()._from_pretrained(
                model_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                map_location=map_location,
                strict=strict,
                **model_kwargs,
            )


def component_dir(repo_id: str, revision: str | None, name: str) -> Path:
    """One component folder of the weights repository: a local staging folder, or the Hub cache."""

    if Path(repo_id).is_dir():
        return Path(repo_id) / name
    snapshot = snapshot_download(repo_id, revision=revision, allow_patterns=[f"{name}/*"])
    return Path(snapshot) / name


def repo_file(repo_id: str, revision: str | None, name: str) -> Path:
    """One top-level file of the weights repository, local or from the Hub cache."""

    if Path(repo_id).is_dir():
        return Path(repo_id) / name
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=[name])) / name
