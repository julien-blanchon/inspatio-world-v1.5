"""`inspatio-world` command: `prepare` a scene folder, `generate` a video."""

from __future__ import annotations

from typing import Annotated

import tyro

from inspatio_world.scripts.generate import GenerateConfig
from inspatio_world.scripts.generate import main as generate
from inspatio_world.scripts.prepare import PrepareConfig
from inspatio_world.scripts.prepare import main as prepare

Command = (
    Annotated[PrepareConfig, tyro.conf.subcommand("prepare")]
    | Annotated[GenerateConfig, tyro.conf.subcommand("generate")]
)


def main() -> None:
    config = tyro.cli(Command)
    if isinstance(config, PrepareConfig):
        prepare(config)
    else:
        generate(config)
