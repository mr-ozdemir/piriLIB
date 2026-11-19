"""Reading YAML documents the way their authors mean them.

YAML 1.1, which PyYAML implements, reads an unquoted ``17:00`` as the base-60
number 1020 and ``1:30`` as 90. A time of day written without quotes would
silently become another value, so :func:`load` keeps every scalar with a colon
as the text that was written, as YAML 1.2 does.
"""

from typing import Any, Final

import yaml

_INT: Final = "tag:yaml.org,2002:int"
_FLOAT: Final = "tag:yaml.org,2002:float"


class Loader(yaml.SafeLoader):
    """Safe YAML loader without base-60 numbers."""


def _int(loader: Loader, node: yaml.ScalarNode) -> int | str:
    if ":" in node.value:
        return str(node.value)
    return loader.construct_yaml_int(node)


def _float(loader: Loader, node: yaml.ScalarNode) -> float | str:
    if ":" in node.value:
        return str(node.value)
    return loader.construct_yaml_float(node)


Loader.add_constructor(_INT, _int)
Loader.add_constructor(_FLOAT, _float)


def load(text: str) -> Any:
    """Parse one YAML document; ``17:00`` stays the text ``"17:00"``."""
    return yaml.load(text, Loader=Loader)
