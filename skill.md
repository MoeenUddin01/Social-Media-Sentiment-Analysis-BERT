# Reusable Patterns and Code Templates

## Module Template

```python
"""Module description.

Brief explanation of what this module provides.
"""

from __future__ import annotations


class ExampleClass:
    """Class description.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Raises:
        ValueError: When invalid input is provided.
    """

    def __init__(self, param1: str, param2: int) -> None:
        """Initialize the class."""
        pass

    def example_method(self, arg: str) -> str:
        """Method description.

        Args:
            arg: Description of argument.

        Returns:
            Description of return value.
        """
        pass
```

## Configuration Pattern

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration dataclass."""

    param: str = "default"

    def __post_init__(self) -> None:
        """Validate configuration."""
        pass
```

## Factory Pattern

```python
class Factory:
    """Factory for creating instances."""

    def create(self, type_: str) -> Any:
        """Create instance by type."""
        creators = {
            "type_a": TypeA,
            "type_b": TypeB,
        }
        return creators[type_]()
```

## Common Imports

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
```
