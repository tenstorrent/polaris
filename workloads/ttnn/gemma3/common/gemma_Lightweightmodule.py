# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any


class LightweightModule:
    """Torch modules add a surprising amount of host overhead for attribute
    access and method calls. This class is a lightweight alternative that
    just wraps a forward function for now."""
    
    def __init__(self) -> None:
        pass
    
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Override this method in subclasses."""
        raise NotImplementedError("Subclasses must implement forward()")
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)