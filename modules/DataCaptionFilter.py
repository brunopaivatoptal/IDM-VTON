# -*- coding: utf-8 -*-
from typing import Protocol

class Filter(Protocol):
    def load(self, max_elements: int):
        """Load any necessary resources, e.g., models into GPU memory."""
        pass

    def unload(self):
        """Unload any resources to free up space, e.g., clear models from GPU memory."""
        pass

    def apply(self, image_path: str, verbose: bool = False) -> bool:
        """Apply the filter to an image. Return True if the image passes the filter."""
        pass
