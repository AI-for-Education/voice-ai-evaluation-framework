"""Inference-family adapters implemented with Hugging Face Transformers.

Concrete adapters are intentionally not imported here. That keeps profile
validation and CLI help usable in environments where the inference dependencies are not
installed; :mod:`inference.transformers.factory` imports only the selected
adapter when inference starts.
"""
