"""Alias de compatibilidad; PyAutoGUI fue reemplazado por ydotool."""

from autoplay.adapters import InputError, YdotoolInputBackend

__all__ = ["InputError", "YdotoolInputBackend"]
