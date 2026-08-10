"""Compatibilidad mínima: el autoplay vigente vive en :mod:`autoplay.cli`."""

from autoplay.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
