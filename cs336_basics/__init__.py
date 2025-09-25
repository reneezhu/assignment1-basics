import importlib.metadata

# __version__ = importlib.metadata.version("cs336_basics")
try:
    # Attempt to fetch version using standard method
    from importlib.metadata import version
    __version__ = version("cs336_basics")
except Exception:
    # Fallback for development, testing, or failed installation
    __version__ = "unknown"
