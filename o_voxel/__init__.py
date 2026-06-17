from . import (
    convert,
    io,
    rasterize,
    serialize
)


def __getattr__(name):
    if name == "postprocess":
        import importlib
        mod = importlib.import_module('.postprocess', __name__)
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")