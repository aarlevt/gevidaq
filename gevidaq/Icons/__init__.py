import importlib.resources
import sys

_MODULE = sys.modules[__package__]
_FILES = importlib.resources.files(_MODULE)


class Path:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        trav = _FILES.joinpath(self.name)
        self.context = importlib.resources.as_file(trav)
        path = self.context.__enter__()
        return path.as_posix()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.context.__exit__(exc_type, exc_value, traceback)
