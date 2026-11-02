import importlib
import pkgutil


def _methods():
    found = {}
    for info in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{info.name}")
        fit = getattr(mod, "fit", None)
        if fit is not None:
            found[info.name] = fit
    return found


METHODS = _methods()


class Trainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def train(self, method, **kwargs):
        return METHODS[method](self.X, self.y, **kwargs)
