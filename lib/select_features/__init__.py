import importlib
import pkgutil

from GLOBAL import FS_CORR_MAX, FS_METHOD


def _selectors():
    found = {}
    for info in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{info.name}")
        select = getattr(mod, "select", None)
        if select is not None:
            found[info.name] = select
    return found


SELECT = _selectors()


def select_features(X, y, selected_cols=None, method=None):
    if selected_cols is None:
        from .drop_high_corr import drop_high_corr

        X = X[drop_high_corr(X, y, FS_CORR_MAX)]
        selected_cols = SELECT[FS_METHOD if method is None else method](X, y)
    return X[selected_cols], y, selected_cols
