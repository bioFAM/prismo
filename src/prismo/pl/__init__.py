import inspect

from . import _plotting

for _attrname in dir(_plotting):
    _attr = getattr(_plotting, _attrname)
    if (
        (inspect.isroutine(_attr) or inspect.isclass(_attr))
        and _attr.__module__.startswith(__package__)
        and not _attrname[0] == "_"
    ):
        globals()[_attrname] = _attr
