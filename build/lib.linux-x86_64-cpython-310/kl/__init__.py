from . import _C

reduce = _C.reduce
softmax = _C.softmax

__all__ = ["reduce", "softmax"]
