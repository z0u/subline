from typing import NamedTuple


def sliceable(cls: NamedTuple):
    """Decorator that adds advanced slicing to NamedTuples: foo['a c'], foo['b:d'], foo[[0,2]]"""
    original_getitem = cls.__getitem__

    def __getitem__(self, idx):
        if isinstance(idx, str):
            if ":" in idx:
                start, end = idx.split(":")
                start_idx = self._fields.index(start)
                end_idx = self._fields.index(end)
                return tuple(self[i] for i in range(start_idx, end_idx + 1))
            else:
                idx = idx.split(" ")
        if isinstance(idx, (list, tuple)):
            if all(isinstance(i, str) for i in idx):
                return tuple(getattr(self, i) for i in idx)
            return tuple(original_getitem(self, i) for i in idx)
        return original_getitem(self, idx)

    cls.__getitem__ = __getitem__
    return cls
