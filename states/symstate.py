from dataclasses import fields, dataclass
from symforce.values import Values
import symforce.symbolic as sf
from typing import TypeVar
import numpy as np

Class = TypeVar("Class", bound="SymState")


class SymState(Values):
    """TODO311 use dataclass_transform"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for i, att in enumerate(fields(self)):
            if i < len(args):
                val = args[i]
            elif att.name in kwargs:
                val = kwargs[att.name]
            else:
                val = att.type()
            kwargs[att.name] = val
            self[att.name] = val

    @classmethod
    def symbolic(cls, name=""):  # pylint: disable=arguments-differ
        name = name or cls.__name__
        kwargs = dict()
        for att in fields(cls):
            if hasattr(att.type, "symbolic"):
                kwargs[att.name] = att.type.symbolic(f"{name}.{att.name}")
            elif att.type == sf.Scalar:
                kwargs[att.name] = sf.Symbol(f"{name}.{att.name}")
            else:
                raise ValueError(f"Unknown state type {att.type}")
        return cls(**kwargs)

    def __getattr__(self, item):
        if item in (f.name for f in fields(self)):
            return self[item]
        raise AttributeError(item)

    def __setattr__(self, key, value):
        if key in (f.name for f in fields(self)):
            self[key] = value
        else:
            super().__setattr__(key, value)
