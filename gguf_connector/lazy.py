from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
import logging
from typing import Any, Callable
import numpy as np
from numpy.typing import DTypeLike
logger = logging.getLogger(__name__)
class LazyMeta(ABCMeta):
    def __new__(cls, name, bases, namespace, **kwargs):
        def __getattr__(self, name):
            meta_attr = getattr(self._meta, name)
            if callable(meta_attr):
                return type(self)._wrap_fn(lambda s, *args, **kwargs:
                    getattr(s, name)(*args, **kwargs), use_self=self)
            elif isinstance(meta_attr, self._tensor_type):
                return type(self)._wrap_fn(lambda s: getattr(s, name))(self)
            else:
                return meta_attr
        namespace['__getattr__'] = __getattr__
        def mk_wrap(op_name, *, meta_noop: bool=False):
            def wrapped_special_op(self, *args, **kwargs):
                return type(self)._wrap_fn(getattr(type(self)._tensor_type,
                    op_name), meta_noop=meta_noop)(self, *args, **kwargs)
            return wrapped_special_op
        for binary_op in ('lt', 'le', 'eq', 'ne', 'ge', 'gt', 'notabs',
            'add', 'and', 'floordiv', 'invert', 'lshift', 'mod', 'mul',
            'matmul', 'neg', 'or', 'pos', 'pow', 'rshift', 'sub', 'truediv',
            'xor', 'iadd', 'iand', 'ifloordiv', 'ilshift', 'imod', 'imul',
            'ior', 'irshift', 'isub', 'ixor', 'radd', 'rand', 'rfloordiv',
            'rmul', 'ror', 'rpow', 'rsub', 'rtruediv', 'rxor'):
            attr_name = f'__{binary_op}__'
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=True)
        for special_op in ('getitem', 'setitem', 'len'):
            attr_name = f'__{special_op}__'
            namespace[attr_name] = mk_wrap(attr_name, meta_noop=False)
        return super().__new__(cls, name, bases, namespace, **kwargs)
class LazyBase(ABC, metaclass=LazyMeta):
    _tensor_type: type
    _meta: Any
    _data: Any | None
    _args: tuple
    _kwargs: dict[str, Any]
    _func: Callable[[Any], Any] | None
    def __init__(self, *, meta: Any, data: (Any | None)=None, args: tuple=(
        ), kwargs: (dict[str, Any] | None)=None, func: (Callable[[Any], Any
        ] | None)=None):
        super().__init__()
        self._meta = meta
        self._data = data
        self._args = args
        self._kwargs = kwargs if kwargs is not None else {}
        self._func = func
        assert self._func is not None or self._data is not None
    def __init_subclass__(cls):
        if '_tensor_type' not in cls.__dict__:
            raise TypeError(
                f"property '_tensor_type' must be defined for {cls!r}")
        return super().__init_subclass__()
    @staticmethod
    def _recurse_apply(o, fn):
        if isinstance(o, (list, tuple)):
            L = []
            for item in o:
                L.append(LazyBase._recurse_apply(item, fn))
            if isinstance(o, tuple):
                L = tuple(L)
            return L
        elif isinstance(o, LazyBase):
            return fn(o)
        else:
            return o
    @classmethod
    def _wrap_fn(cls, fn, *, use_self: (LazyBase | None)=None, meta_noop: (
        bool | DTypeLike | tuple[DTypeLike, Callable[[tuple[int, ...]],
        tuple[int, ...]]])=False):
        def wrapped_fn(*args, **kwargs):
            if kwargs is None:
                kwargs = {}
            args = ((use_self,) if use_self is not None else ()) + args
            meta_args = LazyBase._recurse_apply(args, lambda t: t._meta)
            if isinstance(meta_noop, bool) and not meta_noop:
                try:
                    res = fn(*meta_args, **kwargs)
                except NotImplementedError:
                    res = None
            else:
                assert len(args) > 0
                res = args[0]
                assert isinstance(res, cls)
                res = res._meta
                if meta_noop is not True:
                    if isinstance(meta_noop, tuple):
                        dtype, shape = meta_noop
                        assert callable(shape)
                        res = cls.meta_with_dtype_and_shape(dtype, shape(
                            res.shape))
                    else:
                        res = cls.meta_with_dtype_and_shape(meta_noop, res.
                            shape)
            if isinstance(res, cls._tensor_type):
                return cls(meta=cls.eager_to_meta(res), args=args, kwargs=
                    kwargs, func=fn)
            elif isinstance(res, tuple) and all(isinstance(t, cls.
                _tensor_type) for t in res):
                shared_args: list = [args, None]
                def eager_tuple_element(a: list[Any], i: int=0, /, **kw):
                    assert len(a) == 2
                    if a[1] is None:
                        a[1] = fn(*a[0], **kw)
                    return a[1][i]
                return tuple(cls(meta=cls.eager_to_meta(res[i]), args=(
                    shared_args, i), kwargs=kwargs, func=
                    eager_tuple_element) for i in range(len(res)))
            else:
                del res
                eager_args = cls.to_eager(args)
                return fn(*eager_args, **kwargs)
        return wrapped_fn
    @classmethod
    def to_eager(cls, t):
        def simple_to_eager(_t):
            if _t._data is not None:
                return _t._data
            assert _t._func is not None
            _t._args = cls._recurse_apply(_t._args, simple_to_eager)
            _t._data = _t._func(*_t._args, **_t._kwargs)
            assert _t._data is not None
            assert _t._data.dtype == _t._meta.dtype
            assert _t._data.shape == _t._meta.shape
            return _t._data
        return cls._recurse_apply(t, simple_to_eager)
    @classmethod
    def eager_to_meta(cls, t):
        return cls.meta_with_dtype_and_shape(t.dtype, t.shape)
    @classmethod
    @abstractmethod
    def meta_with_dtype_and_shape(cls, dtype, shape):
        pass
    @classmethod
    def from_eager(cls, t):
        if type(t) is cls:
            return t
        elif isinstance(t, cls._tensor_type):
            return cls(meta=cls.eager_to_meta(t), data=t)
        else:
            return TypeError(
                f'{type(t)!r} is not compatible with {cls._tensor_type!r}')
class LazyNumpyTensor(LazyBase):
    _tensor_type = np.ndarray
    shape: tuple[int, ...]
    @classmethod
    def meta_with_dtype_and_shape(cls, dtype, shape):
        cheat = np.zeros(1, dtype)
        return np.lib.stride_tricks.as_strided(cheat, shape, (0 for _ in shape)
            )
    def astype(self, dtype, *args, **kwargs):
        meta = type(self).meta_with_dtype_and_shape(dtype, self._meta.shape)
        full_args = (self, dtype) + args
        return type(self)(meta=meta, args=full_args, kwargs=kwargs, func=lambda
            a, *args, **kwargs: a.astype(*args, **kwargs))
    def tofile(self, *args, **kwargs):
        eager = LazyNumpyTensor.to_eager(self)
        return eager.tofile(*args, **kwargs)