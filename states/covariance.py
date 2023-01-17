from symforce import typing as T
from symforce.geo import Matrix
from symforce.ops.interfaces import Storage


class Cov(Storage):
    mat: Matrix
    SHAPE: T.Tuple[int, int]

    def __init__(self, *args, _mat=None, **kwargs):
        mat = self.symmetrize(_mat or Matrix(*args, **kwargs))
        self.mat = mat

    @classmethod
    def symbolic(cls, name, **kwargs):
        obj = object.__new__(cls)
        obj.mat = cls.symmetrize(Matrix(*cls.SHAPE).symbolic(name))
        return obj

    @classmethod
    def diag(cls, *args):
        obj = object.__new__(cls)
        obj.mat = Matrix.diag(*args)
        return obj

    @staticmethod
    def symmetrize(mat: Matrix):
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[0]):
                mat[i, j] = mat[j, i]
        return mat

    @property
    def shape(self):
        return self.SHAPE

    @classmethod
    def storage_dim(cls) -> int:
        return cls.SHAPE[0] * (cls.SHAPE[1] + 1) // 2

    def to_storage(self) -> T.List[T.Scalar]:
        storage = []
        rows = self.shape[0]
        for i, j in ((i, j) for i in range(rows) for j in range(i + 1)):
            storage.append(self.mat[i, j])
        return storage

    @classmethod
    def from_storage(cls, elements: T.Sequence[T.Scalar]) -> Matrix:
        obj = object.__new__(cls)

        mat = Matrix.zeros(*cls.SHAPE)
        rows = cls.SHAPE[0]
        for e, (i, j) in enumerate(((i, j) for i in range(rows) for j in range(i + 1))):
            mat[i, j] = mat[j, i] = elements[e]
        obj.mat = mat
        return obj

    def inv(self):
        return self.mat.inv()

    def __mul__(self, other):
        return self.mat * other

    def __add__(self, other):
        return self.mat + other

    def __radd__(self, other):
        return self.mat + other

    def __rmul__(self, other):
        return self.mat * other

    def __sub__(self, other):
        return self.mat - other

    def __rsub__(self, other):
        return other - self.mat

    def __getitem__(self, item):
        return self.mat[item]


class Cov99(Cov):
    SHAPE = (9, 9)


class Cov66(Cov):
    SHAPE = (6, 6)


class Cov33(Cov):
    SHAPE = (3, 3)
