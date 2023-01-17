from dataclasses import dataclass
from symforce.geo import Vector3
from se23.pose23_SE23 import Pose23_SE23

from .covariance import Cov99, Cov66
from .symstate import SymState


KWARGS = dict(init=False, repr=False)


@dataclass(**KWARGS)
class ImuNoise(SymState):
    gyro: Vector3
    accl: Vector3

    @property
    def cov(self):
        return Cov66.diag([*self.gyro, *self.accl])


@dataclass(**KWARGS)
class ImuBias(SymState):
    gyro: Vector3
    accl: Vector3


@dataclass(**KWARGS)
class ZImuEst(SymState):
    gyro: Vector3
    accl: Vector3


@dataclass(**KWARGS)
class ZImuRaw(SymState):
    gyro: Vector3
    accl: Vector3

    def __sub__(self, other: ImuBias) -> ZImuEst:
        assert isinstance(other, ImuBias)
        return ZImuEst(self.gyro - other.gyro, self.accl - other.accl)


@dataclass(**KWARGS)
class ImuPreint(SymState):
    upsilon: Pose23_SE23
    cov: Cov99


@dataclass(**KWARGS)
class State(SymState):
    nom: Pose23_SE23
    err_cov: Cov99
    imu_bias: ImuBias
