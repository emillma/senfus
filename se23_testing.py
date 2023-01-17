"""module for testing stuff"""

from pathlib import Path
from functools import partial, wraps
import inspect
import symforce
import numpy as np

symforce.set_symbolic_api("symengine")  # pylint: disable=wrong-import-position
symforce.set_epsilon_to_number(1e-6)  # pylint: disable=wrong-import-position

from symforce import symbolic as sf
from symforce.geo import Matrix
from symforce.values import Values

from se23.pose23_SE23 import Pose23_SE23
from se23.integration import preintegrate, Phi, Gamma
from states import ImuNoise, ZImuRaw, ZImuEst, ImuPreint, State, SymState, Cov99
from codegen.get_code import FuncWrapper
import sympy as sp
import pybind11
import array

OUTPUT_DIR = Path(__file__).parent / "codegen/cpp/generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n = 100
a = Cov99.symbolic("a")
a, b, c, d, e, f = sf.symbols("a b c d e f")


@FuncWrapper.wrap
def myfunc(inputs: Pose23_SE23):
    return inputs.rotation().to_rotation_matrix()


preintegrate = FuncWrapper.wrap(preintegrate)

FuncWrapper.generate_bindings()
FuncWrapper.generate_cpp_funcs()

# FuncWrapper.compile_and_import()


# preintegrate = FuncWrapper.wrap(preintegrate)
# preintegrate.crete_lib()


# preintegrate.import_clib()

# inputs = np.random.random(68)
# out = np.zeros(91)
# out = preintegrate.call_c(inputs)
# here = True
# inputs = wrapped_func.get_inputs()


# outfile = OUTPUT_DIR / "preintegrate.h"
# FuncWrapper.crete_lib()
# print(dir(FuncWrapper._clib))
# pycode = wrapped_func.create_cfunc()


# wrapped_func.get_numba_func(output_file=outfile)
# wrapped_func.get_cfunc(output_file=outfile.with_suffix(".h"))
# # numba_func = wrapped_func.get_numba_func()
# # out = preintegrate(imu_noise, preint_prev, z_imu_est, dt)
# # ins = inspect.signature(preintegrate)
# here = True
# out = preintegrate(imu_noise=)
