from dataclasses import dataclass, field
import inspect
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
import sys
from shutil import copyfile
from functools import partial, wraps
import inspect
import sys
from typing import ClassVar, TypeVar, Generic, ParamSpec, Callable
import numpy as np

from symforce import symbolic as sf, codegen
from symforce.values import Values
from symforce import python_util
from symforce.geo import Matrix
from states import SymState
import subprocess
import jinja2
import re

CPP_DIR = Path(__file__).parent / "cpp"


Params = ParamSpec("Params")
RetVal = TypeVar("RetVal")
# Function = TypeVar("RetVal", bound=Callable[Params, RetVal])
@dataclass
class jinjafunc:
    name: str
    shape_in: tuple[int, ...]
    shape_out: tuple[int, ...]
    name_cpp: str = field(init=False)

    def __post_init__(self):
        self.name_cpp = python_util.snakecase_to_camelcase(self.name)


class FuncWrapper(Generic[Params, RetVal]):
    func: Callable[Params, RetVal]

    _registered: ClassVar[set["FuncWrapper"]] = set()
    _cmodule: ClassVar = None

    def __init__(self, func: Callable[Params, RetVal]):
        self.__class__._registered.add(self)
        self.func = func

    @classmethod
    def wrap(
        cls, func: Callable[Params, RetVal]
    ) -> "FuncWrapper[Params, RetVal]" | Callable[Params, RetVal]:
        """Remove this for python 11 and use Params in __call__ like here
        https://rednafi.github.io/reflections/static-typing-python-decorators.html"""
        return cls(func)

    def __call__(self, *args, **kwargs):
        return self.func(**kwargs)

    @property
    def name(self):
        return self.func.__name__

    @property
    def name_cpp(self):
        return python_util.snakecase_to_camelcase(self.name)

    def symbolic_input(self):
        inputs = Values()
        for k, v in inspect.signature(self.func).parameters.items():
            if hasattr(v.annotation, "symbolic"):
                inputs[k] = v.annotation.symbolic(k)
            elif v.annotation is sf.Scalar:
                inputs[k] = sf.symbols(k)
            else:
                raise ValueError(f"Unknown state type {v.annotation}")
        return inputs

    def cpp_input(self):
        inputs = self.symbolic_input()
        input_cpp = Values()
        for k, v in inputs.items():
            if isinstance(v, Matrix):
                input_cpp[k] = v
            elif hasattr(v, "to_storage"):
                input_cpp[k] = Matrix(v.to_storage())
            elif isinstance(v, sf.Symbol):
                input_cpp[k] = v
            else:
                raise ValueError(f"Unknown state type {v}")
        return input_cpp

    def symbolic_output(self):
        output = self.func(**self.symbolic_input())
        assert isinstance(output, (SymState, Matrix, sf.Scalar))
        return output

    def cpp_output(self):
        output = self.symbolic_output()
        if isinstance(output, Values):
            output_cpp = Values()
            for k, v in output.items():
                if isinstance(v, Matrix):
                    output_cpp[k] = v
                elif isinstance(v, sf.Symbol):
                    output_cpp[k] = v
                elif hasattr(v, "to_storage"):
                    output_cpp[k] = Matrix(v.to_storage())
                else:
                    raise ValueError(f"Unknown state type {v}")
            return output_cpp

        elif isinstance(output, Matrix):
            return Values(output=Matrix(output.to_storage()))

        elif isinstance(output, sf.Scalar):
            return Values(output=output)

        elif hasattr(output, "to_storage"):
            return Values({type(output).__name__: Matrix(output.to_storage())})

    def get_cpp_signatures(self, values: Values):
        sig_eigen = "Eigen::Matrix<Scalar, {}, {}>"
        sig_scalar = "Scalar"
        signatures = dict()
        for k, v in values.items():
            if isinstance(v, Matrix):
                signatures[k] = sig_eigen.format(*v.shape)
            elif isinstance(v, sf.Symbol):
                signatures[k] = sig_scalar
            else:
                raise ValueError(f"Unknown state type {type(v)}")
        return signatures

    def cpp_input_signatures(self):
        return self.get_cpp_signatures(self.cpp_input())

    def cpp_output_signatures(self):
        return self.get_cpp_signatures(self.cpp_output())

    def cpp_input_string(self):
        return ",".join(
            f"{v}* {k}" if v != "Scalar" else f"{v} {k}"
            for k, v in [
                *self.cpp_input_signatures().items(),
                *self.cpp_output_signatures().items(),
            ]
        )

    def cpp_call_string(self):
        inputs = ",".join(
            f"*reinterpret_cast<{v}*>({k})" if v != "Scalar" else f"{k}"
            for k, v in self.cpp_input_signatures().items()
        )
        outputs = ",".join(f"{k}" for k in self.cpp_output_signatures())
        return f"{inputs}, {outputs}"

    @classmethod
    def generate_cpp_funcs(cls):
        for func in cls._registered:
            inputs_cpp = func.cpp_input()
            outputs_cpp = func.cpp_output()

            preint_cgen = codegen.Codegen(
                inputs=inputs_cpp,
                outputs=outputs_cpp,
                config=codegen.CppConfig(),
                name=func.name,
            )
            info = preint_cgen.generate_function(
                output_dir=CPP_DIR / "generated_symforce", skip_directory_nesting=True
            )
            oldfile = info.generated_files[0]
            newfile = CPP_DIR / "generated" / oldfile.name
            if not newfile.is_file() or oldfile.read_bytes() != newfile.read_bytes():
                copyfile(oldfile, newfile)

    @classmethod
    def generate_bindings(cls):
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(CPP_DIR / "templates"),
            lstrip_blocks=True,
            trim_blocks=True,
            undefined=jinja2.StrictUndefined,
        )
        template = env.get_template("main.cpp.jinja")
        content = template.render(
            functions=cls._registered,
        )
        outfile = CPP_DIR / "generated/bindings.cpp"
        if not outfile.is_file() or content != outfile.read_text():
            outfile.write_text(content)

    @classmethod
    def compile_cpplib(cls):
        for func in cls._registered:
            func.create_cfunc()

        subprocess.run(
            "cmake -DCMAKE_BUILD_TYPE=Release ..",
            cwd=CPP_DIR / "build",
            check=True,
            shell=True,
        )
        subprocess.run("make", cwd=CPP_DIR / "build", check=True)

    @classmethod
    def import_cpplib(cls):
        module_name = "mylib"
        spec = spec_from_file_location(module_name, CPP_DIR / "build/mylib.so")
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        cls._cmodule = module

    @classmethod
    def compile_and_import(cls):
        cls.compile_cpplib()
        cls.import_cpplib()

    @property
    def cfunc(self):
        return getattr(self._cmodule, self.fname)

    def call_c(self, inputs):
        return getattr(self._cmodule, self.fname)(inputs)

    @staticmethod
    def to_cpp_inputs(inputs: Values, dtype=np.float64):
        cpp_inputs = {}
        for k, v in inputs.items():
            assert isinstance(v, SymState)
            cpp_inputs[k] = np.array(v.to_storage(), dtype=dtype)
        return cpp_inputs
