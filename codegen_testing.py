from pathlib import Path
import numpy as np
import os
import symforce

symforce.set_symbolic_api("symengine")
symforce.set_log_level("warning")
symforce.set_epsilon_to_number(1e-6)

from symforce import codegen
from symforce import cc_sym
from symforce.geo import Pose3
import symforce.symbolic as sf
from symforce.values import Values
from sym import Pose3

from se23.pose23_SE23 import Pose23_SE23
from symforce.codegen import geo_package_codegen

OUTDIR = Path(__file__).parent / "generated_output"

# geo_package_codegen.generate(config=codegen.CppConfig(), output_dir=OUTDIR)


a, b, c, d = sf.symbols("a b c d")
inputs = Values(inputs=[a, b, c, d])
outputs = Values(c=[a + b, a * b, a - b])


myfunction_codegen = codegen.Codegen(
    inputs=inputs,
    outputs=outputs,
    name="myfunction",
    config=codegen.CppConfig(),
)
myfunction_codegen_data = myfunction_codegen.generate_function(
    output_dir=OUTDIR,
    lcm_bindings_output_dir=OUTDIR / "lcm",
    skip_directory_nesting=True,
)
