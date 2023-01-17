from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

template_dir = Path(__file__).parent / "templates"
environment = Environment(
    loader=FileSystemLoader(template_dir),
    lstrip_blocks=True,
    trim_blocks=True,
    undefined=StrictUndefined,
)
template = environment.get_template("nameprovider.j2")

known_ids = {"foo": ["foo2"], "bar": ["bar2"], "baz": ["baz2"]}


content = template.render(known_ids=known_ids)
print(content)


class Foo:
    a: int

    class _:
        c: int

    b: _

    class _:
        g: int

    d: _


a = Foo()
a.b.c
a.d.g
