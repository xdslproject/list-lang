import pytest

from main import program_to_mlir_module
from source import ParseError


def test_line_col():
    CODE = """

let x = 0;
let y = x + true;
"""

    with pytest.raises(ParseError) as err:
        program_to_mlir_module(CODE)

    assert err.value.line_column(CODE) == (4, 13)
    assert (
        err.value.msg == "expected expression of type int in addition, got bool"
    )
