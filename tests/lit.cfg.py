import lit.formats

from pathlib import Path

config.name = "list-lang-parser-tests"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".lsl"]

project_folder = Path(__file__).parent.parent
parse_python_script = project_folder.joinpath("src").joinpath("main.py")

config.substitutions.append((r"%lsl-parser", parse_python_script))
