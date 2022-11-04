import os
import tempfile
from pathlib import Path

import nox

PYTHON_VERSIONS = ["3.8", "3.9", "3.10", "3.11"]

@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    session.install("-e", ".", '-v', silent=False)
    session.install("-r", "informant/tests/requirements.txt", silent=False)

    pytest_args = []
    if 'cov' in session.posargs:
        session.log("Running with coverage")
        xml_results_dest = Path(os.getenv('SYSTEM_DEFAULTWORKINGDIRECTORY', tempfile.gettempdir()))
        assert xml_results_dest.exists() and xml_results_dest.is_dir(), 'no dest dir available'
        cover_pkg = 'informant'
        #junit_xml = str((xml_results_dest / 'junit.xml').absolute())
        cov_xml = str((xml_results_dest / 'coverage.xml').absolute())

        pytest_args += [f'--cov={cover_pkg}', f"--cov-report=xml:{cov_xml}", #f"--junit-xml={junit_xml}",
                        "--cov-config=.coveragerc"]
    else:
        session.log("Running without coverage")

    session.run('pytest', *pytest_args)
