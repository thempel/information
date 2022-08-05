import nox

PYTHON_VERSIONS = ["3.8", "3.9", "3.10"]

@nox.session(python=PYTHON_VERSIONS)
def tests(session):
    session.install("-e", ".", '-v', silent=False)
    session.install('pytest')
    session.run('pytest')
