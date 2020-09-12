import nox


@nox.session
def lint(session):
    session.install("black", "flake8")
    session.run("black", "--check", "halsey/")
    session.run("flake8", "halsey/")
