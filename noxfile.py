import nox

    
poetryURL = "https://raw.githubusercontent.com"
poetryURL += "/python-poetry/poetry/master/get-poetry.py"


@nox.session
def lint(session):
    session.install("black", "isort", "flake8")
    session.run("isort", "--check", "halsey/")
    session.run("black", "--check", "halsey/")
    session.run("flake8", "halsey/")


@nox.session
def test_pipelines(session):
    session.run("curl", "-ssL", poetryURL, "|", "python")
    session.run("poetry", "install")
    session.run("pytest", "-s", "-v", "tests/test_pipelines/") 


@nox.session
def test_explorers(session):
    session.run("curl", "-ssL", poetryURL, "|", "python")
    session.run("poetry", "install")
    session.run("pytest", "-s", "-v", "tests/test_explorers/") 
