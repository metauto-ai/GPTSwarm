# This is a guide for developers to contribute to the codebase

## Configure the package for local development

### New way (poetry)

Install needed packages and GPTSwarm into the current environment.
```bash
poetry install
```

Install needed and uninstall not listed packages:
```bash
poetry install --sync
```

If dev packages not installed for some reason:
```bash
poetry install --with=dev --sync
```

### Old way (setuptools)

The following command installs the `gptswarm` package as a symbolic link to the current github repo clone. All edits to the repo will be immediately reflected in the "installed" package.
```bash
pip insall -e .
```

To install the package along with the developer's tools:
```bash
pip install -e .[dev]
```

## How to run tests

Quick no-API test without logging output:
```bash
pytest -m mock_llm test/
```

Quick no-API test with logging output:
```bash
pytest -s -m mock_llm test/
```

Without logging output:
```bash
pytest test/
```

With logging output:
```bash
pytest -s test/
```

Test specific function:
```bash
pytest -s test/swarm/graph/test_swarm.py -k 'test_raises'
```

Running any of the above commands with poetry will auto activate the virtual environment:

```bash
poetry run pytest -m mock_llm
```

## Run code coverage

```bash
coverage erase
coverage run --source=. -m pytest .
coverage html -i
open htmlcov/index.html
```

## Working with git LFS

[The instructions to work with git LFS (large file storage) can be found here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

## Working with submodules

[The instructions to work with git submodules can be found here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

## Packaging instructions

https://packaging.python.org/en/latest/tutorials/packaging-projects/

### Poetry

```bash
poetry config pypi-token.pypi "<your-token>"
poetry config repositories.test-pypi https://test.pypi.org/legacy/
poetry config pypi-token.test-pypi "<your-token>"

poetry version prerelease
poetry build
poetry publish -r test-pypi
poetry publish

pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple gptswarm==0.1.3a1
```

More see [here](https://stackoverflow.com/a/72524326/23308099).
