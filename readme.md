## Getting started
1. replace all instances of `project_name` with your project name.
If you are using VSCode, you can use the `replace all (ctrl + shift + H)` function to do this.
Use lowercase, capitalized names are not supported in docker.

3. Edit the author name in `conf.py` in the docs folder.

4. Create a virtual environment and activate it
```bash
python -m venv .venv
```
4. install base requirements
```bash
pip install requirements/base.txt
```
## Get help about the cli tool in the terminal
To view the available commands, run

```bash
.\manage --help
```

## Dependency management

This project uses [pip-compile-multi](https://pypi.org/project/pip-compile-multi/) for hard-pinning dependencies versions.
Please see its documentation for usage instructions.
In short, `requirements/base.in` contains the list of direct requirements with occasional version constraints (like `Django<2`)
and `requirements/base.txt` is automatically generated from it by adding recursive tree of dependencies with fixed versions.
The same goes for `testing` and `development`.

### Pinning dependencies
To generate pinned versions of the project's dependencies, run 

```bash
python manage.py pin
```

### Upgrading dependencies
To upgrade pinned dependencies, run

```bash
python manage.py pin --upgrade
```

### Installing dependencies
To install pinned dependencies, run

```bash
python manage.py install [environment]
```
where `environment` is one of `base`, `testing` or `development`. If not specified, `base` is used.


### Synchronizing dependencies
To synchronize pinned dependencies with the current environment, run
```bash
python manage.py sync [environment]
```
where `environment` is one of `base`, `testing` or `development`. If not specified, `base` is used.

this command will first run deinstall all unused dependencies, generate pinned dependencies and then install all missing dependencies.


## Using Docker to run the project
Before running, unsure docker is installed on your machine.

### Running in Production Environment in Docker

To deploy the project in a production environment, execute the following command:

```shell
docker python manage.py docker prod
```

This command will start the project in a detached mode, suitable for production use.

### Running in Development Environment in Docker
For development purposes, use the following command to run the project:

```shell
docker python manage.py docker dev
```

This command will start the project in the development environment, providing you with the necessary tools for testing and debugging.

### Running Tests in Docker
For testing purposes, use the following command to run the projects tests:

```bash
docker python manage.py docker test
```

This command will start the project services and evaluate all available tests.


## Running tests locally
Ensure all test dependencies are installed by running
```bash
python manage.py install testing
```

To run tests suite, use the following command:
```bash
python manage.py test [type]
```
where `type` is the type of tests to run. If not specified, only unit tests are run.
Available types are:
- `unit` - unit tests
- `integration` - integration tests
- `functional` - functional tests
- `all` - all tests

## Generating Sphinx documentation
To generate documentation, run
```bash
python manage.py docs
```
This will generate documentation in `docs/_build/html` directory.

### Viewing documentation
To view documentation, open `docs/_build/html/index.html` in your browser. A documentation template is provided that is ready to be filled in, the arc42 template is used for the software architecture section. 

## Check code style and type checking
To check code style and type checking, run
```bash
python manage.py check
```
This will run `ruff` and `mypy` on the project with the configurations found in `pyproject.toml`.

These checks are also run automatically in CI/CD on every push/pull request to main. Ensure all checks pass before pushing your code. Consider using a pre-commit hook to run these checks automatically before every commit (not part of the template).

## Running the project locally
Ensure all dependencies are installed by running
```bash
python manage.py install
```

To run the project, use the following command:
```bash
python manage.py run
```
This will run the script in `src/main.py`, which should be the main entrypoint to the application.






