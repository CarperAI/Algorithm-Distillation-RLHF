SHELL=/bin/bash
LINT_PATHS=algorithm_distillation/

pytest:
	python -m pytest tests/

pytype:
	pytype ${LINT_PATHS}

type: pytype

lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 ${LINT_PATHS} --count --select=E9,F63,F7 --show-source --statistics --max-line-length=88
	flake8 ${LINT_PATHS} --count --exit-zero --ignore=E203,E501,F811 --show-source --statistics --max-line-length=88

format:
	# Sort imports
	isort ${LINT_PATHS}
	# Reformat using black
	black -l 88 ${LINT_PATHS}

check-codestyle:
	# Sort imports
	isort --check ${LINT_PATHS}
	# Reformat using black
	black --check -l 88 ${LINT_PATHS}

commit-checks: format type lint

clean:
	cd docs && make clean

.PHONY: clean lint format check-codestyle commit-checks