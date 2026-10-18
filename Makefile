UV ?= uv
PYTHON ?= 3.12

.PHONY: sync lint test docs build ci ci-docker
sync:
	$(UV) sync --all-groups --frozen

lint:
	$(UV) run --frozen ruff check .
	$(UV) run --frozen ruff format --check .
	$(UV) run --frozen pyright
	uvx --from pydoclint==0.9.1 pydoclint .

test:
	$(UV) run --frozen pytest --cov=storecheck --cov-fail-under=85

docs:
	$(UV) run --frozen --group docs sphinx-build -W -b html docs docs/_build/html
	$(UV) run --frozen --group docs sphinx-build -W -b doctest docs docs/_build/doctest

build:
	$(UV) build
	$(UV) run --frozen --with twine twine check dist/*

ci: lint test docs build

ci-docker:
	docker run --rm -v "$(CURDIR):/source:ro" python:$(PYTHON)-slim sh -ec '\
		mkdir /work; \
		cp /source/pyproject.toml /source/uv.lock /source/README.md /source/LICENSE /work/; \
		cp -R /source/src /source/tests /source/examples /work/; \
		cd /work; \
		python -m pip install --quiet uv; \
		uv sync --frozen --no-default-groups --group test; \
		uv run --frozen --no-sync pytest --cov=storecheck --cov-fail-under=85; \
		uv build'
