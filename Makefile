init:
	( \
		python3 -m venv .venv; \
		source .venv/bin/activate; \
		python3 -m pip install --upgrade pip; \
		python3 -m pip install poetry; \
		pre-commit install; \
		poetry install; \
	)

validate: .pre-commit-config.yaml
	( \
		source .venv/bin/activate; \
		pre-commit run --all-files; \
	)

build:
	( \
		source .venv/bin/activate; \
		poetry build; \
	)

update:
	( \
		source .venv/bin/activate; \
		poetry update; \
		pre-commit autoupdate; \
	)

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	find . -name "*.pyc" -type f -delete
