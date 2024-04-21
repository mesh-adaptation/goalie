all: install

.PHONY: demos test

install:
	@echo "Installing Goalie..."
	@python3 -m pip install -e .
	@echo "Done."
	@echo "Setting up pre-commit..."
	@pre-commit install
	@echo "Done."

lint:
	@echo "Checking lint..."
	@ruff check
	@echo "PASS"

test: lint
	@echo "Running test suite..."
	@cd test && make
	@cd test_adjoint && make
	@echo "PASS"

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run -a --source=goalie -m pytest -v test
	@python3 -m coverage run -a --source=goalie -m pytest -v test_adjoint
	@python3 -m coverage html

demo:
	@echo "Running all demos..."
	@cd demos && make
	@echo "Done."

tree:
	@tree -d .

clean:
	@cd demos && make clean
	@cd test && make clean
