all: install

.PHONY: demos test

install:
	@echo "Installing Goalie..."
	@python3 -m pip install -e .
	@echo "Done."

install_dev:
	@echo "Installing Goalie for development..."
	@python3 -m pip install -e .[dev]
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
	@cd test/adjoint && make
	@echo "PASS"

# `mpiexec -n N ... parallel[N]` only runs tests with @pytest.mark.parallel(nprocs=N)
coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --parallel-mode --source=goalie -m pytest -v -k "not parallel" test
	@mpiexec -n 2 python3 -m coverage run --parallel-mode --source=goalie -m pytest -v -m parallel[2] test
	@python3 -m coverage combine
	@python3 -m coverage report -m
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
