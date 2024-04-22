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

convert_demos:
	@echo "Converting demos into integration tests..."
	@mkdir -p test_adjoint/demos
	@cd demos && for file in *.py; do \
		cp $$file ../test_adjoint/demos/test_demo_$$file; \
	done
	@cd test_adjoint && for file in demos/*.py; do \
		bash to_test.sh $$file; \
		ruff --fix $$file; \
	done
	@echo "Done."

test: lint convert_demos
	@echo "Running test suite..."
	@cd test && make
	@cd test_adjoint && make
	@echo "PASS"

coverage: convert_demos
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run -a --source=goalie -m pytest -v test
	@python3 -m coverage run -a --source=goalie -m pytest -v test_adjoint
	@python3 -m coverage html
	@cd test && make clean
	@cd test_adjoint && make clean
	@echo "Done."

demo:
	@echo "Running all demos..."
	@cd demos && make
	@echo "Done."

tree:
	@tree -d .

clean:
	@cd demos && make clean
	@cd test && make clean
