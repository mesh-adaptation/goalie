all: run

run:
	@echo "Running all non-adjoint tests..."
	@python3 -m pytest -v -n auto --durations=20 test_*.py
	@echo "Done."

clean:
	@echo "Running test caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing test outputs..."
	@rm -rf *.jpg *.png
	@rm -rf outputs*
	@echo "Done."
	@cd adjoint && make clean
