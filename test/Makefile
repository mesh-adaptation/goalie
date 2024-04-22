all: run

run:
	@echo "Running all tests..."
	@python3 -m pytest -v --durations=20 .
	@echo "Done."

clean:
	@echo "Removing test caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing test outputs..."
	@rm -rf *.jpg *.png *.pvd *.vtu
	@rm -rf outputs*
	@echo "Done."
