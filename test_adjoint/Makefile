all: run

run:
	@echo "Running all adjoint tests..."
	@python3 -m pytest -v --durations=10 .
	@echo "Done."

clean:
	@echo "Removing adjoint test caches..."
	@rm -rf __pycache__
	@echo "Done."
	@echo "Removing demo tests..."
	@rm -rf demos
	@echo "Done."
	@echo "Removing adjoint test outputs..."
	@rm -rf *.jpg *.png *.pvd *.vtu
	@rm -rf outputs*
	@echo "Done."
