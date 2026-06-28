# Contributing to Project RIPRA

Thank you for your interest in contributing to Project RIPRA!

## Code Style
- **C/CUDA**: Follow the Linux kernel coding style. Use 4 spaces for indentation.
- **Python**: Follow PEP 8 guidelines. Use `black` and `flake8` for formatting and linting.
- **Documentation**: All public API functions must include Doxygen comments describing their purpose, parameters, return values, and mathematical derivations if applicable.

## Pull Request Process
1. Fork the repository and create your feature branch: `git checkout -b feature/my-new-feature`
2. Ensure your code passes all tests and memory safety checks (e.g., Valgrind, AddressSanitizer).
3. Update the `CHANGELOG.md` with your changes.
4. Submit a Pull Request targeting the `main` branch.

## Issues
If you encounter a bug or have a feature request, please use the provided GitHub Issue templates to submit your report. Include reproducible steps and logs where applicable.
