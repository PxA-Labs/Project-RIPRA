# Contributing to Project RIPRA

Thank you for your interest in contributing to Project RIPRA!

## Code Style
- **C/CUDA**: Follow the Linux kernel coding style. Use 4 spaces for indentation.
- **Python**: Follow PEP 8 guidelines. Use `black` and `flake8` for formatting and linting.
- **Documentation**: All public API functions must include Doxygen comments describing their purpose, parameters, return values, and mathematical derivations if applicable.

## Pull Request Process
1. Fork the repository and create your feature branch: `git checkout -b feature/my-new-feature`
2. Ensure your code passes all tests and memory safety checks (e.g., Valgrind, AddressSanitizer).
3. Use a [conventional commit](https://www.conventionalcommits.org/) prefix in your PR title (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `chore:`, `ci:`, `build:`). The prefix determines which section of `CHANGELOG.md` your entry will appear under.
4. Submit a Pull Request targeting the `main` branch, following the [Pull Request Template](.github/PULL_REQUEST_TEMPLATE.md).

After merge, a GitHub Action automatically appends the PR title to `CHANGELOG.md` based on the conventional-commit prefix. You do not need to manually edit the changelog.

## Issues
If you encounter a bug or have a feature request, please use the provided [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) or [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md) to submit your report. Include reproducible steps and logs where applicable.
