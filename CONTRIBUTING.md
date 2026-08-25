# Contributing to ScalarPulse

Thanks for helping make ScalarPulse better. Bug fixes, documentation improvements,
tests, and focused feature proposals are all welcome.

## Set up a development environment

ScalarPulse requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the package and development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`. On macOS or Linux, use
`source .venv/bin/activate`.

## Run the checks

Run the full test suite before submitting a pull request:

```bash
python -m pytest
python -m build
```

You can also smoke-test the CLI locally:

```bash
scalarpulse --version
scalarpulse --help
```

## Propose a change

1. Search existing issues before opening a new one.
2. For bugs, include a minimal reproduction, expected behavior, and environment
   details.
3. Keep pull requests focused on one change.
4. Add or update tests for behavior changes.
5. Update documentation when users need to learn a new command or API.

## Code guidelines

- Keep the core package dependency-free unless a change has a strong,
  documented reason.
- Preserve compatibility with Python 3.10 through 3.13.
- Avoid importing optional machine-learning frameworks from the core package.
- Prefer small, readable changes and clear error messages.
- Never commit training data, generated run logs, secrets, or credentials.

By contributing, you agree that your contribution is licensed under the
repository's MIT License.
