# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ultimate RVC is an AI voice cloning and conversion tool that extends AiCoverGen with improved features. It provides a comprehensive suite for:

- **Voice conversion**: Converting vocals using RVC (Retrieval-based Voice Conversion) models
- **Song cover generation**: Creating AI covers from YouTube videos or uploaded audio
- **Text-to-speech**: Generating speech from text using voice models  
- **Voice model training**: Training custom RVC models from audio datasets
- **Audio processing**: Advanced vocal extraction, noise reduction, pitch correction

The project supports both web UI (Gradio) and CLI interfaces, with GPU acceleration via CUDA/ROCm.

## Development Commands

### Setup and Environment

- `./urvc install` - Install all dependencies and set up environment (Linux/WSL)
- `./urvc.ps1 install` - Install dependencies (Windows PowerShell)
- `./urvc update` - Update to latest version
- `./urvc uninstall` - Clean uninstall of dependencies and data

### Running the Application

- `./urvc run` - Start web UI at <http://127.0.0.1:7860>
- `./urvc cli [subcommand]` - Access CLI interface
- `./urvc-web` - Start web UI (when installed as PyPI package)
- `urvc` - CLI interface (when installed as PyPI package)

### Development Tools

- `./urvc dev` - Start in development mode with hot reloading
- `./urvc docs [module] [output_dir]` - Generate documentation using Typer
- `./urvc uv [command]` - Run arbitrary uv commands
  - Code linting: `uv run ruff check` (configured in pyproject.toml)
  - Type checking: `uv run pyright` (Pyright configuration in pyproject.toml)
  - Code formatting: `uv run ruff format` (Black-compatible formatting)

### Testing

The project uses pytest for comprehensive testing with unit, integration, and end-to-end test coverage.

**Running Tests:**

```bash
# Run all tests
./urvc uv run pytest

# Run specific test file
./urvc uv run pytest tests/unit/core/test_common.py

# Run with verbose output
./urvc uv run pytest -v

# Run specific test class or method
./urvc uv run pytest tests/unit/core/test_common.py::TestDisplayProgress
./urvc uv run pytest tests/unit/core/test_common.py::TestDisplayProgress::test_display_progress_with_message_only

# Run tests matching pattern
./urvc uv run pytest -k "test_display_progress"

# Run with coverage report
./urvc uv run pytest --cov

# Run and stop on first failure
./urvc uv run pytest -x

# Run specific test categories
./urvc uv run pytest -m "not slow"     # Exclude slow tests
./urvc uv run pytest -m "network"     # Only network tests
./urvc uv run pytest -m "gpu"         # Only GPU tests
```

**Test Structure:**

- `tests/unit/` - Unit tests (isolated function testing)
- `tests/integration/` - Integration tests (module interactions)  
- `tests/e2e/` - End-to-end tests (full workflows)

**Test Requirements:**

- All tests must pass pre-commit hooks (linting, type checking)
- Minimum 90% test coverage required
- Use realistic data for integration/E2E tests, synthetic for unit tests

**Critical Testing Principles:**

- **NEVER change application code without EXPLICIT permission from user**
- **Always test ALL cases**: positive, negative, and edge cases for each function
- **Look for bugs in functions being tested** - most are correct, but some may have issues
- **Report any suspected bugs** but do not fix without permission
- **Test comprehensively** - every function parameter, return value, and exception path

## Architecture Overview
  
### Core Structure

```console
src/ultimate_rvc/
├── cli/           # Command-line interface using Typer
├── core/          # Core business logic
│   ├── generate/  # Audio generation pipelines
│   ├── manage/    # Model and data management  
│   └── train/     # Model training workflows
├── rvc/           # RVC model implementation
│   ├── configs/   # Sample rate configurations (32k/40k/48k)
│   ├── infer/     # Inference pipeline and models
│   ├── lib/       # Core algorithms, predictors, utilities
│   └── train/     # Training data processing and model training
├── web/           # Gradio web interface
│   ├── config/    # UI configuration management
│   └── tabs/      # Modular UI tabs (generate, manage, train)
└── stubs/         # Type stubs for external libraries
```

### Key Components

**Audio Processing Pipeline** (`core/generate/`):

- Vocal extraction using audio-separator with UVR models
- Pitch extraction with FCPE/RMVPE methods
- Voice conversion using RVC models
- Audio effects and post-processing

**Model Management** (`core/manage/`):

- Voice model downloading from URLs/AI Hub Discord
- Model upload and validation
- Pretrained model management
- Custom configuration persistence

**Training System** (`rvc/train/`):

- Dataset preparation and audio slicing
- Feature extraction with different embedders (ContentVec, HuBERT variants)
- RVC model training with configurable parameters
- Index creation for voice similarity matching

**Web Interface** (`web/`):

- Modular tab-based UI using Gradio 5
- Real-time configuration updates
- Audio preview and download capabilities
- Multi-step generation workflows for experimentation

## Technology Stack

- **Package Management**: uv with Python 3.13
- **ML Framework**: PyTorch 2.7+ with CUDA 12.8/ROCm support
- **Web UI**: Gradio 5.25+
- **CLI**: Typer with rich markup
- **Audio Processing**: librosa, soundfile, audio-separator, pedalboard
- **Code Quality**: Ruff (linting + formatting), Pyright (type checking)

## Environment Variables

Key environment variables for customization:

- `URVC_MODELS_DIR` - Model storage location
- `URVC_AUDIO_DIR` - Audio file storage  
- `URVC_LOGS_DIR` - Log file location
- `URVC_ACCELERATOR` - Hardware accelerator (`cuda` or `rocm`)
- `URVC_CONFIG` - Custom configuration name to load
- `YT_COOKIEFILE` - YouTube cookies for downloads

## Development Notes

- Uses lazy loading for heavy dependencies to improve startup time
- Extensive type hinting with py.typed marker
- Modular architecture allows independent development of components
- Caching system reduces inference time by reusing intermediate results
- Supports both one-click and multi-step generation workflows
- Configuration system allows saving/loading custom UI states

## Development Workflow (For Claude)

- **When user says "remember" something: ALWAYS Add it to this CLAUDE.md file**

  - User instructions prefixed with "remember" should be documented here
  - This creates a persistent record of important project-specific guidance

- **ALWAYS run commands from the project root directory**

  - Never run commands from subdirectories unless explicitly stated
  - This prevents path confusion and ensures consistent behavior

- **after completing a task ALWAYS**

  - stage relevant files with `git add`
  - run pre-commit hooks to ensure code quality
  - DO NOT commit unless asked to do so.

- **IMPORTANT: When commiting always include all changes for the given task**

### REMEMBER: Comprehensive testing reveals implementation issues

### Bug Discovery and Documentation Workflow

- **ALWAYS document implementation issues immediately when discovered during testing**
  - As soon as you discover something, document it in appropriate analysis files
  - Don't wait for user to ask for implementation issues
  - Create detailed analysis with severity, category, location, impact, and recommendations
  - After testing completion, user reviews the document to decide true positives vs false positives
  - Then fix true positives and update tests accordingly
  - This workflow ensures systematic issue tracking and prevents bugs from being forgotten

- **When implementing a new feature always publish a new version to pypi**

  1. use `./urvc uv version --bump [major|minor|patch]` to bump version
  2. Use `./urvc uv build` to build the package
  3. Use `./urvc uv publish` to publish to PyPI
  - Latter will require PyPI credential

### Git conventions

- **ALWAYS follow these git commit message guidelines**

- Use concise, one-line commit messages
- Example: "Initial commit: Advent of Code 2024 Rust workspace setup"
- **NEVER include co-authoring or Claude references in commit messages**

## Test Development Methodology

**Note**: Comprehensive test plan is located in `notes/test_plan.md`

### Test Development Order (Evidence-Based)

Based on research from Google, Microsoft, and the Testing Pyramid, ALWAYS follow this development order:

1. **Unit Tests First** (70-80% of total tests)
   - Test individual functions in isolation
   - Use mocks for external dependencies
   - Focus on edge cases, error conditions, and happy paths

2. **Intra-module Integration Tests** (after unit tests for each module)
   - Test how functions within the same module work together
   - Use realistic data where appropriate
   - Test module-level workflows

3. **Inter-module Integration Tests** (after dependent modules are unit tested)
   - Test how different modules interact
   - Test data flow between modules
   - Test module boundary conditions

4. **System Integration Tests** (after all module testing is complete)
   - Test cross-system workflows
   - Test system-wide error handling
   - Test performance at system level

5. **End-to-End Tests Last** (5-10% of total tests)
   - Test complete user workflows
   - Most expensive to maintain
   - Use realistic audio and real models

### Critical Testing Practices

**ALWAYS Follow These Rules:**

1. **Folder Structure Flexibility**
   - Test folder structure is NOT set in stone and should not restrict development
   - NEVER repeat tests - if there's a better way to bundle existing tests, restructure as needed
   - Test organization should serve the tests, not the other way around

2. **Quality Assurance Process**
   - **ALWAYS run pre-commit hooks after finishing ANY test** to ensure it works correctly
   - Alternatively, run linting and type checking directly on test files
   - **Never claim tests are "fixed" or "done" without running pre-commit successfully**
   - This includes both type checking (pyright) and linting (ruff) validation

3. **Test Development Process**
   - **Read the test plan THOROUGHLY between writing tests** (located in `notes/test_plan.md`)
   - Don't skip ahead - understand the full context before implementing
   - Each test should align with the overall testing strategy
   - When in doubt, refer back to the plan before making decisions

4. **Parallel Task Execution**
   - **Use Task tool for parallel work** when fixing clearly parallel tasks like separate files
   - **Use Task tool when waiting for user input** - spawn subworkers to continue other tasks
   - **Decide whether to keep or discard** subworker results based on user feedback
   - This allows continuous progress rather than blocking on user input

### Test Development Principles

- **ALWAYS do in-depth analysis** of each function before writing tests
- **Bottom-up approach**: Test foundational modules first (dependencies)
- **Module-by-module**: Complete all test types for a module before moving to next
- **Conservative mocking**: Only mock external APIs, not internal logic
- **Case-by-case decisions**: Audio and model mocking based on specific test needs
- **Integration/E2E tests**: Use realistic audio and real models most of the time
- **Avoid test duplication**: Always check existing codebase for similar tests before creating new ones
  - Especially important for integration tests at different levels (intra-module, inter-module, system)
  - Review existing test structure to avoid redundant coverage
  - Ensure each test has a unique purpose and scope

### Audio/ML Testing Strategy

- **Unit tests**: Synthetic audio often acceptable (decide case-by-case)
- **Integration tests**: Realistic audio preferred
- **E2E tests**: Always use realistic audio and real models
- **Performance tests**: Use realistic data for accurate measurements
- **Training tests**: Use small but realistic datasets

## Code Quality Enforcement

### Pre-commit Hooks (Active)

**Purpose**: Automatically enforce code quality standards before each commit,
preventing broken or poorly formatted code from entering the repository.

**Setup Instructions:**

```bash
# Install development dependencies (one-time setup)
./urvc uv sync --group dev

# Install pre-commit hooks in repository (one-time setup)
./urvc uv run pre-commit install

# Test hooks manually (optional)
./urvc uv run pre-commit run --all-files
```

**IMPORTANT - Never bypass hooks:**

- Never use `git commit --no-verify` unless absolutely necessary
- If hooks fail, fix the issues rather than bypassing them
- This maintains code quality and prevents technical debt

### Common Linting/Type Errors and Fixes

- **Docstring line breaking**: NEVER use `#` continuation in comments
  - ✅ Extend continuation lines to ~80 characters when possible unless at section end
  - ✅ Break at natural sentence/phrase boundaries for readability
- **Docstring format**: If docstrings can't fit on one line (>72 chars), use proper multi-line format:
  - ✅ Correct multi-line format - BREAK THE CONTENT within the docstring:

    ```python
    def function():
        \"\"\"
        Test ConfigNotFoundError with different configuration
        names.
        \"\"\"
    ```

  - ❌ NEVER use single-line docstrings that exceed 72 characters
  - ❌ NEVER shorten descriptive text to fit on one line
  - ❌ NEVER keep long content on single line within multi-line format
  
  **WRONG APPROACH**: Just putting triple quotes on separate lines but keeping long content on one line
  **CORRECT APPROACH**: Break the content itself across multiple lines

## Documentation Best Practices

- Follow Numpy-style docstrings for consistency

### Standard Documentation Sections (in order)

1. **Brief description** - What the function does (first paragraph)
2. **Detailed explanation (if relevant)** - How it works, algorithms used, performance characteristics
3. **Parameters**: Describe input parameters with types and default values if applicable
4. **Returns**: Describe return values with types
5. **Raises**: List exceptions that may be raised

- **Line Length Standard** : 79 characters maximum
If unsure of style consult existing code for examples.

## Research Methodology

**When user requests "ultrathink and research" or "deep research":**

### Phase 1: Primary Source Investigation

- **RFCs & Language Issues**: Search python rfcs for design decisions and rationale
- **Library Documentation**: Read actual API docs, not just descriptions
- **Community Forums**: Access Python Internals, not just user forums
- **Version History**: Research when features were added and why

### Phase 2: Cross-Domain Analysis

- **Multiple Domains**: Check game engines, scientific computing, image processing, etc..
- **Real Codebases**: Examine how major libraries solve problems
- **Performance Data**: Look for benchmarks and real-world measurements
- **Historical Context**: Understand evolution of approaches over time

### Phase 3: Evidence Quality Assessment

- **Authoritative**: Language RFCs, core team discussions, library maintainer posts
- **Practical**: Stack Overflow with multiple upvotes, real project examples
- **Speculative**: Blog posts, opinions without backing data
- **Verify Access**: Confirm you can actually read sources, not just descriptions

### Phase 4: Honest Reporting

- **Qualify Claims**: "Based on limited sources" vs "definitive consensus"
- **Cite Limitations**: Note 403 errors, paywalls, incomplete access
- **Multiple Perspectives**: Present competing viewpoints with evidence
- **Avoid Extrapolation**: Don't claim "industry standard" from one example
