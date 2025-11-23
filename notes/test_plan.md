# Test Plan

## Test Development Process

- **Read this test plan THOROUGHLY between writing tests**
- Don't skip ahead - understand the full context before implementing
- Each test should align with the overall testing strategy outlined above
- When in doubt, refer back to this plan before making decisions

## My initial notes for Claude

I have already set up some configuration for testing with pytest in the `pyproject.toml` file.

## How to Run Tests

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
./urvc uv run pytest -m "audio"       # Only audio tests
./urvc uv run pytest -m "cli"         # Only CLI tests

# Run tests in parallel (faster)
./urvc uv run pytest -n auto

# Run with detailed coverage report
./urvc uv run pytest --cov --cov-report=html
```

## Test Infrastructure Setup

The next step is to setup the rest of the testing infrastructure:

1. You should start by reading your general claude.md file for general info on how to do testing

2. Then you should read the project specific claude.md file in this workspace for more info on this project. This will help you determine what kinds of tests are necessary, and how these tests should be structured:
    - Folder structure
        - Where to put tests
        - How to name test files and test functions
    - Types of tests
        - Unit tests ?
        - Integration tests ?
        - End-to-end tests ?
    - Test cases to cover
        - Positive cases
        - Negative cases
        - Edge cases
        - Validation cases
    - The modules to test
        - including:
            - core module
            - cli module
            - web module
            - rvc module excluded
        - Which modules to test first
        - Which modules to test later
        - properly testing submodules
    - How to structure tests
        - How to use fixtures
        - How to use parametrization
    - What to do with test data, i.e. what to generate and what to use real data for
    - How to generate fake data
    - What needs to be mocked, in general we want to mock as little as possible

3. For further insight into this project you should read the README.md and also glance through the code itself

4. You should make a plan based on your findings. This plan should be appended to this file. You then extend the plan with more specific tasks and subtasks as progress, ideally split into sections and subsections which have items to cross off

5. Once you have a starting plan, you should report back to me so i can review it and we can discuss it further

6. once the plan is approved, you should test up a dummy test to test that all infrastructure is working as expected. If this requires setting up lots of files that is okay, but you should REMEMEBER to delete them once you are done(assuming they are not needed for the next step).
7. You should report your findings to me from the dummy experiment. Once we have established that the testing infrastructure is working, you should document all aspects of how to use it both in your own Claude.md file and also in the README.md file of the project. This should include:
   - How to run tests
   - How to add new tests
   - How to mock data
   - How to generate fake data
   - How to use fixtures
   - How to use parametrization
   - Any other relevant information

You should ULTRATHINK about all the points above. It is PARAMOUNT in particular that you research properly the best way of testing this project, given that it is a complex audio processing project with many moving parts, especially machine learning components that take a long time to run.

---

## Finalizd plan by Claude

### Analysis Summary

Based on my analysis of the project structure, CLAUDE.md guidance, and research into ML/audio testing best practices, I've developed a comprehensive testing strategy that addresses the unique challenges of the Ultimate RVC project.

### Key Project Characteristics Affecting Testing Strategy

1. **Complex Audio Processing Pipeline**: Multiple stages (vocal extraction, conversion, mixing)
2. **Machine Learning Components**: RVC models, PyTorch inference, embedders
3. **Multiple Interfaces**: Web UI (Gradio), CLI (Typer), and core Python API
4. **Long-running Operations**: Model training, audio processing, model downloads
5. **Hardware Dependencies**: CUDA/ROCm acceleration, audio processing
6. **External Dependencies**: YouTube downloads, model downloads from Discord

### General Testing Approach

- Test each public function with all edge cases, happy paths, and error conditions
- **ALWAYS do in-depth analysis** of each function before testing to understand exactly what cases require testing
- **ALWAYS run pre-commit hooks after finishing ANY test** to ensure it works correctly
  - Alternatively, run linting and type checking directly on the test files
- **Never claim tests are "fixed" or "done" without running pre-commit successfully**
  - This includes both type checking (pyright) and linting (ruff) validation
- **NEVER add noqa directives without EXPLICIT permission from user**
  - Always ask before suppressing any linting warnings or errors
  - Fix the underlying issue instead of suppressing it when possible

### Parallel Task Execution

- **Use Task tool for parallel work** when fixing clearly parallel tasks like separate files
- **Use Task tool when waiting for user input** - spawn subworkers to continue other tasks
- **Decide whether to keep or discard** subworker results based on user feedback
- This allows continuous progress rather than blocking on user input

### Test Development Order (Evidence-based from Google, Microsoft, Testing Pyramid)

1. Unit tests first (70-80% of total tests):  Black box testing of each public definition in each module
    - Decide case-by-case whether to use synthetic vs realistic audio. Mostly synthetic audio probably.
    - Decide case-by-case whether to mock models. There will probably be some degree of mocking for most tests.
2. integration tests: testing how definitions in different modules and packages interact
    - relevant only to the extend that unit tests of main/wrapper functions do not properly test their dependencies, i.e. by mocking them, their inputs or outputs
    - In this case need to test the interaction right after all the dependencies are tested either at
       - Intra module level (testing how different definitions in the same module interact)
       - inter module level (testing how different modules interact)
       - or inter package level (testing how different packages interact)
    - Use realistic audio and real models most of the time, but synthetic audio and mocked models are acceptable for some cases
3. E2E tests (5-10% of total tests):
    - Complete user workflows via CLI and Web UI
    - Realistic audio and real models
    - Performance and memory usage validation

### Test infrastructure

**IMPORTANT**: Implement below as needed, do not over-engineer but rather add infrastructure as required during test development. This includes:

#### Core Testing Infrastructure

- Test directory structure
- Base test classes for common patterns
- Conservative mocking for external APIs only
- tmp_path usage for file I/O testing

#### Test Data infratructure

- Realistic audio generators using libraries (chirps, speech synthesis, music-like audio) - small but realistic samples for fast testing
- audio generation Fixtures for each pipeline stage (raw audio, vocals, converted audio)
- Configuration file fixtures for testing different settings
- Test dataset creation utilities (primarily for training functionality)
- Minimal RVC model fixtures for testing (lightweight models)

## Critical Testing Principles

- All tests that are written should comply with the linting and type checking rules we have set up
- All tests should have at least 90% test coverage. This is a necessary requirement but not sufficient for tests to be done
- You should never move on to a new test before the above mentioned points are satisfied
- **NEVER change application code without EXPLICIT permission from user**
- **Always test ALL cases**: positive, negative, and edge cases for each function
- **Look for bugs in functions being tested** - most are correct, but some may have issues
- **Report any suspected bugs** but do not fix without permission
- **Test comprehensively** - every function parameter, return value, and exception path
  - You should work methodically and start small: this means starting with testing of the core module, and specifically those submodules that are used first, i.e. stuff like common files. Same methodology applies when testing other modules later on.
  - As you work, you should ALWAYS clean up after yourself, i.e. remove any files as soon as they are no longer needed. You should make a habit of asking yourself: "Do i need to clean up now?" after every step you take.
  - Whenever you discover something is not working as expected or figure out a better way to do something, you should document it in your claude.md file. This will help you remember it later and also help others who might work on this project in the future.
  - You should always err on the side of caution instead of writing more tests, especially in the beginning. Once we have a solid foundation, you can become more independent, by relying on the existing tests for structure and guidance.
  - NEVER THINK that you are done with this testing task.
    - Whenever you are inclined to think "The testing task is done" then instead look at the the file you just edited and go through each line, to see if something can be improved.
    - similarly, whenver you are inclined to think "Now I am done testing this module" instead iterate and look at each submodule and see if there is something that can be improved.
    - WHen you iterate, you should keep in mind not to regress by overengineering or removing things that are already working. Remember, you can always ask me if you are unsure about something.

### Phase 1: Testing Infrastructure Status

**Already Configured** ✅:

- pytest with strict configuration in pyproject.toml
- Coverage reporting (90% threshold)
- Test markers for categorization (slow, integration, gpu, network, audio, cli)
- Parallel testing with pytest-xdist
- Development dependencies including pytest-mock, faker

### Phase 2: Core Module Testing (Bottom-Up, Module-by-Module)

#### 2.1 core/manage module (Foundation - Test First)

**Rationale**: Foundational module that other modules depend on

- [ ] **models.py**: Model downloading, validation, and loading
- [ ] **config.py**: Configuration management
- [ ] **settings.py**: Settings management
- [ ] **audio.py**: Audio file management

#### 2.2 core/generate module (Depends on core/manage)

- [ ] **common.py**: Shared generation utilities
- [ ] **song_cover.py**: Song cover generation pipeline components
- [ ] **speech.py**: Text-to-speech functionality components

#### 2.3 core/train module (Depends on core/manage)

- [ ] **common.py**: Shared training utilities
- [ ] **extract.py**: Feature extraction functionality
- [ ] **prepare.py**: Dataset preparation functionality
- [ ] **train.py**: Model training functionality

### Phase 3: CLI Module Testing (Depends on All Core Modules)

- [ ] **cli/main.py**: Main CLI entry point
- [ ] **cli/generate/**: Generation commands
- [ ] **cli/train/**: Training commands

### Phase 4: Web Module Testing (Depends on All Core Modules)

**Development Order**: Unit tests → Intra-module integration → Inter-module integration with all core modules

- [ ] **web/config/**: Configuration management components
- [ ] **web/tabs/**: UI tab components

### Phase 5: End-to-End Testing (Develop Last - 5-10% of Total Tests)

**Development Order**: After all other testing is complete

**Note**: Research Playwright for Gradio testing, use realistic audio and real models

#### 5.1 Performance Testing

- [ ] **Audio processing performance benchmarks**: System-wide performance validation
- [ ] **Memory usage validation**: Resource consumption testing
- [ ] **GPU acceleration validation**: Hardware acceleration testing
- [ ] **Large file handling**: Stress testing with large audio files

#### 5.2 Complete User Workflow Testing

- [ ] **Complete workflows via CLI**: Full user scenarios with realistic data
- [ ] **Complete workflows via Web UI**: Full user scenarios with realistic data
- [ ] **External dependency integration**: Testing with real external services
- [ ] **Configuration migration testing**: User configuration scenarios

### Testing Strategy Details

#### Mock Strategy

**Mock These (Conservative approach)**:

- External API calls (YouTube, Discord)
- Model downloads (falls under external APIs)

**Don't Mock These**:

- Audio processing algorithms
- ML model inference (use small models)
- ML model training (use small datasets)
- Configuration validation
- Core business logic
- File I/O operations (use tmp_path instead)
- GPU operations (test real GPU paths in E2E tests)

#### Test Data Strategy

- Generate realistic audio using libraries (chirps, speech synthesis, music-like)
- Use minimal but realistic datasets for fast testing
- Create parametrized tests for multiple formats
- Use faker for generating test configurations
- **Unit tests**: Synthetic audio often acceptable (case-by-case decision)
- **Integration/E2E tests**: Realistic audio preferred unless consulted otherwise

#### Coverage Strategy

- Layer-by-layer testing from utilities to complex workflows
- Test all error paths and exception handling
- Test all configuration combinations
- Test integration boundaries between modules
- **Avoid test duplication**: Always check existing tests before creating new ones, especially integration tests at different levels

### Test Execution Strategy

#### Test Categories

- **Unit Tests**: Fast, isolated, no external dependencies
- **Integration Tests**: Module interaction testing
- **E2E Tests**: Complete workflow testing (run after integration tests)
- **Slow Tests**: Marked with @pytest.mark.slow
- **GPU Tests**: Marked with @pytest.mark.gpu
- **Network Tests**: Marked with @pytest.mark.network

#### CI/CD Integration

- Separate fast tests from slow tests
- E2E tests run after integration tests
- Use matrix testing for different configurations
- Cache dependencies and models

### Final Success Criteria

- [ ] 90%+ test coverage across all tested modules
- [ ] All tests pass consistently
- [ ] All linting and type checking passes
- [ ] Test suite runs in <5 minutes
- [ ] Comprehensive error handling validation
- [ ] Integration with CI/CD pipeline
- [ ] Documentation of testing procedures

### Risk Mitigation

1. **Long-running operations**: Use timeouts and mocking
2. **GPU dependencies**: Provide CPU fallbacks
3. **External dependencies**: Mock and provide offline alternatives
4. **Large files**: Use synthetic data and streaming
5. **Model dependencies**: Create minimal test models
