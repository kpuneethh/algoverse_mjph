# AlgoVerse Unit-Test Overfitting Experiment

A research project investigating how LLM permission levels affect reward hacking behaviors in coding tasks.

## Project Structure

- `explore/` - Experimental/PoC code (your active workspace)
- `src/` - Future production code (empty for now)
- `tests/` - Test suite for explore/ code
- `docs/` - Documentation and experiment design

## Quick Start

1. **Setup environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Run PoC experiment:**
   ```bash
   # Coming soon - will be in explore/poc_hardcoding.py
   ```

## Experiment Design

This project tests whether LLMs exhibit reward hacking behaviors when given different permission levels:

- **PL0 (Text-only)**: Model returns code as text, no execution
- **PL1 (Execute)**: Model can request test execution  
- **PL2 (Write)**: Model can modify test files

We measure gaming behaviors like hardcoding answers, test modification, and overfitting to visible test cases.

## Development Status

🚧 **Currently in Phase 0**: Environment setup and project scaffolding

## Contributing

This is a research project. See `docs/experiment_design.md` for methodology details.
