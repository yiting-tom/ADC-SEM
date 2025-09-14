# Repository Guidelines

## Project Structure & Module Organization
- Root modules: `dataset.py` (data loading), `model.py` (TinyViT mods), `ssl_training.py` (SSL pipeline), CLI scripts: `train_ssl.py`, `optimize_hyperparams.py`, `optuna_optimization.py`, `train_optimized.py`.
- Support files: `README.md`, `TIFF_SETUP_VERIFICATION.md`, `requirements.txt`.
- Example and outputs: `example_data/`, `lightning_logs/`, `outputs/` (artifacts, checkpoints). Do not commit large data or checkpoints.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Quick sanity run (small batch/epoch):
  - `python train_ssl.py --data_path ./example_data --batch_size 4 --epochs 1 --num_workers 0`
- Train (directory of files):
  - NPY: `python train_ssl.py --data_path ./data`
  - TIFF: `python train_ssl.py --data_path ./tiff_data --file_extension tiff`
- Hyperparameter optimization: `python optimize_hyperparams.py --data_path ./data --n_trials 50 --save_plots`
- Train with best config: `python train_optimized.py --config_path ./optimization_results/best_config.json --data_path ./data`

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4-space indentation, type hints where reasonable.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `ALL_CAPS`.
- Prefer `pathlib` for paths, f-strings for formatting, `logging`/Lightning loggers over bare prints (except CLI progress).
- Keep scripts idempotent and CLI-friendly; avoid hardcoded paths.

## Testing Guidelines
- No formal test suite yet. Validate changes with the “Quick sanity run” and a short Optuna trial: `--n_trials 5 --max_epochs 5`.
- Use `example_data/` for lightweight checks. When feasible, add minimal tests co-located with modules (e.g., `tests/test_dataset.py`).

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scope-first when helpful (e.g., `Add SSL training pipeline`, `Optimize TIFF loader`).
- PRs must include: purpose summary, reproduction commands, expected outputs (metrics/plots paths), and linked issues. Include screenshots of Optuna plots if relevant.
- Keep diffs focused; avoid unrelated refactors. Update `README.md` when changing CLI flags or defaults.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or API keys. Outputs live in `outputs/` and `lightning_logs/` (already ignored).
- Validate input shapes and channels; prefer explicit `--num_channels` when deviating from defaults.

## Agent-Specific Instructions
- Keep patches minimal and targeted. Follow naming and CLI patterns. If modifying CLI args, update help strings and corresponding docs.
