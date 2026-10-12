# Contributing to ProDMM

Start with the [source map](README.md#source-map) and [development guide](docs/development.md). Keep changes focused and preserve existing import paths, vocabulary IDs, checkpoint parameter names, and sequence formats unless an intentional migration is included.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

Install `.[train,dev]` when working on training code. Without the training extra, its tests are skipped; CI runs them in a separate training job. Tests should run on CPU without downloading pretrained weights or requiring private data.

## Making a change

1. Describe the input or task you are improving and the expected output.
2. Change the smallest relevant module. Separate data parsing, model inference, and output handling when adding new workflows.
3. Add a regression test for changed scoring, tokenization, model behavior, or file formats. Use synthetic inputs and tiny models.
4. Run `python -m pytest`, `python -m build`, and `git diff --check`.
5. Explain the behavior change and validation in the pull request. For a performance claim, include hardware, input sizes, and a reproducible benchmark.

The model files retain upstream research naming and checkpoint structure. Avoid renaming their parameters just for style: pretrained weights depend on these names. New public functions should have a short docstring that states accepted inputs, returned values, and positional conventions.

## Reporting issues

Include the command or minimal Python example, Python/PyTorch/Transformers versions, CPU or GPU details, expected behavior, and traceback. Remove credentials and private data before sharing. For research results, specify the checkpoint and dataset version.

Contributions are covered by this repository's [license](LICENSE). Do not include model checkpoints, generated scoring outputs, caches, or private datasets in commits.
