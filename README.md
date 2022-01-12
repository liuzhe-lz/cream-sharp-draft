# Usage #

0. Prepare a Cream ChildNet model.
1. Use `export(model, path)` from `cream/export_tool.py` to export architecture and weights.
2. Use `Loader.LoadArchAndWeights(path)` from `creamcs/Loader.cs` to load the model in C#.
3. Save checkpoint in TorchSharp native format: `model.save(...)`.

# Known Issues #

See "FIXME"s in `creamcs`.
