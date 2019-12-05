# FreezeGroupNormalization
PoC / experiment / performance comparison of freeze group normalization with other state-of-the-art normalization method
- The implementation of `FreezeGroupNormalization` was based on keras and it was all run on colab.
- Used another model record system `wandb` for collecting and visualizing performance.
- Used `papermill` to control & execute notebooks like **notebook as a function**.

## Folder Structure
- src: implementation of `FreezeGroupNormalization` source code.
- configs: parameters file of model.
- train.ipynb: training script.
- main.ipynb: main entry for setup & execute `train.ipynb`.

## Result
- [Result Page](https://app.wandb.ai/kyoyachuan/Freeze%20Group%20Normalization%20Exp.?workspace=user-kyoyachuan)
