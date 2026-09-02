# makemore — Neural Networks: Zero to Hero

My notes and code from working through Andrej Karpathy's
[Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series. It involves building a character-level language model that makes more names, starting from a scalar autograd engine written from scratch.

The notebooks are split into two folders: `1_Lectures` holds the rough notes taken while
following each lecture, and `2_Extensions` holds cleaner notebooks written afterwards to extend the ideas.

## Contents

### 1_Lectures

| Notebook | What it covers |
| --- | --- |
| [`1_micrograd.ipynb`](1_Lectures/1_micrograd.ipynb) | A scalar-valued `Value` class to perform backpropogation, plus graphviz plots of the computation graph. |
| [`2_bigrams.ipynb`](1_Lectures/2_bigrams.ipynb) | A bigram character model, first by counting, then as a single-layer network trained with gradient descent. Notes on likelihood, log-likelihood and negative log-likelihood as a loss. |
| [`3_mlp.ipynb`](1_Lectures/3_mlp.ipynb) | The Bengio-style MLP: character embeddings, a `tanh` hidden layer, minibatch training and a train/dev/test split. Covers machine learning fundamentals like model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting etc |
| [`4_mlp2.ipynb`](1_Lectures/4_mlp2.ipynb) | Making the same MLP train properly. Why a badly initialised network wastes the first steps on squashing the logits, why `tanh` saturates at ±1 and kills the gradient, scaling the weights by `1/sqrt(fan_in)` (Kaiming init), and batch normalisation of the hidden pre-activations — including why the bias is redundant once you normalise, and the running mean/std kept for inference. |

Papers the lectures follow:

- [`3_lm_jmlr.pdf`](1_Lectures/3_lm_jmlr.pdf) — Bengio et al. (2003), *A Neural Probabilistic Language Model* (lecture 3).
- [`4_He.pdf`](1_Lectures/4_He.pdf) — He et al. (2015), *Delving Deep into Rectifiers* (lecture 4, weight initialisation).
- [`4_Ioffe_Szegedy.pdf`](1_Lectures/4_Ioffe_Szegedy.pdf) — Ioffe & Szegedy (2015), *Batch Normalization* (lecture 4).

### 2_Extensions

| Notebook | What it covers |
| --- | --- |
| [`1_micrograd_exercises.ipynb`](2_Extensions/1_micrograd_exercises.ipynb) | The micrograd exercises: filling in the missing `Value` operations (`exp`, `log`, division, negation) and using them to build softmax and a negative log-likelihood loss by hand, then checking the gradients against PyTorch. |
| [`2_trigrams.ipynb`](2_Extensions/2_trigrams.ipynb) | The bigram network generalised to any n-gram. One-hot encodes the previous `n-1` characters, trains a small `tanh` hidden layer on minibatches, and samples from the result. |
| [`3_mlp_clean.ipynb`](2_Extensions/3_mlp_clean.ipynb) | The lecture 3 MLP rewritten from scratch as tidy, reusable functions (`build_dataset`, `train_model`, `evaluate`), with a scatter plot of the learned 2D character embeddings. |
| [`3_mlp_experiments.ipynb`](2_Extensions/3_mlp_experiments.ipynb) | The MLP rewritten as a function of its hyperparameters, then tuned one at a time against a fixed baseline: learning rate, hidden-layer width, embedding dimension and context length. Ends with a summary table of what each change bought and a final model evaluated once on the test set. |
