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

[`3_lm_jmlr.pdf`](1_Lectures/3_lm_jmlr.pdf) is Bengio et al. (2003), *A Neural Probabilistic Language Model* — the paper lecture 3 follows.

### 2_Extensions

| Notebook | What it covers |
| --- | --- |
| [`3_mlp_experiments.ipynb`](2_Extensions/3_mlp_experiments.ipynb) | The MLP rewritten as a function of its hyperparameters, then tuned one at a time against a fixed baseline: learning rate, hidden-layer width, embedding dimension and context length. Ends with a summary table of what each change bought and a final model evaluated once on the test set. |