# First attempt to develop an experiment with modulating fields

See Oct 2024 slides for high-level idea: https://github.com/anhinga/2024-notes/tree/main/modulating-fields

There are a lot of ways to approach this; we are going to start with one of the most simple-minded.

Our starting point is to decorate all tokens in the current text with probabilities predicted by the model
and to use those probabilities to modify computations of attention matrices.

Let's design the changes for https://github.com/karpathy/nanoGPT/blob/master/model.py which are
needed for this. At the moment, I understand the overall code, but might be uncertain of some details.
(This is not quite from scratch, we have some notes and discussions.)

We are starting with only doing inference. Training or fine-tuning with a modified model will come later.

[inference-design.md](inference-design.md) - the file for detailed design work

