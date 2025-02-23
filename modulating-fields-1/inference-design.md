# designing changes for nanogpt/model.py (inference only)

1) quite a bit of changes in `generate` method of `GPT` class (last method of the file)

  * it needs to accept probability values from the previous transformer run (if this is the first run, we should have some graceful default, like 1)

  * it needs to use those probability values to modify attention computations (so they need to be passed through down to `forward` methods of
    `Block` and `CausalSelfAttention` classes)    
  
  * `generate` should generate its current probabilities for all tokens (an interesting aspect here is that these estimates are non-stationary,
    which is different from standard autoregressive scheme, where past probabilities would not change during recompute, and this is an aspect
    which we'll need to investigate separately (first we'll just see if they exhibit a reasonable convergence behavior as one recomputes, and
    then we'll ponder what to do about it, if the values need to be fixes after one or several iterations (hopefully this will not be needed))) 

2) mini-optimization in lines 184-191 needs to be removed (we are going to compute probabilities for all tokens, and so we need to keep the logits

3) in CausalSelfAttention the use of flash attention needs to be turned off (unless we feel like modifying some PyTorch code) and
   before or after `att = F.softmax(att, dim=-1)` (most likely before, in order to not interfere with softmax normalization)
   we are going to play with adjusting the attention (let's consider this as a starting point: we have **A*V**, and the softmax-normalized
   rows of **A** contain coefficients of linear combinations of rows of **V**; what we can do is to multiply values in each row of **A** by the
   predicted probability of the token (what's not to like here: 1) not quite clear how to make this correction stronger or weaker,
   and we would really like to have a parameter controlling this effect; 2) if we are adjusting things pre-softmax, we should be able
   to use logits directly, rather than using probabilities (**TODO for me:** check the math here; but perhaps we should add logits here
   instead of multiplying by probabilities (then it's easy to have a multiplicative parameter controlling the effect, and we can store logits
   instead of computing token probabilities, which is a relatively expensive thing to do for all tokens); **TODO: double-check this**).

4) these probabilities will have to be handled as additional parameters in their respective methods, like `generate` method of `GPT` class
   and `forward` methods of `Block` and `CausalSelfAttention` classes (folding them into an existing parameter is not feasible
   for a number of reasons)
