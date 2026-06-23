Papers
- Form Follows Function: Recursive Stem Model
- Recursive Inference Machines for Neural Reasoning
- LOOPFORMER: ELASTIC-DEPTH LOOPED TRANSFORMERS FOR LATENT REASONING VIA SHORTCUT MODULATION
- Your Latent Reasoning is Secretly Policy Improvement Operator
	- They fix $N_{sup} = 6$ and define a path of increasing SNR by masking solution tokens less. Basic idea is that after each recurse block it matches a time conditioned target that has more signal over time. Similar to diffusion.   
- Generative Recursive Reasoning
	- TODO: double check if the gains come from best of N
- Solve the Loop: Attractor Models for Language and Reasoning
- Looped transformers are better at learning learning algorithms 
- Training Large Language Models to Reason in a Continuous Latent Space (Coconut)
	-  Main idea: Take the last hidden embedding and put that in the tokens after the prompt, then we can train on sequences that include hidden embedding thoughts
- One Pass Is Not Enough: Recursive Latent Refinement for Generative Models
	-  IMLE. Why can't the network just learn a look up table and memorize training set? 
	- For the decoder (thing that maps noise to nearest training point), we use a TRM network
- HRM-Text
	- Prefix LM (question/answer training)
	- pass 
- Time-Modulated Looped Transformers
	- They add adaptive-instance normalization like time conditioning (like in DiT) and get slightly better performance for looped transfomers
- LoopFormer
- Iterative Reasoning through Energy Diffusion
	- pass 
- Tiny Recursive Language Diffusion Models
	- They use the TRM archiecture as the MDLM denoiser. This is the experiement I am thinking. Slightly better results than just MDLM or TRM, but on very small network
	- Interesting that they don't do a typical diffusion training where the noisy sample is forward generated according to the noise level; rather, they run inference in a loop to get $x_t$ based on previous model predictions (argmax decode). Means non parallelizable.
		- But interesting because they pass the latents (z,y) from previous timesteps in the training prediction! 
	- Training is 1 hour on H100 
- COEVOLUTIONARY CONTINUOUS DISCRETE DIFFUSION: MAKE YOUR DIFFUSION LANGUAGE MODEL A LATENT REASONER
	- Basic idea is that the discrete diffusion network takes in both discrete $x_t$ and continuous $z_t$ where $z_t$ is the evolution of a continuous embedding of the text($x_t, z_t$) -> $x_s$. And there is also a continuous diffusion network that takes in ($x_t, z_t$) -> $z_s$ . And then our network takes the product of two networks to get ($z_s, x_s$). 
- ELF: Embedded Language Flows
	- Need to confirm the cfg importance and cfg training, self conditioning training 
	- Can we formulate this as diffusion training vs flow 
	- They think that it will be difficult for the network to get t=1 prediction correct, so decoder needs to be robust to some noise   
- Fixed Point Masked Generative Modeling
	- pass 
- Looped Diffusion Language Models
- AdaFlow
	- Main idea is that the variance of the predicted velocity determines the step size
	- Can double check how they learn the variance of the flow 
- CGAR — Curriculum-Guided Adaptive Recursion
- TRM-Mamba — "Tiny Recursive Reasoning with Mamba-2 Attention Hybrid
- SE-RRM — Symbol-Equivariant Recurrent Reasoning Models
- EqR — Equilibrium Reasoners
- URM — Universal Reasoning Model
- RSM — Recursive Stem Model
- VARIATIONAL AUTOENCODING DISCRETE DIFFUSION WITH ENHANCED DIMENSIONAL CORRELATIONS MODELING
- LOOPHOLING DISCRETE DIFFUSION: DETERMINISTIC BYPASS OF THE SAMPLING WALL
	- Basically they take the final embedding of the logit network $x_0^\theta(\cdot)$ before it gets decoded into logits, and then pass that in to the next layer. 
	- To train, for some timestep t, they can't really get the embedding of the previous since they don't do full sampling, so instead they pass in an zero vector embedding context state, do a prediction, extract the embedding from that.... then use sg(embedding) as input to the network.
		- DOUBLE CHECK SELF CONDITIONING INTUITION
	- Also, we see that they do this self conditioning loss with probability p, otherwise they resort to typical diffusion loss

My reading hit list
-  Your Latent Reasoning is Secretly Policy Improvement Operator (DONE)
- Looped Diffusion Language Models (DONE)
- Iterative Reasoning through Energy Diffusion
	- Energy diffusion, but they don't learn multiple energy functions
	- I do like this idea of solving a more general problem before solving a more specific one
		- Like, mapping $y_s \to y_t$. If the model can also be trained on varying sizes of jumps this would be nice... so we can control at inference time how much to break down the problem (more difficult problems, more compute) 
- Planner and executor
- DUO (understand it again)
- DiffusionForcing

## Continuous Diffusion Language Model
Papers
- Continuous Diffusion for Categorical Data
- Flow Language Model
- Flow Map Language Model
- Discrete Flow Map 
- Think while you generate: discrete diffusion with planned denoising
	- They use the uniform discrete diffusion CTMC, and they have a separate planner network that predicts which tokens are noise, and a separate denoiser that gives the transition probability. They can use the planner to determine the expected time, by seeing how many noisy tokens it predicts still exists.

## Questions to resolve
- In TRM, for T recursive blocks, we apply gradient **only** to the final one. Is this important? Does HRM, stacked transformers, GRAM, looped transformers, do this? 
	- HRM does it on the last two steps total, across all recursive supervision steps. GRAM does this for final transition of each supervision step. Looped Transformer applies to the last B transformer layers.  
- In TRM, the updating network has no idea how good or done the input is. Whereas in diffusion we have a signal/noise ratio given by timestep. Does this matter?
- In TRM + GRAM, the upper network only updates given the current guess $y$ and the refined latent $z$. It doesn't use input, whereas the lower network refines the latent $z$ using $f(x,y,z) \to z$ . TRM shows that removing two networks into one and doing this (not passing x for upper) leads to slightly better generalization. Does this make sense that the upper network updates without the input?   
- Does GRAM do the expansion once? in the first supervision block? otherwise... does it keep branching? 
- FLM is a flow so uses deterministic ODE. This means that there are no stochastic transitions, which is bad for search. Same with TRM. 
	- Can we add stochasticity to FLM? Can we add a search process inside the FLM? Can we add stochasticity to TRM?  
- TRM cannot train with too long recursion block because it has to back propagate through all the steps, so OOM. However, there is nothing stopping it from having more recursion blocks (like first T-1 are without gradient). What happens when we increase the # of recursion blocks? What about increasing at inference time? Can we get inference time scaling? 
	- Probabilistic TRM does this, where they increase the depth rollout larger than training, small improvement. They also do best of N type by adding gaussian noise to the recursive latent
- Current thoughts with flow: say I am trying to map my input $x_t \to x_1$. Like there is no way for the model to be like, oops, my current state $x_t$ is bad, let me go back. It just predicts towards the data. I mean.... if the model is well trained enough, we expect $x_t$ to be on path of the distribution $p(x_t)$, but 
- Can we understand flow map distillation? Like if I have a model that can jump from $y_s \to y_t$ at varying jump sizes (think DUO distillation)... and if I can somehow keep the global reasoning state... like maybe I train in a similar sequential manner, like I can sample some step refinement $\delta$ and then I train, just like Your Latent Reasoning is Secretly Policy Improvement Operator, on these sequences $y_0 \to y_{\delta} \to y_{2 * \delta}...$ 
	- Double check but progressive distillation is this 

More
- [HRM-Text](https://github.com/sapientinc/HRM-Text)

$p(x|y) \propto p(y|x)p(x)$  and $p(y|x) \propto \frac{p(x|y)p(y)}{p(x)}$ so we have $p(x|y) \propto p(x|y)p(y)p(x) \implies \nabla p(x|y) = \nabla p(x|y) + \nabla p(x)$  


### Crash course on control
- Classifier guidance. Strong untrusted model generates, weak trusted model is guiding model. 

Relevant Papers
Review
- Classifier-Free Diffusion Guidance
- Simple Guidance Mechanisms for Discrete Diffusion Models
- FUDGE: Controlled Text Generation With Future Discriminators
- Diffusion-LM Improves Controllable Text Generation

Alignment
- Steering Language Models with Activation Engineering
	- pass 
- Representation Engineering: A Top-Down Approach
- https://www.emergentmind.com/topics/guidance-alignment
- PLUG AND PLAY LANGUAGE MODELS
	- We update the past history of key value embeddings via this gradient update to better satisfy some function 
	- pass 


First ablation experiment: Can we add scheduled stochasticity (decreasing over time, hopefully corresponding to better solutions) to our transition? 
- Motivation: HRM ablation by arc-agi shows that it's the out loop that really matters. Pre-training on more outer loops makes a difference. Like the idea of seeing lots of progressions of latents.    
- Dataset: Sudoku extreme, as in TRM repo. 

### Big picture ideas
- can we train diffusion/flow via RL? like few step generation... 
- how can I combine the planning with the denoising? like maybe denoised estimate (velocity direction) in conjuction with magnitude?
- 

## Week 2 First Ablation Experiment
Tldr; Flow maps are deterministic. We predict an expected $x_1$ data point to move towards. But this prediction might require reasoning. Can we get benefits (at the cost of extra compute) by changing our flow map denoiser to be a TRM architecture? Can we reduce the # of sampling steps (incur linear approximation error) and still preserve quality under a TRM denoiser?

Data: We use the same dataloader as in TRM github. Encoding: Like Hyperspherical Flows, we encode each sequence of length $2*81$, where the first board is board with clues (0 for empty cells) and second board is the board solution. Following SFM, we do not change the timestep to specify which tokens are clean/noisy, we just ensure that the first half does not get noised.   


First plot. Default flow map sudoku extreme success across NFEs. 
- Ablation: Add per token noise so model is aware of difference between hints vs to generate. This shouldn't make a big difference though. The model knows how to copy over given hints from the input to the solution.  

Second plot. Flow map with one TRM config. 

Third plot. Sweep across TRM # of layers, # of things. 



### Research Log/Progress
Jun 1. Forked TRM and trained. 
Jun 2. Non-negotiable: forward backward pass TRM with FLM. Then run the same training job. (DONE)

- We forked TRM and we run Sudoku extreme and train for 50k epochs. It takes 8 hours on my 5090. We see that the accuracy monotonically improves, it looks like it still going up, but get 78% test accuracy. 
- First go: 1024 sampling steps. Get like 6% total hit rate for 25k steps. Further steps don't perform better.
	- We have a problem with overfitting after 25k steps at 1k samples. At 300k steps I can get 100% accuracy on the training samples..... 
June 3. Non negotiable: sweep plot for both FLM and TRM. Run direct prediction ablation. (NOTDONE) 
- Using TRMv0 denoiser with FLM from Claude. 7M params. For 50k training steps, 1024 sampling steps as before, {'sudoku/to_predict_exact_match': 0.162109375, 'sudoku/to_predict_cell_acc': 0.5053932348937061, 'sudoku/given_cells_solution_acc': 0.9981286549707602, 'sudoku/n': 512}
	- This is an improvement! From 6% -> 16%
	- BUT: this is not depth matched. We do H_cycles: 3, L_cycles: 6, L_layers: 2 so approx 36 effective depth. Whereas FLM we chose 8 layers. 

Notes on differences between my current V0 TRM implementation and original:
- pass

Okay, had to take a break and think about the actual research direction. TRM already works really well. I think continuous diffusion language models don't work so well. But here's a general thing: diffusion really shouldn't be separating the denoising prediction and jump size $dt$; like the sampling scheduler is independent of the network's prediction. This is not good!

My plan: boost continuous diffusion language models performance. It will still be less than TRM. That is okay.

Experiment: Train a small head to predict step size, coupled with velocity prediction (via denoising). Start with single scalar prediction. (then maybe move on to per token)

Small ablation: can TRM handle full backprop? So we do the same thing config (which previously got us to 77% acc) but now we just drop the torch no grad, so this means that for all the recursion blocks we are doing back prop (18 cycles). So basically a looped transformer but loss only at end.   
- Get 46.08% at 156k steps.
	- Trains a lot slower and is worse  ![[Pasted image 20260604080934.png|190]]

IMPORTANT: I like the idea of training on toy data. Something that should take like <30 mins to train. We can test for mode collapse. I think it should be this conditional thing... because we can't do a TRM unconditionally. And we need to balance how much of the loss is learning the flow, how much the loss is figuring out the sampling trajectory. Maybe learn together. Maybe distillation. 
- intuition: learning the flow velocity is trajectory independent. it is a static snapshot of the current x_t input. learning how to iterate through the trajectory is this refinement idea. and the model will need to do future credit assignment... but maybe this will be OK with truncated BPTT. 

Todo:
- Put in a toy dataset (either sudoku as in SFM, or something that I can visually see) (DONE)

June 4. Non negotiable: create toy dataset (<30 mins training) and try continuous diffusion BPTT idea. 
One small toy conditional dataset: 14x14 subsampled and binarized MNIST. 10k train, 10k eval. 

Experiment: We first fix the number of steps (12 denoising steps). We train with a mixture of flow matching loss (continuous time) + BPTT (discrete time, uniform spaced with 12 steps). 
 - The idea is that we treat the denoising roll out process like one supervision step in TRM. We can score the final generated thing and then backprop (through the full roll out, or just the last few steps).    

Some ablations:
- Sweep the mixture parameter $\gamma$. What happens if we do only flow matching loss? Only BPTT?
	- WIP
- Sweep number of steps. 12 denoising steps vs 24 vs 36.
- Sweep the number of steps getting back prop. 

(optional) Experiment: We first fix the number of steps (12 denoising steps). We train with a mixture of flow matching loss (discrete time) + BPTT (discrete time, uniform spaced with 12 steps). 
 - The idea is that we treat the denoising roll out process like one supervision step in TRM. We can score the final generated thing and then backprop (through the full roll out, or just the last few steps).    


Future thing to think about once we run this experiment:
- We can think of the diffusion rollout as the latent recursion process in TRM. But if time doesn't necessarily always move forward, if time can go back, then we should be able to do T many iterations of this. Just like TRM.     


TRM ablations I'd like to try on a smaller dataset. 
- What happens when we make n_sup = 1? 

June 5. Non-negotiable. Run experiments to get validation of this (discrete time, i.e only train on a preset uniformly spaced timepoints) BPTT + direction idea.
- Result: We messed up. Conditional binary MNIST with the first half of rows given is too easy, every configuration can learn this. gamma=0 means flow match loss only, gamma=1 means BPTT loss only 
	- Interestingly, non-flow models, trained only using the BPTT, are still able to
![[Pasted image 20260605155523.png]]

**June 6**: Non-negotiable Write out and train a denoiser + step planner. And train! (DID NOT DO)
Still waiting on results... most of the sweep isn't useful! 

Experiment: Does doing looped transformer in addition to flow matching loss improve over just flow matching loss at low sampling step regime? This is assuming that we have discrete time fixed number of sampling time.  Sudoku hard. 6 steps. 64 batch size, T=B=6. Same model=small. 

- Run 1 (gamma=0.5), both losses: best performing out of 160k step. **57.90%** (1158/2000)
- Run 2 (gamma=0.0): only flow match: best performing out of 160k step. 33.50% (670/2000)
- Run 3 (gamma=1.0) only looped loss: best out of 160k. 1109/2000 (55.45%).
- Run 4 :Continuous flow match. (Batch size 64, 160k. We use the 120k checkpoint.)
	- 6 steps: Sudoku accuracy: 740/2000 (37.00%)
	- 12 steps: Sudoku accuracy: 757/2000 (37.85%)
	- 32 steps: 781/2000 (39.05%)
	- 64 steps: Sudoku accuracy: 786/2000 (39.30%)
	- 128 steps: Sudoku accuracy: 792/2000 (39.60%)
	- 256 steps: Sudoku accuracy: 797/2000 (39.85%)


We see a significant boost in performance with the looped loss vs flow model, slightly better than just looped!

Experiments that I would like to have run so I can get some idea of direction
- Can we use the diffusion transformer blocks (so no weird optimizer, no weird gradient clipping) but a TRM objective and can we get reasonably high success rate?
- How important is the fact that TRM can train by giving it like partially warm started solutions? Like what happens if we force it to reset to noise each time I run through all the recursion blocks  (Ablation experiment)

Experiment: Run the same but this time do generation in 12 steps. (Batch size 32, max 120001 step)
- Run 1: (gamma=0.0) T=12, B=12, B not used.
	- 503/2000 (25.15%) (Best out of all 120k steps)
- Run 2: (gamma=0.5) Both losses, T=12,B=12. 
	- 1433/2000 (71.65%) (Best out of 100k steps)
- Run 3: (gamma=1.0). T=12, B=12
	- 999/2000 (49.95%) (Best out of 120k)

**June 7**: Non-negotiable. Write and forward/backward pass the adaptive compute version.
Previously I've shown that when we add in a discrete time looped loss to discrete time flow map, we get better results. 

Ablation: (Note: i ran the wrong thing. So the ablation is incorrect. It uses discrete timesteps for flow loss. Need to re-run)
So can we get a similar effect even with the continuous time flow map? 


What happens if we formulate flow matching denoising as a looped transformer? (see Recurrent Transformer CSP) So for now suppose that we fix the number of denoising steps and we also fix to define the schedule. $\{t_i\}_{i=1}^{T_{max}}$ . Then we define $x_{t+1} = x_t + \frac{1}{T_{max}} \hat{v}(x_t, t)$. And we have the total loss
$L = \sum_{i=1}^{T_{max}} L(\hat{x_0}(x_t,t), x_0)$  where we backprop through at each timestep. Note that we can include in the loss the cross entropy between final generation and actual target. 
(TODO: double check how the other looped transformers do it). 

This is different than what I was doing: I was doing a mixture of just independent sampled x_t to learn to denoise (i.e different elements in batch, not interacting with each other) and then also doing this looping BPTT, where we pass the gradients through ONCE from the very end.

Results:
- T=B=6, discrete time, this is looped transformer but the network predicts a clean $x_1$ which is used to calculate a $x_{t+1} = x_t + \frac{1}{T_{max}} \hat{v}(x_t, t)$ . 180k steps, batch size 32, best performing checkpoint is 30k. 
	- accuracy: 0.3805
- T=B=12. Best performing checkpoint is 30k.  
	- accuracy: 56.65%




**June 8**: Need to fix the inflexiblity . f setting one time schedule. So either you try this looped flow model (READ loopedMDM) or you think about the adaptive schedule. The adaptive schedule I think matters.

Double checking: what does nll correspond to? So for gamma<1 it's the nll of the flow denoiser, else it's the nll of the final loop prediction. But: overconfident models can get really low mean nll even if on average it's more accurate. 

For example here is the sudoku val accuracy for different checkpoints. But here is the nll.. is diverges.... 
![[Pasted image 20260608201608.png|411]]
CKPT_Name      | accuracy
---------------+----------
6-10000.ckpt   | 20.50%
13-20000.ckpt  | 40.40%
19-30000.ckpt  | 46.70%
26-40000.ckpt  | 48.80%
33-50000.ckpt  | 49.35%
39-60000.ckpt  | 49.60%
46-70000.ckpt  | 49.90%
53-80000.ckpt  | 48.80%
59-90000.ckpt  | 47.30%
66-100000.ckpt | 47.15%
73-110000.ckpt | 47.00%
79-120000.ckpt | 46.90%
best_nll.ckpt  | 40.40%
last.ckpt      | 40.40%

Here is another graph
=== Summary ===
CKPT_Name     | accuracy
--------------+----------
13-10000.ckpt | 45.00%
26-20000.ckpt | 51.90%
39-30000.ckpt | 52.75%
53-40000.ckpt | 52.95%
66-50000.ckpt | 54.70%
79-60000.ckpt | 54.15%
best_nll.ckpt | 45.00%
last.ckpt     | 45.00%
![[Pasted image 20260608203323.png]]


June 9. Non-negotiable. V0 blog post public, ect. (DONE)

June 10: Chiller day. No deadlines. Can keep running things in the background. More idea day. Recall that Friday June 12 is Manifest. 

June 11: (Non-negotiable) Smoke test the noise conditioning idea. (NOT DONE) 
First: is this done? Quick summary
- Unlocking Prompt Infilling Capability for Diffusion Language Models
	- They fine-tune MDM models for the purpose of prompt infilling... i.e to generate prompts.  
- Cascaded Diffusion Models for High Fidelity Image Generation
	- Pretty sure that they just corrupt the pre-training conditioning data.... for image generation, this feels more like for robustness, they say that it is to learn on corrupted datasets.
		- My direction is different: our input is always clean (well, I guess I can argue OOD is noise, so this might be similar). I mean in some sense the idea is similar; but the conditioning input at inference is never noised. 
		- Double check: does the network know the noise level?
- BLOCKWISE SFT FOR DIFFUSION LANGUAGE MODELS: RECONCILING BIDIRECTIONAL ATTENTION AND AUTOREGRESSIVE DECODING
	- They show that corrupting the prefix (block wise decoding) leads worse performance
- Corrective Diffusion Language Models
	- pass 

We can position it as a generalization of classifier free guidance. (Ablation: only do on/off conditioning training.... I can see how well this works for CONDITIONAL generation)

(**NOTE**): from now on.... we are going to give the model in the forward pass the conditioning token locations. Thus... any experiments from before will not yield the same results


June 12: Smoke test the noise conditioning idea.
Okay: the simplest test: with probability $p_{condclean}$  we keep the noise time for conditioning tokens to $1$; else we keep the noise time to be same as the time of the other tokens.
- This is easy to do a test on how much it helps to have both the unconditional loss objective with the conditional loss objective. Since if we set p=1 then we recover the conditional only. p=0 is unconditional only.

The more complicated objective: with probability $p_{condclean}$ we set the noise time for conditioning tokens to some random $t'$, whereas the noise time for the rest is $t$. 

June 15: Non-negotiable: Visualization plots supporting the overfitting hypothesis. (DNF)

(**NOTE**: I made a mistake. I only changed the per token noise level tensor that I provided the model , not the actual noise on the conditioning tokens. Conditioning tokens were always preserved.) We are trained with full board prediction objective. What this means is that even if the model has conditioning tokens that are clean, if you lie and tell it that it's actually noisy.... this doesn't make any sense, it really should just ignore it and realize that the tokens whose simplex is one-hot always are clean. But it would mean that this added objective of varying the amount of "trust" in the conditioning leads to some improvement, at least in 10k case. 

SEB_N:100 acc=0.0000 (0/2000)  
SEB_N:1000 acc=0.0090 (18/2000)  
SEB_N:10000 acc=0.1645 (329/2000)  
  
SEB_N:10000_CTR_CP:0.01 acc=0.5265 (1053/2000)  
SEB_N:10000_CTR_CP:0.1 acc=0.7200 (1440/2000)  
SEB_N:10000_CTR_CP:0.2 acc=0.2065 (413/2000)  
SEB_N:10000_CTR_CP:0.5 acc=0.6930 (1386/2000)  
SEB_N:10000_CTR_CP:0.9 acc=0.4810 (962/2000)  
  
SEB_N:1000_CTR_CP:0.01 acc=0.0000 (0/2000)  
SEB_N:1000_CTR_CP:0.1 acc=0.0015 (3/2000)  
SEB_N:1000_CTR_CP:0.2 acc=0.0010 (2/2000)  
SEB_N:1000_CTR_CP:0.5 acc=0.0000 (0/2000)  
SEB_N:1000_CTR_CP:0.9 acc=0.0005 (1/2000)  
  
SEB_N:100_CTR_CP:0.01 acc=0.0000 (0/2000)  
SEB_N:100_CTR_CP:0.1 acc=0.0000 (0/2000)  
SEB_N:100_CTR_CP:0.2 acc=0.0000 (0/2000)  
SEB_N:100_CTR_CP:0.5 acc=0.0000 (0/2000)  
SEB_N:100_CTR_CP:0.9 acc=0.0000 (0/2000)

June 16. Visualization. Something running over night (either more sudoku, NQueens code) (DONE)

Result: Sudoku easy. 128 steps. 
- Sequence length: 81 (full 9×9 board)
- Training set: 10,000 puzzles (`train_subset_n: 10000`, `subset_seed: 0`)
- Validation set: 2,000 puzzles
- Difficulty: easy
- Loss region: board tokens only (`infill_loss_region: board`)
We train 160k steps, batch 32, FLM but we vary the probability of conditioning tokens remaining clean. 
|`conditioning_prob_clean`|Accuracy|Correct / Total|
- |0.2|**86.6%**|1732 / 2000|
- |0.5|74.75%|1495 / 2000|
- |1.0|56.05%|1121 / 2000| (default method)
- 0.0 |2.85% | 57/2000

Now for 1,000 puzzles train, 2,000 validation.  
- 0.2 **11.0%** 220 / 2000
- 0.5 9.9% 198 / 2000 
- 1.0 2.3% 46 / 2000
- 0.0 | 0.5% | 10/2000

Experiment: We mask out all the attention for non-conditioning tokens. In a healthy diffusion model, this should be catastrophic. But if it still is able to solve... then it means that it is literally just mapping x->y. 
- It failed. 0% solve rate.


(Later hypothesis: flow matching objective is curriculum learning... i.e being able to get x + partial y -> predict y. We can ablate and compare to only training on timestep t=0 for non-conditioning tokens, t=1 for conditioning tokens.)

June 17: What I'd like to see. NQueens test. Especially looking at coverage.
(I FOFO. Didn't get what I needed. Oh well. Be back by 8 ready to fire.)

1. **Equal split across k** — each `k` contributes the same number of `(input, solution)` pairs, capped at the scarce single-clue bucket. Verified exact: **558/558/558** per k for 8×8 (~1.7k train), **5536×3** for 10×10 (~16.6k train).
2. **85:15 split by unique input** — no clue board crosses train↔test; validation is likewise carved from the train side by unique input; the eval set now draws from the held-out 15% test side via the _same seeded partition_. Confirmed **train / valid / eval are fully disjoint by input**.


June 18: Run Mini GSM8k (lambda). This means we'll have sudoku +  nqueens + mini gsm come Sat.

- What I shipped: Nqueens training. Sudoku hard.

June 19: Run Mini GSM8k (lambda). 
- Double check our launch. Are we training what we want? Is the eval score what we want? (DONE)
- Nqueens eval (DONE)
- Make nqueens eval happen during training
- GSM8k. Did we copy and paste correctly? Do we know how long it will take on H100?
- Start thinking.... (think about ablation. Use your brain.)


NQueens 8: Training Dataset (1758). Validation Dataset (174). Eval dataset: (200 puzzles)
- For each eval puzzle, we generate 20 solutions, and consider the coverage 
Results for N=8

| `conditioning_prob_clean` | Accuracy   | Coverage   |
| ------------------------- | ---------- | ---------- |
| 0.2                       | **99.20%** | **94.65%** |
| 0.5                       | 99.03%     | 80.78%     |
| 1.0                       | 98.58%     | 75.25%     |

Run it again, but this time for 750 puzzles

Results (last checkpoint, 750 puzzles × 20 samples)

| `conditioning_prob_clean` | Accuracy   | Coverage   |
| ------------------------- | ---------- | ---------- |
| 0.2                       | **99.24%** | **97.31%** |
| 0.5                       | 98.75%     | 91.41%     |
| 1.0                       | 97.67%     | 88.07%     |

NQueens 10. (last checkpoint, 750 puzzles x 20 samples)
TODO: Double check the train/val #. Otherwise cap is 15k/2k.  

| `conditioning_prob_clean` | Accuracy   | Coverage   |
| ------------------------- | ---------- | ---------- |
| 0.2                       | **94.66%** | **80.01%** |
| 0.5                       | 57.42%     | 71.05%     |
| 1.0                       | 64.56%     | 64.03%     |


June 20:
Does single quote '\n' vs "\n" mean anything? tokenizer seperator
We'd like to be able to launch this. Problem is: it takes forever to save the damn dataset.

June 21: 
Non-negotiable. Enjoy the writing process. We should have a draft with Sudoku + Nqueens results. 

Quick thing: We re-launch sudoku hard across A10s or A100s. Doesn't matter. 1 hour. Just make sure that it only evaluates the last checkpoint.  
- DONE

Work on paper draft. 

Some natural questions:
- does it make sense to train on the PAD tokens? they make up so much of the loss...... 


















. 