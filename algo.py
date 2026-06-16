'''
FLM code: https://github.com/david3684/flm/blob/main/algo.py
'''
import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import trainer_base
import utils
import math
import models
from torch.func import functional_call
from models.dit import modulate_fused
import functools
from entmax import entmax_bisect


class FLMBase(trainer_base.TrainerBase):
    """Base class for FLM/FMLM.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.diffusion_forcing = getattr(config.algo, 'diffusion_forcing', False)
        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        self._is_resuming = (
            config.checkpointing.resume_from_ckpt
            and config.checkpointing.resume_ckpt_path is not None
            and utils.fsspec_exists(config.checkpointing.resume_ckpt_path)
        )

    def _validate_configuration(self):
        pass

    def training_step(self, batch, batch_idx):
        return super().training_step(batch, batch_idx)

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens
    
    @staticmethod
    def _resolve_conditioning(valid_tokens, conditioning_mask):
        """Tokens to hold clean (clamp) during diffusion.

        When the batch supplies an explicit ``conditioning_mask`` (e.g. the
        in-place infilling format, where the clue cells to keep clean are NOT
        simply the complement of the loss mask), use it directly. Otherwise fall
        back to the default ``[puzzle | solution]`` convention where everything
        outside the loss region (``valid_tokens``) is conditioning/padding.
        """
        if conditioning_mask is not None:
            return conditioning_mask.bool()
        return torch.logical_not(valid_tokens)

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False,
              conditioning_mask=None):
        """Override to always dispatch to self.loss() for all FLM classes."""
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        
        conditioning_tokens = self._resolve_conditioning(valid_tokens, conditioning_mask)

        loss = self.loss(input_tokens, output_tokens, conditioning_tokens, 
                         current_accumulation_step, train_mode,
                         xT=xT, given_t=given_t,
                         not_sampling_t=not_sampling_t)
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        return trainer_base.Loss(loss=token_nll,
                                 nlls=nlls,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)

    def loss(self, x0, output_tokens,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        if t_max is None:
            t_max = self.t_max
        if accum_step is not None:
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]
        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            t = t[:batch_dim]
        return t

    def _tau_to_t(self, tau):
        """Convert t to reparameterized time tau."""
        return utils.alpha_to_gamma(tau, self.lut_a2g)

    def _t_to_tau(self, t):
        """Convert t to reparameterized time tau."""
        return utils.gamma_to_alpha(t, self.lut_g2a)

    def corrupt_continuous(self, x0, t, conditioning_mask=None):
        """Corrupt data x0 at time t using linear interpolation with Gaussian noise;
            if conditioning_mask supplied than keep those clean    

        Params:
            conditioning_mask: (torch.Tensor) (B,L)
        """
        t = t.unsqueeze(-1).unsqueeze(-1)
        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        if conditioning_mask is not None: # keep conditioning tokens clean
            x_t = torch.where(conditioning_mask.unsqueeze(-1), target_data, x_t)
        return x_t, target_data

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)

    def on_load_checkpoint(self, checkpoint):
        print("Resuming training from checkpoint...")
        self._is_resuming = True
        if 'state_dict' in checkpoint:
            checkpoint['state_dict'] = self._filter_checkpoint_state_dict(
                checkpoint['state_dict'])
        if self.config.mode == 'sample_eval':
            if getattr(self.backbone, 'learnable_loss_weighting', None) is not None:
                if not any(k.startswith('backbone.learnable_loss_weighting')
                           for k in checkpoint['state_dict'].keys()):
                    print("Learnable_loss_weighting not found in checkpoint. "
                          "Initializing from scratch for eval mode.")
                    for name, param in self.backbone.learnable_loss_weighting.named_parameters():
                        param_key = f'backbone.learnable_loss_weighting.{name}'
                        checkpoint['state_dict'][param_key] = param.data.clone()
        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def _filter_checkpoint_state_dict(self, state_dict):
        """Filter teacher keys and strip _orig_mod from checkpoint state_dict."""
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('teacher'):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        return new_state_dict

    def forward_no_softmax(self, xt, tau, tau_prime=None, **kwargs):
        tau = self._process_sigma(tau)
        if tau_prime is not None:
            tau_prime = self._process_sigma(tau_prime)
        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, tau, tau_prime, **kwargs)
        return model_output

    def _extract_ema_state_dict(self, model, checkpoint):
        """Extract EMA parameters from checkpoint into a state_dict for model."""
        ema_state = checkpoint.get('ema', None)
        if not ema_state:
            print("Warning: No EMA found, using regular state_dict")
            return {k.replace('backbone.', '').replace('._orig_mod.', ''): v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('backbone.')}

        new_sd = collections.OrderedDict()
        shadow_params = ema_state['shadow_params']
        param_names = [n for n, p in model.named_parameters()
                       if p.requires_grad]
        print(f"EMA shadow_params: {len(shadow_params)}, "
              f"Model param_names: {len(param_names)}")
        min_len = min(len(shadow_params), len(param_names))
        for name, val in zip(param_names[:min_len],
                             shadow_params[:min_len]):
            new_sd[name] = val
        for k, v in checkpoint['state_dict'].items():
            clean_k = k.replace('backbone.', '').replace('._orig_mod.', '')
            if (clean_k not in new_sd
                    and clean_k in [n for n, _ in model.named_parameters()]):
                new_sd[clean_k] = v
                print(f"Loaded missing param from state_dict: {clean_k}")
        if len(shadow_params) != len(param_names):
            print(f"Warning: EMA param count mismatch. "
                  f"Loaded {min_len}/{len(param_names)} from EMA, "
                  f"rest from state_dict")
        return new_sd

    def _load_teacher_model(self, path, use_plain_config=True):
        """Load a frozen teacher model from checkpoint.

        Args:
            path: Path to checkpoint file.
            use_plain_config: If True, temporarily disable double_temb and
                learnable_loss_weighting when building the teacher
                (to match EMA parameter shapes from a base model).
        """
        print(f"Loading teacher model from: {path}")
        if use_plain_config:
            saved = (self.config.algo.double_temb,
                     self.config.algo.learnable_loss_weighting)
            self.config.algo.double_temb = False
            self.config.algo.learnable_loss_weighting = False

        assert self.config.algo.backbone == 'dit', \
            "Only DIT backbone supported for teacher model"
        model = models.dit.DIT(self.config, vocab_size=self.vocab_size)

        if use_plain_config:
            (self.config.algo.double_temb,
             self.config.algo.learnable_loss_weighting) = saved

        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = self._extract_ema_state_dict(model, checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _copy_teacher_weights_to_student(self, teacher_dict):
        """Copy teacher weights to student backbone and zero-init sigma_map_prime."""
        with torch.no_grad():
            student_dict = self.backbone.state_dict()
            for name, param in teacher_dict.items():
                print(f"Copying parameter: {name}")
                if name in student_dict:
                    student_dict[name].copy_(param)
            if (hasattr(self.backbone, 'sigma_map_prime')
                    and self.backbone.sigma_map_prime is not None):
                for name, param in self.backbone.sigma_map_prime.named_parameters():
                    if 'mlp.2' in name:
                        param.zero_()
                        print(f"Zero initialized student sigma_map_prime: {name}")

    @staticmethod
    def _zero_init_module(module):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    @staticmethod
    def _random_init_module(module, std=0.02):
        for m in module.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(mean=0.0, std=std)
                if m.bias is not None:
                    m.bias.data.zero_()

class FLM(FLMBase):
    def loss(self, x0, output_tokens, conditioning_tokens=None,
             current_accumulation_step=None, train_mode=False,
             xT=None, given_t=None, not_sampling_t=False):
        '''
        conditioning_tokens: (torch.Tensor) Boolean mask where 1 for tokens which are not
            corrupted, else 0
        '''
        del given_t, not_sampling_t, output_tokens
        B = x0.shape[0]
        tau_t = self._sample_t_interval(B, current_accumulation_step,
                                    t_min=self.t_min, t_max=self.t_max)
        t = self._tau_to_t(tau_t)
        if self.diffusion_forcing and conditioning_tokens is not None:
            clean_mask = self._sample_clean_mask(conditioning_tokens)
        else:
            clean_mask = conditioning_tokens
        df_mask = clean_mask if self.diffusion_forcing else None
 
        x_t, target_data = self.corrupt_continuous(x0, t, clean_mask)
        f = self.forward(x_t, tau_t, conditioning_mask=df_mask) #condition on tau_t
        loss = -(target_data * f).sum(dim=-1)

        #self.log('loss', loss.mean(), prog_bar=True)
        if self.config.algo.learnable_loss_weighting is True:
            loss_weight = self.backbone.learnable_loss_weighting(tau_t)
            loss_weight = loss_weight.unsqueeze(-1)
            loss = torch.exp(-loss_weight) * loss + loss_weight
            #self.log('loss_weighted', loss.mean(), prog_bar=True)
        return loss

    @torch.no_grad()
    def conditional_generate_samples(self, puzzle_solution_input, conditioning_mask, num_steps=None, eps=1e-5):
        """Conditionally generate samples using Euler ODE solver.
        
        Params:
            puzzle_solution_input: (torch.Tensor) (B,L) sequences
            conditioning_mask: (torch.Tensor) Boolean where 1 represents conditioning (i.e puzzle) tokens         
        
        Returns:
            predicted solution tokens sequence, including clamped conditioning tokens (B,L) (torch.Tensor) 
        """
        if num_steps is None:
            num_steps = self.config.sampling.steps
        assert len(puzzle_solution_input.shape) == 2
        B,L = puzzle_solution_input.shape 
        V = self.vocab_size
        device = self.device

        puzzle_solution_onehot = F.one_hot(puzzle_solution_input, num_classes=V)

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        #z = torch.randn((B, L, V), device=device, dtype=self.dtype)
        z, _ = self.corrupt_continuous(puzzle_solution_input, t=torch.Tensor([0]).to(device), conditioning_mask=conditioning_mask)
        z = z.to(device).to(self.dtype)

        for i in range(num_steps):
            tau_t_curr = tau_vals[i]
            tau_t_next = tau_vals[i + 1]
            tau_t_in = tau_t_curr.expand(B)
            t_in = self._tau_to_t(tau_t_in)
            dt = self._tau_to_t(tau_t_next.expand(B)) - t_in
            df_mask = conditioning_mask if self.diffusion_forcing else None
            x_1_pred = self.forward(z, tau_t_in, conditioning_mask=df_mask)
            #x_1_pred = self.forward(z, tau_t_in)
            x_1_pred_probs = x_1_pred.exp()

            if i == num_steps - 1:
                z = x_1_pred_probs
                # clamp clean conditioning values 
                z = torch.where(conditioning_mask.unsqueeze(-1), puzzle_solution_onehot, z) 
                break

            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
            z = z + dt.view(-1, 1, 1) * v
            # clamp clean conditioning values 
            z = torch.where(conditioning_mask.unsqueeze(-1), puzzle_solution_onehot, z)

        return z.argmax(dim=-1)
    
    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples using Euler ODE solver."""
        if num_steps is None:
            num_steps = self.config.sampling.steps
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        tau_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
        z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)

        for i in range(num_steps):
            tau_t_curr = tau_vals[i]
            tau_t_next = tau_vals[i + 1]
            tau_t_in = tau_t_curr.expand(B)
            t_in = self._tau_to_t(tau_t_in)
            dt = self._tau_to_t(tau_t_next.expand(B)) - t_in
            x_1_pred = self.forward(z, tau_t_in)
            x_1_pred_probs = x_1_pred.exp()

            if i == num_steps - 1:
                z = x_1_pred_probs
                break

            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
            z = z + dt.view(-1, 1, 1) * v

        return z.argmax(dim=-1)

class DiscreteRecurrentFLM(FLM):
    """FLM trained with a looped (unrolled-ODE) objective.
    We consider fixed uniformly spaced timesteps; and unrolls using the first 
    order Euler approximation during inference. At each step the model predicts 
    E[x_1], which we add to a total cross entropy objective. Moreover we can define  
    the velocity ``v = (E[x_1] - x_t) / (1 - t)`` and take ``x_t <- x_t + v*dt``,
    clamping conditioning tokens to their clean values. We also include the final step's logits
    to be scored against the true tokens with a cross-entropy loss.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.num_timesteps = config.algo.num_timesteps
        self.backprop_steps = config.algo.backprop_steps
        self.discrete_denoiser_time = config.algo.get('discrete_denoiser_time', False)
        assert self.discrete_denoiser_time, "Current algo can only take fixed discrete timesteps"

    def _loop_loss(self, x0, conditioning_tokens, eps=1e-5):
        """Per-token cross-entropy of the looped-rollout's final prediction.

        Params:
            x0: (B, L) clean target token ids.
            conditioning_tokens: (B, L) bool mask, 1 where tokens are kept clean
                (conditioning / padding), 0 where they must be generated.

        Returns:
            (B, L) per-token cross-entropy of the final x_1 logits vs. x0.
        """
        B, L = x0.shape
        device = self.device
        N = self.num_timesteps
        N_backprop = self.backprop_steps
        assert N_backprop >= 1

        target_data = F.one_hot(x0, self.vocab_size).float()  # (B, L, V)

        # Discrete, uniformly-spaced timesteps (in tau space, matching the
        # sampler). t is derived from tau exactly as in conditional_generate_samples.
        tau_vals = torch.linspace(0.0, 1.0, N + 1, device=device)

        if self.diffusion_forcing and conditioning_tokens is not None:
            clean_mask = self._sample_clean_mask(conditioning_tokens)
        else:
            clean_mask = conditioning_tokens
        df_mask = clean_mask if self.diffusion_forcing else None

        # x_t at t=0 is pure noise, with conditioning tokens noised based on the clean_mask
        t0 = torch.zeros(B, device=device)
        z, _ = self.corrupt_continuous(x0, t0, clean_mask)  # (B, L, V), float32

        final_log_probs = None
        sum_loss_across_time = 0.0 
        for k in range(N):
            tau_t_in = tau_vals[k].expand(B)
            t_in = self._tau_to_t(tau_t_in)
            f = self.forward(z, tau_t_in, conditioning_mask=df_mask)  # log-probs over vocab, (B, L, V)
            
            #f = self.forward(z, tau_t_in)  # log-probs over vocab, (B, L, V)
            loss = -(target_data * f).sum(dim=-1)
            if k >= (N-N_backprop): #only have gradient calculation for last N_backprop
                sum_loss_across_time += loss

            if k == N - 1:
                # Final x_1 prediction: score its logits against the true tokens.
                final_log_probs = f
                break

            x_1_pred_probs = f.exp()
            dt = self._tau_to_t(tau_vals[k + 1].expand(B)) - t_in
            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
            z = z + dt.view(-1, 1, 1) * v
            # Keep conditioning tokens clean at every step.
            z = torch.where(clean_mask.unsqueeze(-1), target_data, z)
            if k < (N-N_backprop): 
                #no gradient passing through the first layers not included in last N_backprop
                # (TODO: you can ablate this) 
                z = z.detach()


        final_loop_ce = -(target_data * final_log_probs).sum(dim=-1)  # (B, L)
        return sum_loss_across_time, final_loop_ce

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False, conditioning_mask=None):
        (input_tokens, _output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        # Tokens kept clean (conditioning + padding); the complement is in-loss.
        conditioning_tokens = self._resolve_conditioning(valid_tokens, conditioning_mask)

        num_tokens = valid_tokens.sum()
        loop_pt, final_loop_ce = self._loop_loss(input_tokens, conditioning_tokens)
        loop_nll = (loop_pt * valid_tokens).sum()
        loop_loss = loop_nll / num_tokens
        final_loop_nll = (final_loop_ce * valid_tokens).sum()
       
        assert loop_pt.ndim == 2       
        self.log('loss_total', loop_loss, prog_bar=True)

        # Report the final denoiser NLL for perplexity-style metrics (comparable to FLM).
        return trainer_base.Loss(loss=loop_loss,
                                 nlls=final_loop_nll,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)

class DiscreteLoopFLM(FLM):
    """FLM variant trained with an additional *looped* (unrolled-ODE) objective.

    On top of the usual per-token denoiser loss (random t, inherited from
    ``FLM.loss``), this class unrolls a fixed number of uniformly-spaced ODE
    steps -- the same first-order Euler integrator used at sampling time -- and
    backpropagates through the whole rollout. The rollout starts from pure noise
    at t=0, and at every step the model ``f`` predicts E[x_1], from which we form
    the velocity ``v = (E[x_1] - x_t) / (1 - t)`` and take ``x_t <- x_t + v*dt``,
    clamping conditioning tokens to their clean values. The final step's logits
    are scored against the true tokens with a cross-entropy loss.

    The total loss is a linear (convex) combination controlled by ``gamma``:

        total = (1 - gamma) * denoiser_loss + gamma * loop_loss

    so ``gamma=0`` recovers the plain FLM denoiser objective and ``gamma=1`` uses
    only the looped objective.

    NOTE: This is not how looped transformers implement the loss. This only BPTT from the final output. 
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.num_timesteps = config.algo.num_timesteps
        self.backprop_steps = config.algo.backprop_steps
        self.gamma = config.algo.gamma
        self.discrete_denoiser_time = config.algo.get('discrete_denoiser_time', False)

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        """Sample the denoiser-loss time.

        Default: continuous random tau (FLMBase behaviour). When
        ``discrete_denoiser_time`` is set, snap the (antithetic) continuous
        sample onto the uniform grid ``linspace(0, 1, num_timesteps + 1)`` so the
        denoiser loss trains on exactly the timesteps used by the looped rollout.
        """
        tau = super()._sample_t_interval(n, accum_step, t_min=t_min, t_max=t_max)
        if not self.discrete_denoiser_time:
            return tau
        lo = self.t_min if t_min is None else t_min
        hi = self.t_max if t_max is None else t_max
        # Map continuous uniform sample -> uniform index over the N+1 grid points.
        u = ((tau - lo) / (hi - lo)).clamp(0.0, 1.0)
        N = self.num_timesteps
        idx = torch.clamp((u * (N + 1)).long(), 0, N)
        grid = torch.linspace(0.0, 1.0, N + 1, device=tau.device)
        return grid[idx]

    def _loop_loss(self, x0, conditioning_tokens, eps=1e-5):
        """Per-token cross-entropy of the looped-rollout's final prediction.

        Params:
            x0: (B, L) clean target token ids.
            conditioning_tokens: (B, L) bool mask, 1 where tokens are kept clean
                (conditioning / padding), 0 where they must be generated.

        Returns:
            (B, L) per-token cross-entropy of the final x_1 logits vs. x0.
        """
        B, L = x0.shape
        device = self.device
        N = self.num_timesteps
        N_backprop = self.backprop_steps

        target_data = F.one_hot(x0, self.vocab_size).float()  # (B, L, V)

        # Discrete, uniformly-spaced timesteps (in tau space, matching the
        # sampler). t is derived from tau exactly as in conditional_generate_samples.
        tau_vals = torch.linspace(0.0, 1.0, N + 1, device=device)

        if self.diffusion_forcing and conditioning_tokens is not None:
            clean_mask = self._sample_clean_mask(conditioning_tokens)
        else:
            clean_mask = conditioning_tokens
        df_mask = clean_mask if self.diffusion_forcing else None


        # x_t at t=0 is pure noise, with conditioning tokens clamped to clean.
        t0 = torch.zeros(B, device=device)
        z, _ = self.corrupt_continuous(x0, t0, clean_mask)  # (B, L, V), float32

        final_log_probs = None
        for k in range(N):
            tau_t_in = tau_vals[k].expand(B)
            t_in = self._tau_to_t(tau_t_in)
            f = self.forward(z, tau_t_in, conditioning_mask=df_mask)  # log-probs over vocab, (B, L, V)

            if k == N - 1:
                # Final x_1 prediction: score its logits against the true tokens.
                final_log_probs = f
                break

            x_1_pred_probs = f.exp()
            dt = self._tau_to_t(tau_vals[k + 1].expand(B)) - t_in
            v = (x_1_pred_probs - z) / (1.0 - t_in.view(-1, 1, 1) + eps)
            z = z + dt.view(-1, 1, 1) * v
            # Keep conditioning tokens clean at every step.
            z = torch.where(clean_mask.unsqueeze(-1), target_data, z)
            if k < (N-N_backprop): 
                #no gradient 
                z = z.detach()

        loop_ce = -(target_data * final_log_probs).sum(dim=-1)  # (B, L)
        return loop_ce

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False,
              conditioning_mask=None):
        (input_tokens, _output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        # Tokens kept clean (conditioning + padding); the complement is in-loss.
        conditioning_tokens = self._resolve_conditioning(valid_tokens, conditioning_mask)

        total_loss = 0
        num_tokens = valid_tokens.sum()
        if self.gamma < 1:
            # 1) Standard denoiser loss at a random t (inherited FLM.loss) -> (B, L).
            denoiser_pt = self.loss(input_tokens, _output_tokens, conditioning_tokens,
                                    current_accumulation_step, train_mode,
                                    xT=xT, given_t=given_t,
                                    not_sampling_t=not_sampling_t)
            denoiser_nll = (denoiser_pt * valid_tokens).sum()
            denoiser_loss = denoiser_nll / num_tokens
            total_loss += (1.0 - self.gamma)*denoiser_loss
            assert denoiser_pt.ndim == 2
            self.log('loss_denoiser', denoiser_loss, prog_bar=True)
            
        if self.gamma > 0:
            # 2) Looped-rollout cross-entropy of the final prediction -> (B, L).
            loop_pt = self._loop_loss(input_tokens, conditioning_tokens)
            loop_nll = (loop_pt * valid_tokens).sum()
            loop_loss = loop_nll / num_tokens
            total_loss += self.gamma*loop_loss 
            assert loop_pt.ndim == 2
            self.log('loss_loop', loop_loss, prog_bar=True)

       
        self.log('loss_total', total_loss, prog_bar=True)

        # Report the denoiser NLL for perplexity-style metrics (comparable to FLM).
        return trainer_base.Loss(loss=total_loss,
                                 nlls=denoiser_nll if self.gamma < 1 else loop_nll,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)
    
class CondUncondLoopFLM(DiscreteLoopFLM):
    """Loop-FLM variant that stochastically mixes an *unconditional* denoiser
    objective with a *conditional* looped-rollout objective.
    IMPORTANT: Read the description of the unconditional training. This is NOT
    learning the joint distribution p(x,y). The loss is only on the predicted solution
    so it learns D(y | noisy(x)) -> y.  

    For each batch we flip a coin:

      * with probability ``prob_unconditional`` we train the plain FLM denoiser
        loss with **no conditioning** -- the conditioning (board) tokens are no
        longer kept clean, so the whole sequence (including the board) is noised
        and the model must denoise the solution labels unconditionally. The loss
        is still scored only on the valid (solution) tokens, so this learns the
        unconditional marginal over solutions.

      * with probability ``1 - prob_unconditional`` we train the conditional
        looped-rollout cross-entropy (``_loop_loss``), where ``input_tokens`` is
        the usual ``board | clean solution`` sequence and the board tokens are
        kept clean throughout the unrolled ODE.

    ``prob_unconditional=0`` recovers a purely conditional looped objective and
    ``prob_unconditional=1`` recovers a purely unconditional denoiser objective.
    """

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.prob_unconditional = float(config.algo.prob_unconditional)

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False,
              conditioning_mask=None):
        (input_tokens, _output_tokens,
         valid_tokens) = self._process_model_input(x0, valid_tokens)
        # Tokens kept clean (conditioning + padding); the complement is in-loss.
        conditioning_tokens = self._resolve_conditioning(valid_tokens, conditioning_mask)
        num_tokens = valid_tokens.sum()

        if torch.rand(1).item() < self.prob_unconditional:
            # Unconditional denoiser branch: drop the conditioning so the clean
            # solution labels (and the board) are all noised, then score only the
            # valid (solution) tokens.
            # (TODO: need to provide a conditioning token indicator such that I can generate unconditionally and not have it condition on the first half
            no_conditioning = torch.zeros_like(conditioning_tokens)
            denoiser_pt = self.loss(input_tokens, _output_tokens, no_conditioning,
                                    current_accumulation_step, train_mode,
                                    xT=xT, given_t=given_t,
                                    not_sampling_t=not_sampling_t)
            assert denoiser_pt.ndim == 2
            nll = (denoiser_pt*valid_tokens).sum()
            loss = nll / num_tokens
            self.log('loss_uncond', loss, prog_bar=True)
        else:
            # Conditional looped-rollout branch (board kept clean throughout).
            loop_pt = self._loop_loss(input_tokens, conditioning_tokens)
            assert loop_pt.ndim == 2
            nll = (loop_pt * valid_tokens).sum()
            loss = nll / num_tokens
            self.log('loss_loop', loss, prog_bar=True)

        self.log('loss_either', loss, prog_bar=True)

        return trainer_base.Loss(loss=loss,
                                 nlls=nll,
                                 prior_loss=0.0,
                                 num_tokens=num_tokens)