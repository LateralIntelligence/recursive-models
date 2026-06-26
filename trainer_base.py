import itertools
import os
import random
import inspect

from dataclasses import dataclass

from tqdm import tqdm
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from torch.cuda.amp import autocast
import torch.distributed as dist
import dataloader
import metrics
import models
import utils
from omegaconf import ListConfig


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor

class LogLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = - (1 - self.eps) + t * 0
        assert alpha_t.shape == dalpha_t.shape
        return dalpha_t, alpha_t


class TrainerBase(L.LightningModule):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer,
            vocab_size=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, 'ignore_bos'):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
        if hasattr(self.config.algo, 'loss_type'):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization
        if self.config.algo.backbone == 'dit':
            self.backbone = models.dit.DIT(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'dimamba':
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id)
        elif self.config.algo.backbone == 'trm':
            self.backbone = models.trm.TRM(
                self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == 'hf_dit':
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True)
            
        self._pending_ema_state = None
        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # Noise Schedule
        self.noise = LogLinear()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size)

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
            self._get_parameters(),
            decay=self.config.training.ema)
        else:
            self.ema = None


        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self.target_tokens = None


    def _validate_configuration(self):
        assert self.config.algo.backbone in {'dit', 'hf_dit', 'trm'}
        if self.config.algo.backbone == 'trm':
            assert not self.config.algo.causal_attention, \
                "TRM backbone is bidirectional only"
        if self.config.algo.parameterization == 'ar':
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == 'none'

        if self.parameterization in {'score', 'mean'}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != 'score'

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(),
                               self.noise.parameters())

    def _eval_mode(self):
        if self.ema and not self.config.eval.disable_ema:
            print('Copying EMA parameters to model')
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        else:
            print('No EMA parameters')
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def load_state_dict(self, state_dict, strict=True):
        if any('_orig_mod' in k for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('._orig_mod.', '.')
                new_state_dict[new_key] = v
            state_dict = new_state_dict
        
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            filtered_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('teacher_model.'):
                    filtered_state_dict[k] = v
            state_dict = filtered_state_dict
        
        ret = super().load_state_dict(state_dict, strict=strict)
        
        if self.ema:
            ema_sd = getattr(self, "_pending_ema_state", None)
            ema_loaded = False

            if ema_sd is not None:
                try:
                    self.ema.load_state_dict(ema_sd)
                    current_params = list(self._get_parameters())

                    if len(self.ema.shadow_params) == len(current_params):
                        shapes_match = all(
                            s.shape == p.shape
                            for s, p in zip(self.ema.shadow_params, current_params)
                        )
                        if shapes_match:
                            ema_loaded = True
                        else:
                            print("[WARNING] EMA shape mismatch - will reinitialize from loaded weights")
                    else:
                        print("[WARNING] EMA count mismatch - will reinitialize from loaded weights")

                except Exception as e:
                    print(f"[WARNING] Failed to load EMA after weights load: {e}")

            if not ema_loaded:
                print("Initializing EMA from loaded model weights")
                import models.ema
                self.ema = models.ema.ExponentialMovingAverage(
                    list(self._get_parameters()),
                    decay=self.config.training.ema
                )

            self._pending_ema_state = None

        return ret

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self._pending_ema_state = checkpoint.get('ema', None)
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed']
        # is 1 iteration behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total'][
            'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['current'][
            'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps,
        # not the number of local steps, so we don't multiply with
        # self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
                'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler,
                   'state_dict'):
            sampler_state_dict = self.trainer.\
                train_dataloader.sampler.state_dict()
            checkpoint['sampler'][
                'random_state'] = sampler_state_dict.get(
                'random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Iterable datasets cannot handle fault tolerant samplers
        if hasattr(self.config.data, 'is_iterable') and self.config.data.is_iterable:
            return 
        
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                dl_sampler.load_state_dict({'epoch': self.fast_forward_epochs, 'counter': (self.fast_forward_batches * self.config.loader.batch_size)})
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=self.config.loader.num_workers>0))
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError
    
    def _sample_clean_mask(self, conditioning_mask):
        """Which tokens are presented to the model as clean (time 1.0).
        By default every conditioning token is clean. When
        ``algo.conditioning_time_random`` is on, during *training* each
        sequence has a probability "conditional_prob_clean" of the cond tokens remaining clean;
        otherwise it shares
        the sequence time, letting the model see partially-noised conditioning.
        Eval and sampling always keep conditioning clean so generation is
        consistent.

        Args:
            conditioning_mask: (B, L) bool, 1 where the token is clean context.
        Returns:
            (B, L) bool clean mask.
        """
        clean = conditioning_mask.bool()
        if self.training and getattr(self.config.algo, 'conditioning_time_random', False):
            p = float(getattr(self.config.algo, 'conditioning_prob_clean', 1.0))
            keep_clean = torch.rand(
                conditioning_mask.shape[0], device=conditioning_mask.device) < p
            clean = clean & keep_clean.unsqueeze(-1)
        return clean
    
    @staticmethod
    def _apply_clean_time(sigma, clean_mask):
        """Expand per-sequence time to per-token, pinning clean tokens to 1.0.

        Args:
            sigma: (B,) per-sequence time.
            clean_mask: (B, L) bool, 1 where the token's time is forced to 1.0.
        Returns:
            (B, L) per-token time.
        """
        B, L = clean_mask.shape
        sigma = sigma.reshape(B, 1).expand(B, L).clone()
        return torch.where(clean_mask, torch.ones_like(sigma), sigma)

    def forward(self, xt, sigma, sigma_prime=None, use_jvp_attn=False,
                conditioning_mask=None, per_token_time=False):
        if per_token_time:
            # ``sigma`` is already a per-token (B, L) time grid; feed it straight
            # to the backbone without collapsing to per-sequence or pinning any
            # token to clean (used by separate_conditioning_time, where the
            # conditioning tokens carry their own noise level).
            if not self.config.algo.time_conditioning:
                sigma = torch.zeros_like(sigma)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.backbone(xt, sigma, sigma_prime,
                                             use_jvp_attn=use_jvp_attn)
            return self._process_model_output(
                model_output=model_output, xt=xt, sigma=sigma)

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
        
        # Need to have our noise tensor match the conditioning token noise
        if conditioning_mask is not None:
            clean_mask = conditioning_mask.bool()
            sigma = self._apply_clean_time(sigma, clean_mask)
            if sigma_prime is not None:
                sigma_prime = self._apply_clean_time(sigma_prime, clean_mask)

        with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
            model_output = self.backbone(xt, sigma, sigma_prime, use_jvp_attn=use_jvp_attn)
        
        return self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):
        current_accumulation_step = (
            batch_idx % self.trainer.accumulate_grad_batches)

        losses = self._loss(batch['input_ids'],
                            batch['valid_tokens'],
                            current_accumulation_step,
                            train_mode=True,
                            xT=None if 'xT' not in batch else batch['xT'],
                            given_t=batch['given_t'] if 'given_t' in batch else None,
                            not_sampling_t=self.config.training.not_sampling_t,
                            conditioning_mask=batch.get('conditioning_mask', None)
                            )
        self.metrics.update_train(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='training/loss',
                 value=losses.loss.item(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return losses.loss

    def on_train_epoch_end(self):
        # NOTE:
        # Originally, this method re-logged validation NLL metrics at the end
        # of every *training* epoch by iterating over `self.metrics.valid_nlls`
        # and calling `.compute()` again.
        #
        # That extra logging turned out to be a non-trivial bottleneck and also
        # caused `val/*` metrics to appear much more frequently in WandB than
        # actual validation runs (which already log in `on_validation_epoch_end`).
        #
        # We therefore keep this hook but make it a no-op to avoid the
        # unnecessary per-train-epoch metric computation/logging. All
        # validation-related metrics are still logged from
        # `on_validation_epoch_end`, which is called whenever validation runs.
        return

    def _is_sudoku_dataset(self):
        """True when the active (validation) dataset is a sudoku variant.

        Covers ``sudoku``, ``sudoku-gen``, ``sudoku-extreme``, etc. - anything
        with ``sudoku`` in the configured dataset name.
        """
        name = str(getattr(self.config.data, 'valid', '') or '')
        return 'sudoku' in name.lower()
    
    def _sudoku_batch_correct(self, batch, num_steps):
        """Count exactly-solved puzzles in one eval batch.

        Two layouts, selected by ``config.algo.infill``:
          - infill: ``input_ids`` is the solution grid and
            ``batch['conditioning_mask']`` marks the given clues (held clean);
            the whole generated grid is compared against the solution.
          - prepend (default): ``input_ids`` is ``[puzzle | solution]``; condition
            on the puzzle half and compare the generated solution half.

        Returns ``(num_correct, total)`` over real (non-padded) rows.
        """
        if getattr(self.config.algo, 'infill', False):
            solution = batch['input_ids'].to(self.device)
            conditioning_mask = batch['conditioning_mask'].to(self.device).bool()
            real_rows = conditioning_mask.any(dim=-1)  # padded rows have no clues
            pred = self.conditional_generate_samples(
                solution, conditioning_mask, num_steps=num_steps)
            # TODO: decide if accuracy should be full board, including clues.
            # Clue cells are clamped clean, so a full-grid match == filling right.
            correct = (pred == solution).all(dim=1)
        else:
            input_ids = batch['input_ids'].to(self.device)
            valid_tokens = batch['valid_tokens'].to(self.device).bool()  # 1 over solution
            real_rows = valid_tokens.any(dim=-1)                         # drop padded rows
            conditioning_mask = torch.logical_not(valid_tokens)
            full_seq_pred = self.conditional_generate_samples(
                input_ids, conditioning_mask, num_steps=num_steps)

            B, S = input_ids.shape
            assert S % 2 == 0, 'expected [puzzle | solution] layout'
            half = S // 2
            gt = input_ids[:, half:]
            generated = full_seq_pred[:, half:]
            correct = (generated == gt).all(dim=1)

        num_correct = int((correct & real_rows).sum().item())
        total = int(real_rows.sum().item())
        return num_correct, total

    @torch.no_grad()
    def _sudoku_eval(self, use_val=True):
        """Exact-match solution accuracy on the sudoku validation/training set.

        Runs the model's conditional ODE sampler on each validation/training batch
        (puzzle tokens held fixed, solution tokens generated) and counts how
        many puzzles are solved exactly. Assumes the model is already in eval
        mode (set in ``on_validation_epoch_start``) and mirrors the standalone
        ``mode=sudoku_eval`` path in ``main.py``.

        Returns the aggregate accuracy (a float), or ``None`` if the model has
        no conditional sampler.
        """
        if not hasattr(self, 'conditional_generate_samples'):
            return None

        # discrete_loop_flm fixes the rollout grid via algo.num_timesteps;
        # plain flm has no such field, so fall back to the sampling steps.
        num_steps = self.config.algo.get('num_timesteps', None)
        if num_steps is None:
            num_steps = self.config.sampling.steps
        max_batches = self.config.eval.get('sudoku_max_batches', -1)
        if use_val:
            dls = self.trainer.val_dataloaders
        else:
            dls = self.trainer.train_dataloader
        if isinstance(dls, (list, tuple)):
            dls = dls[0]

        num_correct = total = num_batches = 0
        for batch in dls:
            if max_batches is not None and max_batches > 0 \
                    and num_batches >= max_batches:
                break
            batch_correct, batch_total = self._sudoku_batch_correct(
                batch, num_steps)
            num_correct += batch_correct
            total += batch_total
            num_batches += 1

        # Aggregate counts (not per-rank accuracies) across GPUs.
        counts = torch.tensor([num_correct, total],
                              device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        num_correct, total = counts[0].item(), counts[1].item()
        return num_correct / max(total, 1.0)
    
    def on_validation_epoch_start(self):
        self.metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        del batch_idx
        losses = self._loss(batch['input_ids'],
                            batch['valid_tokens'],
                            xT=None if 'xT' not in batch else batch['xT'],
                            conditioning_mask=batch.get('conditioning_mask', None),
                            )
        self.metrics.update_valid(losses.nlls, losses.prior_loss,
                                  losses.num_tokens)
        self.log(name='validation/loss',
                 value=losses.loss.item(),
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True)
        return losses.loss

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(name=k,  value=v.compute(), on_step=False,
                     on_epoch=True, sync_dist=True)
        if (self._is_sudoku_dataset()
                and not self.trainer.sanity_checking):
            val_accuracy = self._sudoku_eval(use_val=True)
            train_accuracy = self._sudoku_eval(use_val=False)
            if val_accuracy is not None:
                self.log(name='val/sudoku_accuracy', value=val_accuracy,
                         on_step=False, on_epoch=True, sync_dist=False)
            if train_accuracy is not None:
                self.log(name='train/sudoku_accuracy', value=train_accuracy,
                         on_step=False, on_epoch=True, sync_dist=False)
        
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            step_list = self.config.sampling.steps
            if isinstance(step_list, ListConfig):
                step_list = list(step_list)
            elif isinstance(step_list, int):
                step_list = [step_list]

            for num_steps in step_list:
                if hasattr(self.metrics, 'gen_ppl'):
                    self.metrics.gen_ppl.reset()
                if hasattr(self.metrics, 'sample_entropy'):
                    self.metrics.sample_entropy.reset()

                current_text_samples = []

                for _ in range(self.config.sampling.num_sample_batches):
                    samples = self.generate_samples(
                        num_samples=self.config.loader.eval_batch_size,
                        num_steps=num_steps
                    )

                    self.metrics.record_entropy(samples)

                    decoded_batch = self.tokenizer.batch_decode(samples)

                    if len(current_text_samples) < self.config.sampling.num_sample_log:
                        current_text_samples.extend(decoded_batch)

                    if self.config.eval.compute_generative_perplexity:
                        self.metrics.record_generative_perplexity(
                            decoded_batch, self.num_tokens, self.device)

                if self.config.eval.compute_generative_perplexity:
                    self.log(f'val/gen_ppl_T{num_steps}',
                            self.metrics.gen_ppl.compute(),
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True)
                    self.log(f'val/sample_entropy_T{num_steps}',
                            self.metrics.sample_entropy.compute(),
                            on_epoch=True,
                            on_step=False,
                            sync_dist=True)

                if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
                    log_samples = current_text_samples[:self.config.sampling.num_sample_log]

                    self.trainer.logger.log_table(
                        key=f'samples_T{num_steps}@global_step{self.global_step}',
                        columns=['Generated Samples'],
                        data=[[s] for s in log_samples]
                    )

        self._train_mode()

    def on_test_epoch_start(self):
        self._eval_mode()
        self.xTx0s = []

    def test_step(self, batch, batch_idx):
        xT = batch
        x0 = self.generate_samples(xT.shape[0], xT=xT.detach().clone())
        pair = torch.stack([xT, x0], dim=0)  # 2 B N
        self.xTx0s.append(pair)
        return 0.

    def on_test_epoch_end(self):
        # gather across all GPUs
        self.xTx0s = torch.cat(self.xTx0s, dim=1)  # 2 B N
        torch.distributed.barrier()

        # if multi gpu
        if torch.distributed.is_initialized():
            data_xTx0s_all = [torch.empty_like(self.xTx0s) for _ in range(
                torch.distributed.get_world_size())] if self.trainer.global_rank == 0 else None
            torch.distributed.gather(self.xTx0s,
                                     data_xTx0s_all,
                                     dst=0)

        if self.trainer.global_rank == 0:
            xTx0s = torch.cat(data_xTx0s_all, dim=1).cpu()[
                :, :self.config.sampling.num_reflow_samples]
            xTs, x0s = xTx0s[0], xTx0s[1]

            save_path = self.config.data.cache_dir
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            xTs = xTs.cpu().numpy()
            x0s = x0s.cpu().numpy()
            xT_path = os.path.join(save_path, 'xT.npy')
            x0_path = os.path.join(save_path, 'x0.npy')
            np.save(xT_path, xTs)
            np.save(x0_path, x0s)
            print('xT shape:', xTs.shape)
            print('x0 shape:', x0s.shape)
            print('xT saved to:', xT_path)
            print('x0 saved to:', x0_path)
        return
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                    self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {'scheduler': scheduler,
                          'interval': 'step',
                          'monitor': 'val/loss',
                          'name': 'trainer/lr'}
        return [optimizer], [scheduler_dict]

    def generate_samples(self, num_samples, num_steps, eps, xT, given_t):
        raise NotImplementedError

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self._eval_mode()

        step_list = self.config.sampling.steps
        if isinstance(step_list, ListConfig):
            step_list = list(step_list)
        elif isinstance(step_list, int):
            step_list = [step_list]
        all_samples = []
        for num_steps in step_list:
            batch_samples = self.generate_samples(
                num_samples=self.config.loader.eval_batch_size,
                num_steps=num_steps,
                eps=eps)
            # batch_samples is a tensor of shape (B, L)
            # Convert to list of tensors (one per sample in batch) for extend
            if isinstance(batch_samples, torch.Tensor):
                batch_samples = [batch_samples[i] for i in range(batch_samples.shape[0])]
            all_samples.extend(batch_samples)
        self._train_mode()
        return all_samples

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step=None, train_mode=False):
        raise NotImplementedError

    def _loss(self, x0, valid_tokens,
              current_accumulation_step=None,
              train_mode=False,
              xT=None, given_t=None, not_sampling_t=False, conditioning_mask=None):
        del conditioning_mask
        (input_tokens, output_tokens,
         valid_tokens) = self._process_model_input(
            x0, valid_tokens)
        loss = self.nll(input_tokens, output_tokens,
                        current_accumulation_step, train_mode)
            
        assert loss.ndim == 2
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens

        return Loss(loss=token_nll,
                    nlls=nlls,
                    prior_loss=0.0,
                    num_tokens=num_tokens)

