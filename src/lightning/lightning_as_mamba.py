# """
# PyTorch Lightning Module for AS-Mamba Training

# JamMaのLightning Moduleを拡張し、AS-Mamba特有の機能を追加:
# - Flow予測の可視化
# - Geometric lossの統合
# - 適応スパンの統計情報
# """

# from collections import defaultdict
# import pprint
# from loguru import logger
# from pathlib import Path

# import torch
# import numpy as np
# import pytorch_lightning as pl
# from matplotlib import pyplot as plt

# from src.jamma.as_mamba import AS_Mamba
# from src.jamma.backbone import CovNextV2_nano
# from src.jamma.utils.supervision import compute_supervision_fine, compute_supervision_coarse
# from src.losses.as_mamba_loss import AS_MambaLoss
# from src.optimizers import build_optimizer, build_scheduler
# from src.utils.metrics import (
#     compute_symmetrical_epipolar_errors,
#     compute_pose_errors,
#     aggregate_metrics_train_val, 
#     aggregate_metrics_test
# )
# from src.utils.comm import gather, all_gather
# from src.utils.misc import lower_config, flattenList
# from src.utils.profiler import PassThroughProfiler


# class PL_AS_Mamba(pl.LightningModule):
#     """
#     PyTorch Lightning Module for AS-Mamba
    
#     Extensions over JamMa:
#     1. AS-Mamba specific loss (flow + geometry)
#     2. Flow prediction visualization
#     3. Adaptive span statistics logging
#     """
    
#     def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
#         super().__init__()
        
#         # Configuration
#         self.config = config
#         _config = lower_config(self.config)
#         self.AS_MAMBA_cfg = lower_config(_config['as_mamba'])
#         self.profiler = profiler or PassThroughProfiler()
#         self.dump_dir = dump_dir
        
#         # Visualization settings
#         self.n_vals_plot = max(config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1)
#         self.viz_path = Path('visualization/as_mamba')
#         self.viz_path.mkdir(parents=True, exist_ok=True)
        
#         # Models
#         self.backbone = CovNextV2_nano()
#         self.matcher = AS_Mamba(config=_config['as_mamba'], profiler=profiler)
        
#         # AS-Mamba specific loss
#         self.loss = AS_MambaLoss(_config)
        
#         # Load pretrained weights
#         if pretrained_ckpt == 'official':
#             # TODO: AS-Mambaの公式重みをロード
#             logger.warning("AS-Mamba official weights not available yet")
#         elif pretrained_ckpt:
#             state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
#             self.load_state_dict(state_dict, strict=True)
#             logger.info(f"Loaded pretrained checkpoint: {pretrained_ckpt}")
        
#         # Testing
#         self.dump_dir = dump_dir
#         self.start_event = torch.cuda.Event(enable_timing=True)
#         self.end_event = torch.cuda.Event(enable_timing=True)
#         self.total_ms = 0
        
#         # Parameter count
#         n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         logger.info(f'AS-Mamba parameters: {n_parameters / 1e6:.2f}M')
    
#     def configure_optimizers(self):
#         optimizer = build_optimizer(self, self.config)
#         scheduler = build_scheduler(self.config, optimizer)
#         return [optimizer], [scheduler]
    
#     def optimizer_step(
#         self, epoch, batch_idx, optimizer, optimizer_idx,
#         optimizer_closure, on_tpu, using_native_amp, using_lbfgs
#     ):
#         # Learning rate warm up
#         warmup_step = self.config.TRAINER.WARMUP_STEP
#         if self.trainer.global_step < warmup_step:
#             if self.config.TRAINER.WARMUP_TYPE == 'linear':
#                 base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
#                 lr = base_lr + (self.trainer.global_step / warmup_step) * \
#                      abs(self.config.TRAINER.TRUE_LR - base_lr)
#                 for pg in optimizer.param_groups:
#                     pg['lr'] = lr
#             elif self.config.TRAINER.WARMUP_TYPE == 'constant':
#                 pass
#             else:
#                 raise ValueError(f'Unknown warm-up strategy: {self.config.TRAINER.WARMUP_TYPE}')
        
#         optimizer.step(closure=optimizer_closure)
#         optimizer.zero_grad()
    
#     def _train_inference(self, batch):
#         """Training forward pass"""
#         with self.profiler.profile("Compute coarse supervision"):
#             compute_supervision_coarse(batch, self.config)
        
#         with self.profiler.profile("Backbone"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 self.backbone(batch)
        
#         with self.profiler.profile("AS-Mamba Matcher"):
#             self.matcher(batch, mode='train')
        
#         with self.profiler.profile("Compute fine supervision"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 compute_supervision_fine(batch, self.config)
        
#         with self.profiler.profile("Compute losses"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 self.loss(batch)
    
#     def _val_inference(self, batch):
#         """Validation forward pass"""
#         with self.profiler.profile("Compute coarse supervision"):
#             compute_supervision_coarse(batch, self.config)
        
#         with self.profiler.profile("Backbone"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 self.backbone(batch)
        
#         with self.profiler.profile("AS-Mamba Matcher"):
#             self.matcher(batch, mode='val')
        
#         with self.profiler.profile("Compute fine supervision"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 compute_supervision_fine(batch, self.config)
        
#         with self.profiler.profile("Compute losses"):
#             with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#                 self.loss(batch)
    
#     def _compute_metrics(self, batch):
#         """Compute evaluation metrics"""
#         with self.profiler.profile("Compute metrics"):
#             compute_symmetrical_epipolar_errors(batch)
#             compute_pose_errors(batch, self.config)
            
#             rel_pair_names = list(zip(*batch['pair_names']))
#             bs = batch['imagec_0'].size(0)
            
#             metrics = {
#                 'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
#                 'epi_errs': [batch['epi_errs'][batch['m_bids'] == b].cpu().numpy() for b in range(bs)],
#                 'R_errs': batch['R_errs'],
#                 't_errs': batch['t_errs'],
#                 'inliers': batch['inliers']
#             }
            
#             return {'metrics': metrics}, rel_pair_names
    
#     def training_step(self, batch, batch_idx):
#         """Training step"""
#         self._train_inference(batch)
        
#         # Logging
#         if self.trainer.global_rank == 0 and self.global_step % self.trainer.log_every_n_steps == 0:
#             # Scalar logging
#             for key, value in batch['loss_scalars'].items():
#                 self.logger.experiment.add_scalar(f'train/{key}', value, self.global_step)
            
#             # AS-Mamba specific: Flow statistics
#             if 'as_mamba_flow' in batch:
#                 flow_magnitude = torch.norm(batch['as_mamba_flow'][..., :2], dim=-1).mean()
#                 self.logger.experiment.add_scalar('train/flow_magnitude', flow_magnitude, self.global_step)
            
#             # Adaptive span statistics
#             if 'adaptive_spans' in batch:
#                 spans_x, spans_y = batch['adaptive_spans']
#                 avg_span = (spans_x.float().mean() + spans_y.float().mean()) / 2
#                 self.logger.experiment.add_scalar('train/avg_adaptive_span', avg_span, self.global_step)
        
#         return {'loss': batch['loss']}
    
#     def training_epoch_end(self, outputs):
#         """End of training epoch"""
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         if self.trainer.global_rank == 0:
#             self.logger.experiment.add_scalar(
#                 'train/avg_loss_on_epoch', avg_loss, global_step=self.current_epoch
#             )
    
#     def validation_step(self, batch, batch_idx):
#         """Validation step"""
#         self._val_inference(batch)
#         ret_dict, _ = self._compute_metrics(batch)
        
#         # Dump results if requested
#         with self.profiler.profile("dump_results"):
#             if self.dump_dir is not None:
#                 bs = batch['imagec_0'].shape[0]
#                 dumps = []
#                 for b_id in range(bs):
#                     item = {}
#                     mask = batch['m_bids'] == b_id
#                     epi_errs = batch['epi_errs'][mask].cpu().numpy()
#                     correct_mask = epi_errs < 1e-4
#                     precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
#                     n_correct = np.sum(correct_mask)
                    
#                     item['precision'] = precision
#                     item['n_correct'] = n_correct
#                     item['runtime'] = batch['runtime']
                    
#                     for key in ['R_errs', 't_errs']:
#                         item[key] = batch[key][b_id][0]
                    
#                     # AS-Mamba specific metrics
#                     if 'as_mamba_flow' in batch:
#                         flow_mag = torch.norm(batch['as_mamba_flow'][b_id, ..., :2], dim=-1).mean().item()
#                         item['avg_flow_magnitude'] = flow_mag
                    
#                     if 'adaptive_spans' in batch:
#                         spans_x, spans_y = batch['adaptive_spans']
#                         avg_span = (spans_x[b_id].float().mean() + spans_y[b_id].float().mean()) / 2
#                         item['avg_adaptive_span'] = avg_span.item()
                    
#                     dumps.append(item)
                
#                 ret_dict['dumps'] = dumps
        
#         return ret_dict
    
#     def test_epoch_end(self, outputs):
#         """End of test epoch"""
#         # Gather metrics
#         _metrics = [o['metrics'] for o in outputs]
#         metrics = {
#             k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) 
#             for k in _metrics[0]
#         }
        
#         # Dump predictions
#         if self.dump_dir is not None:
#             Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
#             _dumps = flattenList([o['dumps'] for o in outputs])
#             dumps = flattenList(gather(_dumps))
#             logger.info(f'Results will be saved to: {self.dump_dir}')
        
#         # Rank 0 processing
#         if self.trainer.global_rank == 0:
#             print(self.profiler.summary())
            
#             val_metrics_4tb = aggregate_metrics_test(
#                 metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
#             )
#             logger.info('\n' + pprint.pformat(val_metrics_4tb))
#             logger.info(f'Average matching time: {self.total_ms / len(outputs):.2f} ms')
            
#             if self.dump_dir is not None:
#                 np.save(Path(self.dump_dir) / 'AS_MAMBA_pred_eval', dumps)
                
#                 # Save AS-Mamba specific statistics
#                 if dumps and 'avg_flow_magnitude' in dumps[0]:
#                     flow_stats = {
#                         'avg_flow_magnitude': np.mean([d['avg_flow_magnitude'] for d in dumps]),
#                         'avg_adaptive_span': np.mean([d.get('avg_adaptive_span', 0) for d in dumps])
#                     }
#                     np.save(Path(self.dump_dir) / 'AS_MAMBA_flow_stats', flow_stats)
#                     logger.info(f'AS-Mamba Flow Stats: {flow_stats}')._compute_metrics(batch)
        
#         return {
#             **ret_dict,
#             'loss_scalars': batch['loss_scalars'],
#         }
    
#     def validation_epoch_end(self, outputs):
#         """End of validation epoch"""
#         # Handle multiple validation sets
#         multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
#         multi_val_metrics = defaultdict(list)
        
#         for valset_idx, outputs in enumerate(multi_outputs):
#             cur_epoch = self.trainer.current_epoch
#             if not self.trainer.resume_from_checkpoint and self.trainer.running_sanity_check:
#                 cur_epoch = -1
            
#             # Loss scalars
#             _loss_scalars = [o['loss_scalars'] for o in outputs]
#             loss_scalars = {
#                 k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars])) 
#                 for k in _loss_scalars[0]
#             }
            
#             # Metrics
#             _metrics = [o['metrics'] for o in outputs]
#             metrics = {
#                 k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics]))) 
#                 for k in _metrics[0]
#             }
            
#             val_metrics_4tb = aggregate_metrics_train_val(
#                 metrics, self.config.TRAINER.EPI_ERR_THR, config=self.config
#             )
            
#             for thr in [5, 10, 20]:
#                 multi_val_metrics[f'auc@{thr}'].append(val_metrics_4tb[f'auc@{thr}'])
            
#             # TensorBoard logging (rank 0 only)
#             if self.trainer.global_rank == 0:
#                 for k, v in loss_scalars.items():
#                     mean_v = torch.stack(v).mean()
#                     self.logger.experiment.add_scalar(
#                         f'val_{valset_idx}/avg_{k}', mean_v, global_step=cur_epoch
#                     )
                
#                 for k, v in val_metrics_4tb.items():
#                     self.logger.experiment.add_scalar(
#                         f"metrics_{valset_idx}/{k}", v, global_step=cur_epoch
#                     )
        
#         # Log aggregated metrics
#         for thr in [5, 10, 20]:
#             self.log(f'auc@{thr}', torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}'])))
        
#         logger.info(f"Validation metrics: {val_metrics_4tb}")
    
#     def test_step(self, batch, batch_idx):
#         """Test step"""
#         with torch.autocast(enabled=self.config.AS_MAMBA.MP, device_type='cuda'):
#             self.start_event.record()
            
#             with self.profiler.profile("Backbone"):
#                 self.backbone(batch)
            
#             with self.profiler.profile("AS-Mamba Matcher"):
#                 self.matcher(batch, mode='test')
            
#             self.end_event.record()
#             torch.cuda.synchronize()
#             self.total_ms += self.start_event.elapsed_time(self.end_event)
#             batch['runtime'] = self.start_event.elapsed_time(self.end_event)
        
#         ret_dict, _ = self

"""
PyTorch Lightning Module for AS-Mamba Training and Evaluation

Key differences from JamMa:
1. AS-Mamba architecture with multiple blocks and flow prediction
2. ASMambaLoss with flow supervision
3. Flow and adaptive span supervision/visualization
4. Multi-scale feature processing

Author: Research Team
Date: 2025
"""

from collections import defaultdict
import pprint
from loguru import logger
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from src.jamma.backbone import CovNextV2_nano

# AS-Mamba specific imports
from src.jamma.as_mamba import ASMamba
from src.jamma.backbone import build_backbone
from src.jamma.utils.supervision import (
    compute_supervision_coarse,
    compute_supervision_fine, 
    compute_supervision_flow
)
from src.losses.as_mamba_loss import ASMambaLoss

# Shared utilities
from src.optimizers import build_optimizer, build_scheduler
from src.utils.metrics import (
    compute_f1,
    compute_symmetrical_epipolar_errors,
    compute_pose_errors,
    aggregate_metrics_train_val,
    aggregate_metrics_test
)
from src.utils.comm import gather, all_gather
from src.utils.misc import lower_config, flattenList
from src.utils.profiler import PassThroughProfiler
from src.utils.plotting import (
    make_matching_figures,
    make_flow_visualization,
    make_adaptive_span_visualization
)


class PL_ASMamba(pl.LightningModule):
    """
    PyTorch Lightning Module for AS-Mamba.
    
    This module handles:
    - Training with flow supervision
    - Validation with multi-scale metrics
    - Testing with runtime profiling
    - Visualization of matches, flow, and adaptive spans
    """
    
    def __init__(self, config, pretrained_ckpt=None, profiler=None, dump_dir=None):
        super().__init__()
        
        # Configuration
        self.config = config
        _config = lower_config(self.config)
        self.asmamba_cfg = lower_config(_config['asmamba'])
        
        # Profiler for performance analysis
        self.profiler = profiler or PassThroughProfiler()
        
        # Visualization settings
        self.n_vals_plot = max(
            config.TRAINER.N_VAL_PAIRS_TO_PLOT // config.TRAINER.WORLD_SIZE, 1
        )
        self.viz_path = Path('visualization/as_mamba')
        self.viz_path.mkdir(parents=True, exist_ok=True)
        
        # Model components
        self._build_model(_config)
        
        # Load pretrained weights if specified
        if pretrained_ckpt:
            self._load_pretrained(pretrained_ckpt)
        
        # Testing setup
        self.dump_dir = dump_dir
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.total_ms = 0
        
        # Log model size
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f'AS-Mamba trainable parameters: {n_parameters / 1e6:.2f}M')
        
    def _build_model(self, config):
        """
        Build AS-Mamba model components.
        
        Architecture:
        Input -> Backbone -> AS-Mamba Blocks -> Matching Head -> Output
                              ↓
                         Flow Predictor
                              ↓
                      Adaptive Span Computer
        """
        # Feature extraction backbone
        # Support multiple backbone types (ConvNeXt, ResNet, etc.)
        self.backbone = CovNextV2_nano()
        
        # AS-Mamba matcher with hierarchical blocks
        self.matcher = ASMamba(
            config=config['asmamba'],
            profiler=self.profiler
        )
        logger.info(f"AS-Mamba blocks: {config['asmamba']['num_blocks']}")
        
        # Loss function with flow supervision
        self.loss = ASMambaLoss(config)
        logger.info("Loss components: flow, coarse, fine, epipolar, multiscale")
        
    def _load_pretrained(self, pretrained_ckpt):
        """Load pretrained checkpoint with flexible loading."""
        if pretrained_ckpt == 'official':
            # TODO: Update URL when official weights are released
            logger.warning("Official AS-Mamba weights not yet available")
            return
        
        try:
            state_dict = torch.load(pretrained_ckpt, map_location='cpu')['state_dict']
            
            # Try strict loading first
            try:
                self.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded pretrained checkpoint: {pretrained_ckpt} (strict)")
            except RuntimeError as e:
                # Fall back to non-strict loading
                logger.warning(f"Strict loading failed: {e}")
                missing, unexpected = self.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded with missing keys: {len(missing)}, "
                          f"unexpected keys: {len(unexpected)}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {pretrained_ckpt}: {e}")
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = build_optimizer(self, self.config)
        scheduler = build_scheduler(self.config, optimizer)
        return [optimizer], [scheduler]
    
    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx,
        optimizer_closure, on_tpu, using_native_amp, using_lbfgs
    ):
        """
        Custom optimizer step with warmup.
        
        AS-Mamba benefits from gradual warmup due to:
        - Flow predictor initialization
        - Multi-scale feature alignment
        - Adaptive span learning stability
        """
        warmup_step = self.config.TRAINER.WARMUP_STEP
        
        if self.trainer.global_step < warmup_step:
            if self.config.TRAINER.WARMUP_TYPE == 'linear':
                base_lr = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
                lr = base_lr + \
                     (self.trainer.global_step / warmup_step) * \
                     abs(self.config.TRAINER.TRUE_LR - base_lr)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr
            elif self.config.TRAINER.WARMUP_TYPE == 'constant':
                for pg in optimizer.param_groups:
                    pg['lr'] = self.config.TRAINER.WARMUP_RATIO * self.config.TRAINER.TRUE_LR
            else:
                raise ValueError(f'Unknown warmup type: {self.config.TRAINER.WARMUP_TYPE}')
        
        # Update parameters
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
    
    def _trainval_inference(self, batch, mode='train'):
        """
        Shared inference logic for training and validation.
        
        Pipeline:
        1. Compute supervision (coarse, fine, flow)
        2. Extract features with backbone
        3. Process through AS-Mamba blocks
        4. Compute losses
        
        Args:
            batch: Data batch
            mode: 'train' or 'val'
        """
        # 1. Compute coarse-level supervision
        with self.profiler.profile("Compute coarse supervision"):
            compute_supervision_coarse(batch, self.config)
        
        # 2. CRITICAL: Compute flow supervision for AS-Mamba
        # This is unique to AS-Mamba - flow ground truth needed for adaptive spans
        with self.profiler.profile("Compute flow supervision"):
            compute_supervision_flow(batch, self.config)
        
        # 3. Feature extraction with backbone
        with self.profiler.profile("Backbone"):
            with torch.autocast(
                enabled=self.config.ASMAMBA.MP,
                device_type='cuda'
            ):
                self.backbone(batch)
        
        # 4. AS-Mamba processing
        # This includes: flow prediction -> adaptive spans -> hierarchical Mamba
        with self.profiler.profile("AS-Mamba Matcher"):
            with torch.autocast(
                enabled=self.config.ASMAMBA.MP,
                device_type='cuda'
            ):
                self.matcher(batch, mode=mode)
        
        # 5. Compute fine-level supervision
        with self.profiler.profile("Compute fine supervision"):
            with torch.autocast(
                enabled=self.config.ASMAMBA.MP,
                device_type='cuda'
            ):
                compute_supervision_fine(batch, self.config)
        
        # 6. Compute all losses (flow + matching + geometric)
        with self.profiler.profile("Compute losses"):
            with torch.autocast(
                enabled=self.config.ASMAMBA.MP,
                device_type='cuda'
            ):
                self.loss(batch)
    
    def _compute_metrics(self, batch, compute_f1_score=False):
        """
        Compute evaluation metrics.
        
        Args:
            batch: Data batch with predictions
            compute_f1_score: Whether to compute F1 (only for validation)
        
        Returns:
            Dictionary with metrics
        """
        with self.profiler.profile("Compute metrics"):
            # F1 score (precision/recall for matches)
            if compute_f1_score:
                compute_f1(batch)
            
            # Epipolar errors for each match
            compute_symmetrical_epipolar_errors(batch)
            
            # Pose estimation errors
            compute_pose_errors(batch, self.config)
            
            # Prepare metrics dictionary
            rel_pair_names = list(zip(*batch['pair_names']))
            bs = batch['imagec_0'].size(0)
            
            metrics = {
                'identifiers': [
                    '#'.join(rel_pair_names[b]) for b in range(bs)
                ],
                'epi_errs': [
                    batch['epi_errs'][batch['m_bids'] == b].cpu().numpy()
                    for b in range(bs)
                ],
                'R_errs': batch['R_errs'],
                't_errs': batch['t_errs'],
                'inliers': batch['inliers']
            }
            
            # Add F1 metrics if computed
            if compute_f1_score:
                metrics.update({
                    'precision': batch['precision'],
                    'recall': batch['recall'],
                    'f1_score': batch['f1_score']
                })
            
            # AS-Mamba specific: track flow prediction quality
            if 'flow_errors' in batch:
                metrics['flow_errors'] = batch['flow_errors']
            
            if 'adaptive_span_stats' in batch:
                metrics['span_stats'] = batch['adaptive_span_stats']
            
            ret_dict = {'metrics': metrics}
        
        return ret_dict, rel_pair_names
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        self._trainval_inference(batch, mode='train')
        
        # Logging
        if self.trainer.global_rank == 0 and \
           self.global_step % self.trainer.log_every_n_steps == 0:
            
            # Log total loss
            self.logger.experiment.add_scalar(
                'train/loss', batch['loss'], self.global_step
            )
            
            # Log individual loss components
            for loss_name, loss_value in batch['loss_scalars'].items():
                self.logger.experiment.add_scalar(
                    f'train/{loss_name}', loss_value, self.global_step
                )
            
            # AS-Mamba specific: log flow prediction quality
            if 'predict_flow' in batch:
                # Average flow magnitude across blocks
                flow_list = batch['predict_flow']
                for block_idx, (flow_0to1, flow_1to0) in enumerate(flow_list):
                    avg_flow_mag = torch.sqrt(
                        flow_0to1[..., :2].pow(2).sum(-1)
                    ).mean()
                    self.logger.experiment.add_scalar(
                        f'train/flow_magnitude_block{block_idx}',
                        avg_flow_mag,
                        self.global_step
                    )
                    
                    # Log uncertainty
                    avg_uncertainty = flow_0to1[..., 2:].mean()
                    self.logger.experiment.add_scalar(
                        f'train/flow_uncertainty_block{block_idx}',
                        avg_uncertainty,
                        self.global_step
                    )
            
            # Log adaptive span statistics
            if 'adaptive_spans' in batch:
                spans_x, spans_y = batch['adaptive_spans']
                self.logger.experiment.add_scalar(
                    'train/avg_span_x', spans_x.float().mean(), self.global_step
                )
                self.logger.experiment.add_scalar(
                    'train/avg_span_y', spans_y.float().mean(), self.global_step
                )
        
        return {'loss': batch['loss']}
    
    def training_epoch_end(self, outputs):
        """Aggregate training metrics at epoch end."""
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        if self.trainer.global_rank == 0:
            self.logger.experiment.add_scalar(
                'train/avg_loss_on_epoch',
                avg_loss,
                global_step=self.current_epoch
            )
    
    def validation_step(self, batch, batch_idx):
        """Validation step with visualization."""
        self._trainval_inference(batch, mode='val')
        
        # Compute metrics including F1
        ret_dict, rel_pair_names = self._compute_metrics(batch, compute_f1_score=True)
        
        # Add max matches for tracking
        ret_dict['metrics'] = {
            **ret_dict['metrics'],
            'max_matches': [batch.get('num_candidates_max', 0)]
        }
        
        # Visualization
        val_plot_interval = max(
            self.trainer.num_val_batches[0] // self.n_vals_plot, 1
        )
        figures = {self.config.TRAINER.PLOT_MODE: []}
        
        # Create visualizations for selected batches
        if batch_idx % val_plot_interval == 0 and self.trainer.global_rank == 0:
            # Standard matching visualization
            figures[self.config.TRAINER.PLOT_MODE] = [
                make_matching_figures(
                    batch,
                    self.config.TRAINER.PLOT_MODE,
                    return_fig=True
                )
            ]
            
            # AS-Mamba specific: Flow visualization
            if 'predict_flow' in batch and len(batch['predict_flow']) > 0:
                flow_fig = make_flow_visualization(
                    batch,
                    block_idx=-1,  # Last block
                    return_fig=True
                )
                figures['flow'] = [flow_fig]
            
            # AS-Mamba specific: Adaptive span visualization
            if 'adaptive_spans' in batch:
                span_fig = make_adaptive_span_visualization(
                    batch,
                    return_fig=True
                )
                figures['adaptive_spans'] = [span_fig]
        
        return {
            **ret_dict,
            'loss_scalars': batch['loss_scalars'],
            'figures': figures
        }
    
    def validation_epoch_end(self, outputs):
        """Aggregate validation metrics and log to tensorboard."""
        # Handle multiple validation sets
        multi_outputs = [outputs] if not isinstance(outputs[0], (list, tuple)) else outputs
        multi_val_metrics = defaultdict(list)
        
        for valset_idx, outputs in enumerate(multi_outputs):
            # Current epoch (handle sanity check)
            cur_epoch = self.trainer.current_epoch
            if not self.trainer.resume_from_checkpoint and \
               self.trainer.running_sanity_check:
                cur_epoch = -1
            
            # 1. Aggregate loss scalars
            _loss_scalars = [o['loss_scalars'] for o in outputs]
            loss_scalars = {
                k: flattenList(all_gather([_ls[k] for _ls in _loss_scalars]))
                for k in _loss_scalars[0]
            }
            
            # 2. Aggregate metrics
            _metrics = [o['metrics'] for o in outputs]
            metrics = {
                k: flattenList(all_gather(flattenList([_me[k] for _me in _metrics])))
                for k in _metrics[0]
            }
            
            # Compute aggregated metrics
            val_metrics_4tb = aggregate_metrics_train_val(
                metrics,
                self.config.TRAINER.EPI_ERR_THR,
                config=self.config
            )
            
            # Track AUC for model checkpointing
            for thr in [5, 10, 20]:
                multi_val_metrics[f'auc@{thr}'].append(
                    val_metrics_4tb[f'auc@{thr}']
                )
            
            # 3. Aggregate figures
            _figures = [o['figures'] for o in outputs]
            figures = {
                k: flattenList(gather(flattenList([_me[k] for _me in _figures])))
                for k in _figures[0]
            }
            
            # Log to tensorboard (rank 0 only)
            if self.trainer.global_rank == 0:
                # Loss scalars
                for k, v in loss_scalars.items():
                    mean_v = torch.stack(v).mean()
                    self.logger.experiment.add_scalar(
                        f'val_{valset_idx}/avg_{k}',
                        mean_v,
                        global_step=cur_epoch
                    )
                
                # Metrics
                for k, v in val_metrics_4tb.items():
                    self.logger.experiment.add_scalar(
                        f'metrics_{valset_idx}/{k}',
                        v,
                        global_step=cur_epoch
                    )
                
                # Figures
                for k, v in figures.items():
                    for plot_idx, fig in enumerate(v):
                        if fig is not None:
                            self.logger.experiment.add_figure(
                                f'val_match_{valset_idx}/{k}/pair-{plot_idx}',
                                fig,
                                cur_epoch,
                                close=True
                            )
                
                # Pretty print metrics
                logger.info(f'\nValidation set {valset_idx} metrics:')
                logger.info('\n' + pprint.pformat(val_metrics_4tb))
            
            plt.close('all')
        
        # Log aggregated AUC for checkpoint callback
        for thr in [5, 10, 20]:
            self.log(
                f'auc@{thr}',
                torch.tensor(np.mean(multi_val_metrics[f'auc@{thr}']))
            )
    
    def test_step(self, batch, batch_idx):
        """Test step with runtime profiling."""
        with torch.autocast(
            enabled=self.config.ASMAMBA.MP,
            device_type='cuda'
        ):
            # Time the inference
            self.start_event.record()
            
            # Feature extraction
            with self.profiler.profile("Backbone"):
                self.backbone(batch)
            
            # AS-Mamba matching
            with self.profiler.profile("AS-Mamba Matcher"):
                self.matcher(batch, mode='test')
            
            self.end_event.record()
            torch.cuda.synchronize()
            
            # Record runtime
            elapsed = self.start_event.elapsed_time(self.end_event)
            self.total_ms += elapsed
            batch['runtime'] = elapsed
        
        # Compute metrics
        ret_dict, rel_pair_names = self._compute_metrics(batch, compute_f1_score=False)
        
        # Optional: Save visualizations
        if self.config.TRAINER.get('SAVE_TEST_VIZ', False):
            path = str(self.viz_path / f'test_{batch_idx}')
            make_matching_figures(batch, 'confidence', path=f'{path}_confidence.png')
            
            # AS-Mamba specific visualizations
            if 'predict_flow' in batch:
                make_flow_visualization(batch, block_idx=-1, path=f'{path}_flow.png')
            if 'adaptive_spans' in batch:
                make_adaptive_span_visualization(batch, path=f'{path}_spans.png')
        
        # Dump results for further analysis
        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:
                bs = batch['imagec_0'].shape[0]
                dumps = []
                
                for b_id in range(bs):
                    item = {}
                    
                    # Epipolar errors and precision
                    mask = batch['m_bids'] == b_id
                    epi_errs = batch['epi_errs'][mask].cpu().numpy()
                    correct_mask = epi_errs < 1e-4
                    
                    item['precision'] = np.mean(correct_mask) if len(correct_mask) > 0 else 0
                    item['n_correct'] = np.sum(correct_mask)
                    item['runtime'] = batch['runtime']
                    
                    # Pose errors
                    for key in ['R_errs', 't_errs']:
                        item[key] = batch[key][b_id][0]
                    
                    # AS-Mamba specific: flow errors
                    if 'flow_errors' in batch and b_id < len(batch['flow_errors']):
                        item['flow_error'] = batch['flow_errors'][b_id]
                    
                    # Adaptive span statistics
                    if 'adaptive_span_stats' in batch and b_id < len(batch['adaptive_span_stats']):
                        item['span_stats'] = batch['adaptive_span_stats'][b_id]
                    
                    dumps.append(item)
                
                ret_dict['dumps'] = dumps
        
        return ret_dict
    
    def test_epoch_end(self, outputs):
        """Aggregate test results and save."""
        # Aggregate metrics
        _metrics = [o['metrics'] for o in outputs]
        metrics = {
            k: flattenList(gather(flattenList([_me[k] for _me in _metrics])))
            for k in _metrics[0]
        }
        
        # Save dumps if specified
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o['dumps'] for o in outputs])
            dumps = flattenList(gather(_dumps))
            logger.info(f'Results will be saved to: {self.dump_dir}')
        
        # Rank 0: compute and log final metrics
        if self.trainer.global_rank == 0:
            # Print profiler summary
            print(self.profiler.summary())
            
            # Aggregate metrics
            val_metrics_4tb = aggregate_metrics_test(
                metrics,
                self.config.TRAINER.EPI_ERR_THR,
                config=self.config
            )
            
            logger.info('\n' + '=' * 80)
            logger.info('AS-Mamba Test Results:')
            logger.info('=' * 80)
            logger.info('\n' + pprint.pformat(val_metrics_4tb))
            
            # Runtime statistics
            n_test_pairs = len(outputs) * batch['imagec_0'].shape[0]
            avg_runtime = self.total_ms / n_test_pairs if n_test_pairs > 0 else 0
            logger.info(f'\nAverage matching time: {avg_runtime:.2f} ms per pair')
            logger.info('=' * 80)
            
            # Save results
            if self.dump_dir is not None:
                save_path = Path(self.dump_dir) / 'AS_Mamba_pred_eval.npy'
                np.save(save_path, dumps)
                logger.info(f'Saved predictions to: {save_path}')