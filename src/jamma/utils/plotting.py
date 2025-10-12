"""
Visualization utilities for AS-Mamba.

Provides visualizations for:
1. Flow predictions with uncertainty
2. Adaptive span distributions
3. Multi-scale feature analysis
4. Matching quality assessment

Author: Research Team
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pathlib import Path


def make_flow_visualization(data, block_idx=-1, path=None, return_fig=False):
    """
    Visualize flow predictions with uncertainty.
    
    Creates a comprehensive flow visualization showing:
    1. Flow field as arrows
    2. Flow magnitude heatmap
    3. Uncertainty heatmap
    4. Ground truth comparison (if available)
    
    Args:
        data: Batch dictionary containing:
            - predict_flow: List of flow predictions per block
            - flow_gt_0to1, flow_gt_1to0: Ground truth flows (optional)
            - image0, image1: Input images
        block_idx: Which AS-Mamba block to visualize (-1 for last)
        path: Save path (if None, returns figure)
        return_fig: Whether to return figure object
    
    Returns:
        Figure object if return_fig=True
    """
    if 'predict_flow' not in data or len(data['predict_flow']) == 0:
        return None
    
    # Extract flow predictions
    flow_list = data['predict_flow']
    flow_0to1, flow_1to0 = flow_list[block_idx]
    
    # Take first sample in batch
    flow = flow_0to1[0, 0].detach().cpu().numpy()  # (H, W, 4)
    
    H, W = flow.shape[:2]
    flow_x, flow_y = flow[..., 0], flow[..., 1]
    uncertainty_x, uncertainty_y = flow[..., 2], flow[..., 3]
    
    # Compute flow magnitude
    flow_mag = np.sqrt(flow_x**2 + flow_y**2)
    uncertainty_avg = (np.exp(uncertainty_x) + np.exp(uncertainty_y)) / 2.0
    
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Flow field visualization
    ax1 = plt.subplot(2, 3, 1)
    _plot_flow_field(ax1, flow_x, flow_y, title='Flow Field (0→1)', stride=8)
    
    # 2. Flow magnitude heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(flow_mag, cmap='jet', interpolation='nearest')
    ax2.set_title(f'Flow Magnitude (max: {flow_mag.max():.2f} px)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Uncertainty heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(uncertainty_avg, cmap='hot', interpolation='nearest')
    ax3.set_title(f'Uncertainty (Variance)')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Ground truth flow (if available)
    if 'flow_gt_0to1' in data:
        flow_gt = data['flow_gt_0to1'][0].detach().cpu().numpy()  # (H, W, 2)
        ax4 = plt.subplot(2, 3, 4)
        _plot_flow_field(
            ax4, flow_gt[..., 0], flow_gt[..., 1],
            title='Ground Truth Flow', stride=8
        )
        
        # 5. Flow error map
        flow_error = np.sqrt(
            (flow_x - flow_gt[..., 0])**2 + 
            (flow_y - flow_gt[..., 1])**2
        )
        ax5 = plt.subplot(2, 3, 5)
        im5 = ax5.imshow(flow_error, cmap='Reds', interpolation='nearest')
        ax5.set_title(f'Flow Error (mean: {flow_error.mean():.2f} px)')
        ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # 6. Input images overlay
    if 'image0' in data:
        ax6 = plt.subplot(2, 3, 6)
        img0 = data['image0'][0].permute(1, 2, 0).detach().cpu().numpy()
        img0 = (img0 - img0.min()) / (img0.max() - img0.min() + 1e-8)
        
        # Resize if needed
        if img0.shape[:2] != (H, W):
            img0 = cv2.resize(img0, (W, H))
        
        ax6.imshow(img0)
        # Overlay flow on image
        _plot_flow_field(
            ax6, flow_x, flow_y, title='Flow on Image 0',
            stride=12, overlay=True
        )
    
    plt.suptitle(f'AS-Mamba Flow Visualization - Block {block_idx}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    elif return_fig:
        return fig
    else:
        plt.show()
        return None


def _plot_flow_field(ax, flow_x, flow_y, title='', stride=8, overlay=False):
    """Helper function to plot flow field as arrows."""
    H, W = flow_x.shape
    y_coords, x_coords = np.meshgrid(
        np.arange(0, H, stride),
        np.arange(0, W, stride),
        indexing='ij'
    )
    
    u = flow_x[::stride, ::stride]
    v = flow_y[::stride, ::stride]
    
    # Color by magnitude
    magnitude = np.sqrt(u**2 + v**2)
    
    if not overlay:
        ax.imshow(np.zeros((H, W)), cmap='gray', alpha=0.3)
    
    quiver = ax.quiver(
        x_coords, y_coords, u, v,
        magnitude,
        cmap='jet',
        scale=None,
        scale_units='xy',
        angles='xy',
        width=0.003,
        headwidth=4,
        headlength=5,
        alpha=0.8 if overlay else 1.0
    )
    
    if not overlay:
        plt.colorbar(quiver, ax=ax, fraction=0.046, pad=0.04, label='Flow Magnitude')
    
    ax.set_title(title)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')


def make_adaptive_span_visualization(data, path=None, return_fig=False):
    """
    Visualize adaptive span distributions.
    
    Shows:
    1. Span size heatmaps for both images
    2. Span distribution histogram
    3. Span-uncertainty correlation
    4. Sample adaptive windows
    
    Args:
        data: Batch dictionary with 'adaptive_spans'
        path: Save path
        return_fig: Whether to return figure
    
    Returns:
        Figure object if return_fig=True
    """
    if 'adaptive_spans' not in data:
        return None
    
    spans_x, spans_y = data['adaptive_spans']
    
    # Take first sample
    span_x = spans_x[0].detach().cpu().numpy()
    span_y = spans_y[0].detach().cpu().numpy()
    
    # Average span size
    span_avg = (span_x + span_y) / 2.0
    
    H, W = span_avg.shape
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    
    # 1. X-direction span heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(span_x, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Adaptive Spans (X-direction)\nRange: [{span_x.min():.0f}, {span_x.max():.0f}]')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Y-direction span heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(span_y, cmap='viridis', interpolation='nearest')
    ax2.set_title(f'Adaptive Spans (Y-direction)\nRange: [{span_y.min():.0f}, {span_y.max():.0f}]')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Average span heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(span_avg, cmap='plasma', interpolation='nearest')
    ax3.set_title(f'Average Adaptive Span\nMean: {span_avg.mean():.2f}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 4. Span distribution histogram
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(span_avg.flatten(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(span_avg.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {span_avg.mean():.2f}')
    ax4.axvline(np.median(span_avg), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(span_avg):.2f}')
    ax4.set_xlabel('Span Size (pixels)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Span Distribution')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. Span-uncertainty correlation (if available)
    ax5 = plt.subplot(2, 3, 5)
    if 'predict_flow' in data and len(data['predict_flow']) > 0:
        flow = data['predict_flow'][-1][0][0, 0].detach().cpu().numpy()  # (H, W, 4)
        uncertainty = (np.exp(flow[..., 2]) + np.exp(flow[..., 3])) / 2.0
        
        # Scatter plot
        ax5.scatter(
            uncertainty.flatten()[::10],
            span_avg.flatten()[::10],
            alpha=0.3,
            s=1,
            c='steelblue'
        )
        ax5.set_xlabel('Uncertainty (Variance)')
        ax5.set_ylabel('Adaptive Span Size')
        ax5.set_title('Span-Uncertainty Correlation')
        ax5.grid(alpha=0.3)
        
        # Compute correlation
        corr = np.corrcoef(uncertainty.flatten(), span_avg.flatten())[0, 1]
        ax5.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax5.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax5.text(0.5, 0.5, 'Uncertainty data not available',
                ha='center', va='center', transform=ax5.transAxes)
        ax5.axis('off')
    
    # 6. Sample adaptive windows
    ax6 = plt.subplot(2, 3, 6)
    _plot_sample_windows(ax6, span_avg, n_samples=10)
    
    plt.suptitle('AS-Mamba Adaptive Span Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    elif return_fig:
        return fig
    else:
        plt.show()
        return None


def _plot_sample_windows(ax, span_map, n_samples=10):
    """Plot sample adaptive windows on the span map."""
    H, W = span_map.shape
    
    # Display the span map
    ax.imshow(span_map, cmap='plasma', interpolation='nearest', alpha=0.6)
    
    # Sample random positions
    np.random.seed(42)
    sample_positions = [
        (np.random.randint(10, H-10), np.random.randint(10, W-10))
        for _ in range(n_samples)
    ]
    
    # Draw rectangles for each sample
    colors = plt.cm.Set3(np.linspace(0, 1, n_samples))
    
    for idx, (y, x) in enumerate(sample_positions):
        span_size = int(span_map[y, x])
        half_span = span_size // 2
        
        # Create rectangle
        rect = patches.Rectangle(
            (x - half_span, y - half_span),
            span_size, span_size,
            linewidth=2,
            edgecolor=colors[idx],
            facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(x, y, f'{span_size}', color='white',
               ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor=colors[idx], alpha=0.8))
    
    ax.set_title(f'Sample Adaptive Windows (n={n_samples})')
    ax.axis('off')


def make_multiscale_comparison(data, path=None, return_fig=False):
    """
    Visualize multi-scale feature processing.
    
    Compares:
    1. Global path predictions (coarse)
    2. Local path predictions (fine)
    3. Combined predictions
    
    Args:
        data: Batch with global_matches and local_matches
        path: Save path
        return_fig: Whether to return figure
    """
    if 'global_matches' not in data or 'local_matches' not in data:
        return None
    
    global_conf = data['global_matches'][0].detach().cpu().numpy()
    local_conf = data['local_matches'][0].detach().cpu().numpy()
    
    # Create figure
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Global path
    ax1 = plt.subplot(1, 3, 1)
    im1 = ax1.imshow(global_conf, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
    ax1.set_title('Global Path (Downsampled)')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # 2. Local path
    ax2 = plt.subplot(1, 3, 2)
    im2 = ax2.imshow(local_conf, cmap='jet', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_title('Local Path (Adaptive Spans)')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. Difference
    ax3 = plt.subplot(1, 3, 3)
    diff = np.abs(global_conf - local_conf)
    im3 = ax3.imshow(diff, cmap='RdYlGn_r', interpolation='nearest')
    ax3.set_title(f'Absolute Difference\nMean: {diff.mean():.4f}')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.suptitle('AS-Mamba Multi-Scale Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        return None
    elif return_fig:
        return fig
    else:
        plt.show()
        return None


# Re-export standard visualization for convenience
try:
    from src.utils.plotting import make_matching_figures
except ImportError:
    def make_matching_figures(data, mode, path=None, return_fig=False):
        """Placeholder - import from standard utilities."""
        raise NotImplementedError("Import from standard plotting utilities")