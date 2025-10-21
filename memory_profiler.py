# memory_profiler.py
import torch
import argparse
from loguru import logger
import numpy as np
import time
import matplotlib.pyplot as plt

from src.config.as_mamba_default import get_cfg_defaults
from src.lightning.lightning_as_mamba import PL_ASMamba

def memory_profile_model(config_path, model_size='small'):
    """様々な設定でモデルのメモリ使用量をプロファイリング"""
    config = get_cfg_defaults()
    config.merge_from_file(config_path)
    
    # モデルサイズの設定
    if model_size == 'small':
        config.AS_MAMBA.N_BLOCKS = 1
        config.AS_MAMBA.GLOBAL_DEPTH = 2
        config.AS_MAMBA.LOCAL_DEPTH = 2
    elif model_size == 'medium':
        config.AS_MAMBA.N_BLOCKS = 2
        config.AS_MAMBA.GLOBAL_DEPTH = 3
        config.AS_MAMBA.LOCAL_DEPTH = 3
    elif model_size == 'large':
        config.AS_MAMBA.N_BLOCKS = 3
        config.AS_MAMBA.GLOBAL_DEPTH = 4
        config.AS_MAMBA.LOCAL_DEPTH = 4
    
    # メモリ使用量記録
    memory_usage = []
    batch_sizes = [1, 2, 4, 8]
    
    for bs in batch_sizes:
        # GPUメモリをクリア
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # モデルを初期化
        model = PL_ASMamba(config)
        model = model.cuda()
        
        # ダミー入力を作成
        dummy_batch = {
            'imagec_0': torch.randn(bs, 3, 480, 640).cuda(),
            'imagec_1': torch.randn(bs, 3, 480, 640).cuda(),
            'bs': bs,
            'h_8': 60,
            'w_8': 80,
        }
        
        # 前方伝搬と逆伝搬
        start_time = time.time()
        try:
            with torch.cuda.amp.autocast(enabled=config.AS_MAMBA.MP):
                # バックボーン
                model.backbone(dummy_batch)
                
                # AS-Mambaマッチング
                model.matcher(dummy_batch, mode='train')
                
                # メモリ使用量を記録
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_allocated = torch.cuda.max_memory_allocated() / 1024**3
                
            memory_usage.append({
                'batch_size': bs,
                'model_size': model_size,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'time_sec': time.time() - start_time
            })
            
            logger.info(f"Batch Size {bs}: Allocated {allocated:.2f} GB, "
                      f"Reserved {reserved:.2f} GB, Max {max_allocated:.2f} GB")
            
        except RuntimeError as e:
            logger.error(f"Error with batch size {bs}: {e}")
            memory_usage.append({
                'batch_size': bs,
                'model_size': model_size,
                'allocated_gb': -1,  # エラーマーク
                'error': str(e)
            })
        
        # モデルを解放
        del model
        torch.cuda.empty_cache()
    
    return memory_usage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='memory_profile.png')
    args = parser.parse_args()
    
    model_sizes = ['small', 'medium', 'large']
    all_results = []
    
    for size in model_sizes:
        results = memory_profile_model(args.config, size)
        all_results.extend(results)
    
    # 結果をグラフ化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for size in model_sizes:
        size_results = [r for r in all_results if r['model_size'] == size and 'error' not in r]
        if size_results:
            batch_sizes = [r['batch_size'] for r in size_results]
            memory = [r['max_allocated_gb'] for r in size_results]
            time_values = [r['time_sec'] for r in size_results]
            
            ax1.plot(batch_sizes, memory, 'o-', label=f"{size} model")
            ax2.plot(batch_sizes, time_values, 'o-', label=f"{size} model")
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('GPU Memory (GB)')
    ax1.set_title('Memory Usage vs Batch Size')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Time (sec)')
    ax2.set_title('Forward Pass Time vs Batch Size')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(args.output)
    logger.info(f"Results saved to {args.output}")
    
if __name__ == '__main__':
    main()