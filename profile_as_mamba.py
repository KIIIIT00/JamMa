import torch
import gc
from src.jamma.as_mamba import AS_Mamba
from src.config.as_mamba_default import get_cfg_defaults
import torch.profiler
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.module')
warnings.filterwarnings("ignore", category=FutureWarning)

def get_memory_usage(device, reset=True):
    if reset:
        gc.collect()
        torch.cuda.empty_cache()
    return torch.cuda.max_memory_allocated(device) / 1024**2

def test_profiling(batch_size=1, device='cuda'):
    print("\n==============================")
    print("Profiling Model Forward Pass...")
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    print("==============================")

    # Model configuration
    config = get_cfg_defaults()
    config_dict = {
        'coarse': {'d_model': config.AS_MAMBA.COARSE.D_MODEL},
        'fine': {'d_model': config.AS_MAMBA.FINE.D_MODEL,
                 'dsmax_temperature': 0.1,
                 'inference': True,
                 'thr': 0.2,
                 },
        'resolution': config.AS_MAMBA.RESOLUTION,
        'fine_window_size': config.AS_MAMBA.FINE_WINDOW_SIZE,
        'match_coarse': {
            'thr': 0.2, 'use_sm': True, 'border_rm': 2,
            'dsmax_temperature': 0.1, 'inference': True
        },
        'as_mamba': {
            'n_blocks': 1,  # Reduce for memory test
            'd_geom': 32,
            'use_kan_flow': False,
            'global_depth': 1,
            'local_depth': 1,
            'use_geom_for_fine': False,
            'window_size': 2 
        }
    }
    # config = get_cfg_defaults()
    # print(config)
    model = AS_Mamba(config_dict).to(device).eval()
    # model = AS_Mamba(config.AS_MAMBA).to(device).eval()

    # Dummy data
    image_size = (256, 256) 
    c, h, w = 3, image_size[0], image_size[1]
    
    # Use float32 for stable profiling
    im_ref = torch.randn(batch_size, c, h, w, device=device, dtype=torch.float32)
    im_src = torch.randn(batch_size, c, h, w, device=device, dtype=torch.float32)

    h_c, w_c = h // 8, w // 8  # Coarse level (1/8) dimensions
    h_f, w_f = h // 2, w // 2  # Fine level (1/2) dimensions 

    data = {
        'imagec_0': im_ref,
        'imagec_1': im_src,
        'hw0_i': [h, w],
        'hw1_i': [h, w],
        'bs': batch_size,
        'h_8': h_c,
        'w_8': w_c,
        'h_4': h_f,
        'w_4': w_f, 
    }
    
    # Warm-up run (to exclude initialization costs from profiling)
    print("Performing warm-up run...")
    with torch.no_grad():
        try:
            model(data, mode='test')
        except Exception as e:
            print(f"Error during warm-up: {e}")
            return False
    print("Warm-up complete.")

    # Profiling
    print("Starting profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/as_mamba_profile'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5): # Run multiple iterations to get stable results
                model(data, mode='test')
                prof.step()

    print("Profiling complete.")
    print("\n--- Top 15 operators by CUDA memory usage ---")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=15))
    
    print("\n--- Top 15 operators by CPU memory usage ---")
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=15))

    print("\n--- Top 15 operators by CUDA time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n✅ Profiling finished successfully.")
    print("Detailed trace can be viewed with TensorBoard. Run: tensorboard --logdir ./log")
    
    return True

def main():
    # Execute the profiling test
    success = test_profiling(batch_size=1) 
    
    if success:
        print("\n🎉 All profiling tests passed!")
    else:
        print("\n⚠️ Profiling failed. Please check the errors above.")

if __name__ == '__main__':
    main()