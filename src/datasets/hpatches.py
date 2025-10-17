"""
HPatches Dataset for AS-Mamba

HPatches contains 116 sequences (59 viewpoint + 57 illumination changes).
Each sequence has 6 images with known homographies.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from loguru import logger
import cv2

from src.utils.dataset import (
    read_megadepth_gray,
    get_resized_wh,
    get_divisible_wh,
    pad_bottom_right
)


class HPatchesDataset(Dataset):
    """
    HPatches dataset for image matching evaluation.
    
    Dataset structure:
        hpatches-sequences-release/
            v_abstract/
                1.ppm
                2.ppm
                ...
                6.ppm
                H_1_2  # Homography from image 1 to 2
                H_1_3
                ...
            v_adam/
            ...
            i_ajuntament/
            ...
    
    Args:
        data_root: Path to hpatches-sequences-release directory
        mode: 'test' only (HPatches is for evaluation)
        min_overlap_score: Not used (kept for compatibility)
        img_resize: Resize longer edge to this size
        df: Make dimensions divisible by this factor
        img_padding: Pad to square
        augment_fn: Not used for test
        alteration: 'all', 'v' (viewpoint), 'i' (illumination)
    """
    
    def __init__(
        self,
        data_root,
        mode='test',
        min_overlap_score=0.0,  # Not used but kept for compatibility
        img_resize=480,
        df=8,
        img_padding=True,
        augment_fn=None,
        alteration='all'
    ):
        super().__init__()
        
        assert mode == 'test', "HPatches is only for testing/evaluation"
        assert alteration in ['all', 'v', 'i'], \
            "alteration must be 'all', 'v' (viewpoint), or 'i' (illumination)"
        
        self.data_root = Path(data_root)
        self.mode = mode
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.alteration = alteration
        
        # Load all sequences
        self.sequences = self._load_sequences()
        
        # Create all image pairs
        self.pairs = self._create_pairs()
        
        logger.info(
            f"HPatchesDataset initialized: "
            f"{len(self.sequences)} sequences, "
            f"{len(self.pairs)} pairs, "
            f"alteration={alteration}"
        )
    
    def _load_sequences(self):
        """Load all sequence directories."""
        sequences = []
        
        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            seq_name = seq_dir.name
            
            # Filter by alteration type
            if self.alteration == 'v' and not seq_name.startswith('v_'):
                continue
            if self.alteration == 'i' and not seq_name.startswith('i_'):
                continue
            
            sequences.append(seq_dir)
        
        return sequences
    
    def _create_pairs(self):
        """Create all image pairs with their homographies."""
        pairs = []
        
        for seq_dir in self.sequences:
            seq_name = seq_dir.name
            
            # Reference image is always 1.ppm
            ref_img_path = seq_dir / "1.ppm"
            
            # Pair with images 2-6
            for i in range(2, 7):
                target_img_path = seq_dir / f"{i}.ppm"
                homography_path = seq_dir / f"H_1_{i}"
                
                # Check if files exist
                if not ref_img_path.exists():
                    logger.warning(f"Missing reference image: {ref_img_path}")
                    continue
                if not target_img_path.exists():
                    logger.warning(f"Missing target image: {target_img_path}")
                    continue
                if not homography_path.exists():
                    logger.warning(f"Missing homography: {homography_path}")
                    continue
                
                pairs.append({
                    'seq_name': seq_name,
                    'ref_img': str(ref_img_path),
                    'target_img': str(target_img_path),
                    'homography': str(homography_path),
                    'pair_name': f"{seq_name}_1_{i}"
                })
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def _read_image(self, img_path):
        """
        Read and preprocess image.
        
        Returns:
            image: (1, H, W) tensor, normalized to [0, 1]
            mask: (H, W) tensor, valid region mask
            scale: (2,) tensor, [w_scale, h_scale]
        """
        # Read grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")
        
        h, w = image.shape
        
        # Resize
        if self.img_resize is not None:
            w_new, h_new = get_resized_wh(w, h, self.img_resize)
            w_new, h_new = get_divisible_wh(w_new, h_new, self.df)
            
            image = cv2.resize(image, (w_new, h_new))
            scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float32)
        else:
            w_new, h_new = get_divisible_wh(w, h, self.df)
            if (w_new, h_new) != (w, h):
                image = cv2.resize(image, (w_new, h_new))
            scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float32)
        
        # Padding
        if self.img_padding:
            pad_to = max(h_new, w_new)
            image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones((h_new, w_new), dtype=torch.bool)
        
        # Normalize to [0, 1]
        image = torch.from_numpy(image).float()[None] / 255.0
        
        return image, mask, scale
    
    def _read_homography(self, homography_path):
        """Read 3x3 homography matrix from file."""
        H = np.loadtxt(homography_path)
        return torch.from_numpy(H).float()
    
    def __getitem__(self, idx):
        """
        Returns:
            data (dict): {
                'image0': (1, H, W)
                'image1': (1, H, W)
                'mask0': (H, W)
                'mask1': (H, W)
                'scale0': (2,)
                'scale1': (2,)
                'H_0to1': (3, 3) homography matrix
                'dataset_name': 'HPatch'
                'scene_id': sequence name
                'pair_id': pair identifier
                'pair_names': (ref_name, target_name)
            }
        """
        pair = self.pairs[idx]
        
        # Read images
        image0, mask0, scale0 = self._read_image(pair['ref_img'])
        image1, mask1, scale1 = self._read_image(pair['target_img'])
        
        # Read homography
        H_0to1 = self._read_homography(pair['homography'])
        
        # Adjust homography for resizing
        # H_adjusted = S1 @ H @ S0^{-1}
        # where S0 scales from original to resized coordinates for image0
        S0 = torch.diag(torch.tensor([1.0 / scale0[0], 1.0 / scale0[1], 1.0]))
        S1 = torch.diag(torch.tensor([1.0 / scale1[0], 1.0 / scale1[1], 1.0]))
        H_adjusted = S1 @ H_0to1 @ torch.inverse(S0)
        
        data = {
            'image0': image0,
            'image1': image1,
            'mask0': mask0,
            'mask1': mask1,
            'scale0': scale0,
            'scale1': scale1,
            'H_0to1': H_adjusted,  # Homography in resized coordinates
            'H_0to1_original': H_0to1,  # Original homography
            'dataset_name': 'HPatches',
            'scene_id': pair['seq_name'],
            'pair_id': idx,
            'pair_names': (
                Path(pair['ref_img']).stem,
                Path(pair['target_img']).stem
            )
        }
        
        return data