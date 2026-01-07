"""
Dataset loader for PPI binary classification
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pdb_parser import ProteinStructureParser
from pointcloud_to_image import PointCloudProjector
import json
from pathlib import Path


class PPIDataset(Dataset):
    def __init__(self, data_file=None, 
                 image_size=224, projection_method='density_map', 
                 ca_only=False, transform=None, cache_dir=None, validate=True):
        """
        Dataset for PPI binary classification
        
        Args:
            data_file: JSON file containing samples
            pdb_dir: Directory containing PDB files
            labels_file: JSON file with labels 
            image_size: Size of output images
            projection_method: Method for point cloud projection
            ca_only: If True, use only C-alpha atoms
            transform: Optional transform to apply
            cache_dir: Directory to cache processed images
            validate: If True, validate all PDB files before training
        """
        self.data_file= data_file
        self.image_size = image_size
        self.projection_method = projection_method
        self.ca_only = ca_only
        self.transform = transform
        self.cache_dir = cache_dir
        self.parser = ProteinStructureParser()
        self.projector = PointCloudProjector(
            image_size=image_size,
            method=projection_method
        )
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self.pdb_files, self.labels = self._load_from_data_file(data_file)

        if validate:
            self._validate_dataset()
    
    def _load_from_data_file(self, data_file):

        with open(data_file, 'r') as f:
            samples = json.load(f)
        
        pdb_files = []
        labels = []
        
        for sample in samples:
            pdb_path = sample['pdb_path']
            label = sample['label']
            
            if os.path.exists(pdb_path):
                pdb_files.append(pdb_path)
                labels.append(label)
            else:
                print(f"Warning: PDB file not found: {pdb_path}")
        
        print(f"Loaded {len(pdb_files)} samples from {data_file}")
        return pdb_files, labels
    
    def _load_from_dir(self, pdb_dir, labels_file):

        # Load labels
        with open(labels_file, 'r') as f:
            labels_dict = json.load(f)
        
        pdb_files = []
        labels = []
        
        for pdb_name, label in labels_dict.items():
            pdb_path = os.path.join(pdb_dir, pdb_name)
            if os.path.exists(pdb_path):
                pdb_files.append(pdb_path)
                labels.append(label)
            else:
                print(f"Warning: PDB file not found: {pdb_path}")
        
        print(f"Loaded {len(pdb_files)} samples from {pdb_dir}")
        return pdb_files, labels
    
    def _validate_dataset(self):
        """Validate dataset"""
        print("Validating dataset...")
        valid_indices = []
        
        for idx in range(len(self.pdb_files)):
            try:
                pdb_path = self.pdb_files[idx]
                
                if self.ca_only:
                    coords = self.parser.get_ca_atoms_only(pdb_path)
                else:
                    coords = self.parser.parse_pdb(pdb_path)
                if coords is not None and coords.size > 0:
                    valid_indices.append(idx)
                else:
                    print(f"Empty coordinates: {pdb_path}")
                    
            except Exception as e:
                print(f"Invalid PDB: {self.pdb_files[idx]} - {e}")
        self.pdb_files = [self.pdb_files[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]
        
        print(f"Valid samples: {len(self.pdb_files)}")
    
    def _get_cache_path(self, idx):
        """Get cache file path for sample"""
        if not self.cache_dir:
            return None
        
        pdb_basename = os.path.basename(self.pdb_files[idx])
        cache_name = f"{os.path.splitext(pdb_basename)[0]}.npy"
        return os.path.join(self.cache_dir, cache_name)
    
    def _process_pdb(self, pdb_path):
        """Process PDB file to 2D image"""
        if self.ca_only:
            coords = self.parser.get_ca_atoms_only(pdb_path)
        else:
            coords = self.parser.parse_pdb(pdb_path)
        if coords is None or coords.size == 0:
            raise ValueError(f"Empty coordinates from {pdb_path}")
        
        normalized_coords, _, _ = self.parser.normalize_coordinates(coords)
        image = self.projector.project_to_2d(normalized_coords)
        
        return image
    
    def __len__(self):
        return len(self.pdb_files)
    
    def __getitem__(self, idx):
        """
        Get a sample

        Returns:
            image: Tensor of shape (3, H, W)
            label: Binary label (0 or 1)
        """
        try:
            pdb_path = self.pdb_files[idx]
            label = self.labels[idx]
            cache_path = self._get_cache_path(idx)
            if cache_path and os.path.exists(cache_path):
                image = np.load(cache_path)
            else:
                image = self._process_pdb(pdb_path)
                
                if cache_path:
                    np.save(cache_path, image)
            
            image = torch.from_numpy(image).float()
            if len(image.shape) == 3 and image.shape[2] <= 3:
                image = image.permute(2, 0, 1) 

            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor(label, dtype=torch.long)
            
            return image, label
            
        except Exception as e:
            print(f"\nError at idx {idx}: {self.pdb_files[idx]}")
            print(f"Error: {e}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)




def create_data_loaders(train_file, val_file, batch_size=32, num_workers=4,
                       image_size=224, projection_method='density_map',
                       ca_only=False, transform=None, cache_dir=None,
                       validate=True):
    """
    Create train and validation data loaders

    Args:
        train_file: Path to training data JSON file
        val_file: Path to validation data JSON file
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size
        projection_method: Projection method
        ca_only: Use only CA atoms
        transform: Optional transforms
        cache_dir: Cache directory
        validate: Validate dataset before training

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Create datasets
    print(f"Loading training data from {train_file}...")
    train_dataset = PPIDataset(
        data_file=train_file,
        image_size=image_size,
        projection_method=projection_method,
        ca_only=ca_only,
        transform=transform,
        cache_dir=cache_dir,
        validate=validate
    )

    print(f"\nLoading validation data from {val_file}...")
    val_dataset = PPIDataset(
        data_file=val_file,
        image_size=image_size,
        projection_method=projection_method,
        ca_only=ca_only,
        transform=transform,
        cache_dir=cache_dir,
        validate=validate
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader

