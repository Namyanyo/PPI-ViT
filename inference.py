
import argparse
import torch
import numpy as np
import json
from pathlib import Path

from model import PPIClassifier, PPIResNet
from pdb_parser import ProteinStructureParser
from pointcloud_to_image import PointCloudProjector


class PPIPredictor:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize PPI predictor

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

 
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)


        self.config = checkpoint['config']


        if self.config['model_name'] == 'cnn':
            self.model = PPIClassifier(
                num_classes=2,
                dropout_rate=self.config.get('dropout_rate', 0.5)
            )
        elif self.config['model_name'] == 'resnet':
            self.model = PPIResNet(
                num_classes=2,
                dropout_rate=self.config.get('dropout_rate', 0.5)
            )


        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

  
        self.parser = ProteinStructureParser()
        self.projector = PointCloudProjector(
            image_size=self.config.get('image_size', 224),
            method=self.config.get('projection_method', 'density_map')
        )

        print(f"Model loaded successfully!")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model_name']}")

    def preprocess_pdb(self, pdb_path, ca_only=False):
        """
        Preprocess PDB file to image tensor

        Args:
            pdb_path: Path to PDB file
            ca_only: If True, use only C-alpha atoms

        Returns:
            image_tensor: Tensor of shape (1, 3, H, W)
        """

        if ca_only:
            coords = self.parser.get_ca_atoms_only(pdb_path)
        else:
            coords, _ = self.parser.parse_pdb(pdb_path)


        normalized_coords, _, _ = self.parser.normalize_coordinates(coords)

 
        image = self.projector.project_to_2d(normalized_coords)


        image_tensor = torch.from_numpy(image).float()
        image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        return image_tensor

    def predict(self, pdb_path, ca_only=False):
        """
        Predict PPI interaction for a single protein

        Args:
            pdb_path: Path to PDB file
            ca_only: If True, use only C-alpha atoms

        Returns:
            prediction: Predicted class (0 or 1)
            probability: Probability of positive class
            confidence: Confidence score
        """
  
        image_tensor = self.preprocess_pdb(pdb_path, ca_only)
        image_tensor = image_tensor.to(self.device)


        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            probability = probs[0, 1].item()
            confidence = probs[0, prediction].item()

        return prediction, probability, confidence

    def predict_batch(self, pdb_paths, ca_only=False):
        """
        Predict PPI interaction for multiple proteins

        Args:
            pdb_paths: List of PDB file paths
            ca_only: If True, use only C-alpha atoms

        Returns:
            results: List of dictionaries with predictions
        """
        results = []

        for pdb_path in pdb_paths:
            try:
                prediction, probability, confidence = self.predict(pdb_path, ca_only)

                results.append({
                    'pdb_path': pdb_path,
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'confidence': float(confidence),
                    'success': True
                })

                print(f"✓ {Path(pdb_path).name}: Class {prediction} "
                      f"(Prob: {probability:.4f}, Conf: {confidence:.4f})")

            except Exception as e:
                results.append({
                    'pdb_path': pdb_path,
                    'error': str(e),
                    'success': False
                })
                print(f"✗ {Path(pdb_path).name}: Error - {e}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Predict PPI interactions')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--pdb_file', type=str,
                       help='Path to single PDB file')
    parser.add_argument('--pdb_list', type=str,
                       help='Path to text file with list of PDB files')
    parser.add_argument('--ca_only', action='store_true',
                       help='Use only C-alpha atoms')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output JSON file for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')

    args = parser.parse_args()


    predictor = PPIPredictor(args.checkpoint, device=args.device)


    pdb_files = []

    if args.pdb_file:
        pdb_files.append(args.pdb_file)

    if args.pdb_list:
        with open(args.pdb_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    pdb_files.append(line)

    if not pdb_files:
        print("Error: No PDB files specified!")
        print("Use --pdb_file for single file or --pdb_list for multiple files")
        return

    print(f"\nPredicting {len(pdb_files)} protein(s)...")
    print("-" * 80)


    results = predictor.predict_batch(pdb_files, ca_only=args.ca_only)


    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("-" * 80)
    print(f"\nResults saved to {args.output}")


    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"\nSummary:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    if successful > 0:
        positive = sum(1 for r in results if r['success'] and r['prediction'] == 1)
        negative = successful - positive
        print(f"\nPredictions:")
        print(f"  Positive (interacting): {positive}")
        print(f"  Negative (non-interacting): {negative}")


if __name__ == "__main__":
    main()
