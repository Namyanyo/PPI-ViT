"""
Training script for PPI binary classification
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime

from model import PPIClassifier, PPIViTSmall, PPIViTBase
from dataset import create_data_loaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, config):
        """
        Initialize trainer

        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            config: Training configuration dictionary
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9,
                weight_decay=config['weight_decay']
            )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.writer = SummaryWriter(log_dir=config['log_dir'])

        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)


            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].detach().cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        return metrics

    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]')

            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        return metrics

    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['num_epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['model_name']}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print("-" * 80)

        for epoch in range(self.config['num_epochs']):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            self.writer.add_scalar('AUC/train', train_metrics['auc'], epoch)
            self.writer.add_scalar('AUC/val', val_metrics['auc'], epoch)

            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")

            self.scheduler.step(val_metrics['accuracy'])
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"*** New best model saved! Accuracy: {self.best_val_acc:.4f} ***")

            if (epoch + 1) % self.config['save_interval'] == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_metrics)

        print("\n" + "=" * 80)
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}")
        print("=" * 80)

        self.writer.close()

    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Train PPI classifier')
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to training data JSON file')
    parser.add_argument('--val_file', type=str, required=True,
                       help='Path to validation data JSON file')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'resnet', 'vit_small', 'vit_base'],
                       help='Model architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Image size')
    parser.add_argument('--projection_method', type=str, default='density_map',
                       choices=['density_map', 'scatter', 'voxel'],
                       help='Point cloud projection method')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'sgd'], help='Optimizer')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for processed images')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for tensorboard logs')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading data...")
    train_loader, val_loader = create_data_loaders(
        train_file=args.train_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        projection_method=args.projection_method,
        cache_dir=args.cache_dir
    )

    print(f"Creating {args.model} model...")
    if args.model == 'cnn':
        model = PPIClassifier(num_classes=2, dropout_rate=args.dropout_rate)
    elif args.model == 'vit_small':
        model = PPIViTSmall(image_size=args.image_size, num_classes=2, dropout=args.dropout_rate)
    elif args.model == 'vit_base':
        model = PPIViTBase(image_size=args.image_size, num_classes=2, dropout=args.dropout_rate)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    config = {
        'model_name': args.model,
        'image_size': args.image_size,
        'projection_method': args.projection_method,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'dropout_rate': args.dropout_rate,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'save_interval': args.save_interval,
        'seed': args.seed
    }

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    trainer = Trainer(model, train_loader, val_loader, device, config)
    trainer.train()


if __name__ == "__main__":
    main()
