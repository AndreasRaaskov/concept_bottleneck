
"""
Provides a class to track training metrics and log them to a JSON file
"""

import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from collections import defaultdict
from pathlib import Path
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from collections import defaultdict as ddict

import seaborn as sns
from sklearn.metrics import confusion_matrix
'''
class TrainingLogger:
    def __init__(self, log_file: str = 'training_log.json'):
        self.log_file = log_file
        self.reset()
        self.all_epochs_data = []
        
        # Create the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump({"initialization_time": str(datetime.now())}, f)

    def reset(self):
        """Reset all accumulated data"""
        self.class_data = defaultdict(lambda: {'correct': 0, 'top5_correct': 0, 'total': 0})
        self.concept_data = defaultdict(lambda: {'true_positives': 0, 'true_negatives': 0, 'false_positives': 0, 'false_negatives': 0, 'total': 0})
        self.loss_data = defaultdict(list)
        self.sailency_scores = ddict(list) #Dictionary to store the sailency scores of the model only use for testing

    def update_class_accuracy(self, mode: str, logits: torch.Tensor, correct_label: torch.Tensor):
        logits = logits.detach().cpu().numpy()
        correct_label = correct_label.detach().cpu().numpy()

        self.class_data[mode]['total'] += logits.shape[0]
        top_predictions = np.argsort(logits, axis=1)[:, -5:]
        correct_classes = np.argmax(correct_label, axis=1)
        
        self.class_data[mode]['correct'] += np.sum(top_predictions[:, -1] == correct_classes)
        self.class_data[mode]['top5_correct'] += np.sum([correct_class in top5 for correct_class, top5 in zip(correct_classes, top_predictions)])

    def update_concept_accuracy(self, mode: str, predictions: torch.Tensor, ground_truth: torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()

        # Make sure predictions and ground truth are binary
        predictions = (predictions >= 0.5).astype(bool)
        ground_truth = (ground_truth >= 0.5).astype(bool)

        self.concept_data[mode]['total'] += predictions.shape[0] * predictions.shape[1]
        self.concept_data[mode]['true_positives'] += np.sum(np.logical_and(predictions == 1, ground_truth == 1))
        self.concept_data[mode]['true_negatives'] += np.sum(np.logical_and(predictions == 0, ground_truth == 0))
        self.concept_data[mode]['false_positives'] += np.sum(np.logical_and(predictions == 1, ground_truth == 0))
        self.concept_data[mode]['false_negatives'] += np.sum(np.logical_and(predictions == 0, ground_truth == 1))


    def update_loss(self, mode: str, loss: float):
        """Update loss for the given mode"""
        self.loss_data[mode].append(loss.item())
    
    def update_sailency_score(self, mode: str, score: torch.Tensor):
        """Update sailency scores for the given mode"""
        self.sailency_scores[mode].append(score)

    def get_class_metrics(self, mode: str) -> Dict[str, float]:
        if self.class_data[mode]['total'] == 0:
            return {'top1_accuracy': 0, 'top5_accuracy': 0}
        return {
            'top1_accuracy': self.class_data[mode]['correct'] / self.class_data[mode]['total'],
            'top5_accuracy': self.class_data[mode]['top5_correct'] / self.class_data[mode]['total']
        }

    def get_concept_metrics(self, mode: str) -> Dict[str, float]:
        tp = self.concept_data[mode]['true_positives']
        tn = self.concept_data[mode]['true_negatives']
        fp = self.concept_data[mode]['false_positives']
        fn = self.concept_data[mode]['false_negatives']

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_loss_metrics(self, mode: str) -> Dict[str, float]:
        
        if not self.loss_data[mode]:
            return {'avg_loss': 0}
        return {
            'avg_loss': sum(self.loss_data[mode]) / len(self.loss_data[mode])
        }
    
    def get_sailency_scores(self, mode: str) -> Dict[str, float]:
        return {
            'sailency_scores': sum(self.sailency_scores[mode]) / len(self.sailency_scores[mode])
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        metrics = {}
        all_modes = set(list(self.class_data.keys()) + list(self.concept_data.keys()) + list(self.loss_data.keys()))
        for mode in all_modes:
            mode_metrics = {}
            if mode in self.class_data:
                mode_metrics['class_metrics'] = self.get_class_metrics(mode)
            if mode in self.concept_data:
                mode_metrics['concept_metrics'] = self.get_concept_metrics(mode)
            if mode in self.loss_data:
                mode_metrics['loss_metrics'] = self.get_loss_metrics(mode)
            if mode in self.sailency_scores:
                mode_metrics['sailency_scores'] = self.get_sailency_scores(mode)
            metrics[mode] = mode_metrics
        
        #Reset for a new train run.
        self.reset()
        return metrics

    def log_metrics(self, epoch: int):
        metrics = self.get_all_metrics()
        
        # Add epoch data to all_epochs_data
        epoch_data = {
            "epoch": epoch,
            "timestamp": str(datetime.now()),
            "metrics": metrics
        }
        self.all_epochs_data.append(epoch_data)
        
        # Save all data to JSON file
        with open(self.log_file, 'w') as f:
            json.dump(self.all_epochs_data, f, indent=2)
        
        # Print formatted metrics to console
        #print(f"\nEpoch {epoch} Training Metrics:")
        #print(self.format_metrics(metrics))

    def format_metrics(self, metrics: Dict[str, Dict[str, Any]]) -> str:
        formatted = ""
        for mode, mode_metrics in metrics.items():
            formatted += f"Mode: {mode}\n"
            if 'class_metrics' in mode_metrics:
                formatted += "  Class Metrics:\n"
                for metric, value in mode_metrics['class_metrics'].items():
                    formatted += f"    {metric.capitalize()}: {value:.4f}\n"
            if 'concept_metrics' in mode_metrics:
                formatted += "  Concept Metrics:\n"
                for metric, value in mode_metrics['concept_metrics'].items():
                    formatted += f"    {metric.capitalize()}: {value:.4f}\n"
            if 'loss_metrics' in mode_metrics:
                formatted += "  Loss Metrics:\n"
                for metric, value in mode_metrics['loss_metrics'].items():
                    formatted += f"    {metric.capitalize()}: {value:.4f}\n"
            if 'sailency_scores' in mode_metrics:
                formatted += "  Sailency Scores:\n"
                for metric, value in mode_metrics['sailency_scores'].items():
                    formatted += f"    {metric.capitalize()}: {value:.4f}\n"
            formatted += "\n"
        return formatted


class ConceptLogger:
    def __init__(self, concept_names: list[str], log_file: str = 'concept_metrics.json'):
        """
        Initialize the concept logger with a list of concept names.
        
        Args:
            concept_names: List of strings representing the names of concepts to track
            log_file: Path where the JSON results will be saved
        """
        self.concept_names = concept_names
        self.log_file = log_file
        self.reset()

    def reset(self):
        """Reset all accumulated data"""
        self.concept_stats = {
            concept: {
                'true_positives': 0,
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'total_samples': 0
            } for concept in self.concept_names
        }

    def update(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """
        Update statistics for all concepts based on predictions and ground truth.
        
        Args:
            predictions: Tensor of shape (batch_size, num_concepts) with predicted probabilities
            ground_truth: Tensor of shape (batch_size, num_concepts) with binary ground truth
        """
        predictions = predictions.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()
        
        # Convert predictions to binary (0/1) using 0.5 threshold
        predictions = (predictions >= 0.5).astype(bool)
        ground_truth = (ground_truth >= 0.5).astype(bool)
        
        for i, concept in enumerate(self.concept_names):
            pred = predictions[:, i]
            truth = ground_truth[:, i]
            
            self.concept_stats[concept]['true_positives'] += np.sum(np.logical_and(pred == 1, truth == 1))
            self.concept_stats[concept]['true_negatives'] += np.sum(np.logical_and(pred == 0, truth == 0))
            self.concept_stats[concept]['false_positives'] += np.sum(np.logical_and(pred == 1, truth == 0))
            self.concept_stats[concept]['false_negatives'] += np.sum(np.logical_and(pred == 0, truth == 1))
            self.concept_stats[concept]['total_samples'] += len(truth)

    def get_concept_metrics(self, concept: str) -> Dict[str, float]:
        """Calculate metrics for a specific concept"""
        stats = self.concept_stats[concept]
        tp = stats['true_positives']
        tn = stats['true_negatives']
        fp = stats['false_positives']
        fn = stats['false_negatives']
        total = stats['total_samples']
        
        # Calculate all metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        positive_ratio = (tp + fn) / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'positive_ratio': positive_ratio
        }

    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all concepts"""
        return {
            concept: self.get_concept_metrics(concept)
            for concept in self.concept_names
        }

    def save_metrics(self):
        """Save metrics to file in a human-readable format"""
        metrics = self.get_all_metrics()
        
        # Save to JSON file
        with open(self.log_file, 'w') as f:
            json.dump({
                "concepts": metrics
            }, f, indent=2)
        
        # Print formatted metrics
        #print("\nConcept Metrics:")
        #print(self.format_metrics(metrics))

    def format_metrics(self, metrics: Dict[str, Dict[str, float]]) -> str:
        """Format metrics for human-readable console output"""
        formatted = ""
        for concept, concept_metrics in metrics.items():
            formatted += f"Concept: {concept}\n"
            for metric, value in concept_metrics.items():
                formatted += f"  {metric.capitalize()}: {value:.4f}"
                if metric == 'positive_ratio':
                    formatted += f" ({value*100:.1f}% positive samples)"
                formatted += "\n"
            formatted += "\n"
        return formatted
'''
class Logger:
    def __init__(
        self,
        cfg,
        concept_mask: Optional[torch.Tensor] = None,
        concept_names: Optional[list[str]] = [],
        class_names: Optional[list[str]] = [],
        confusion_matrix = False
    ):
        self.use_wandb = cfg.logger.use_wandb
        self.training_mode = cfg.mode # 'Joint', 'Independent', or 'Sequential'
        self.current_phase = 'joint'  # Default to joint, can be 'concept' or 'class' for sequential
        self.start_time = datetime.now()
        
        # Initialize separate epoch data for each type
        self.ctoy_epochs_data = []
        self.xtoc_epochs_data = []
        
        # Load CUB dataset specific names if using CUB
        self.concept_names = concept_names
        self.class_names = class_names

        # Initialize confusion matrix if enabled
        if confusion_matrix:
            self.confusion_matrix_enabled = True
        else:
            self.confusion_matrix_enabled = False


        # Initialize W&B if enabled
        if self.use_wandb and not wandb.run:
            run_name = (cfg.logger.run_name if cfg.logger.run_name 
                    else f"{self.training_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            wandb.init(
                project=cfg.logger.project_name,
                name=run_name,
                group=cfg.logger.group,
                tags=cfg.logger.tags,
                config=OmegaConf.to_container(cfg, resolve=True)
            )
        
        self.reset()

    def set_phase(self, phase: str):
        """Set current training phase for sequential mode"""
        assert phase in ['concept', 'class'], "Phase must be either 'concept' or 'class'"
        
        self.current_phase = phase
        if self.use_wandb:
            wandb.log({"current_phase": phase})
        self.reset()  # Reset metrics for new phase

    def load_cub_names(self, cub_dir: str):

        """Load concept and class names from CUB dataset"""
        # Load concept names from attributes.txt
        attr_path = Path(cub_dir) / "attributes.txt"
        if attr_path.exists():
            with open(attr_path, 'r') as f:
                self.concept_names = [
                    line.strip().split(' ', 1)[1].strip() 
                    for line in f.readlines()
                ]
                
        # Load class names from classes.txt
        class_path = Path(cub_dir) / "classes.txt"
        if class_path.exists():
            with open(class_path, 'r') as f:
                self.class_names = [
                    line.strip().split(' ', 1)[1].strip() 
                    for line in f.readlines()
                ]
        

    def reset(self):
        """Reset all accumulated metrics"""
        self.metrics = defaultdict(lambda: defaultdict(list))
        self.concept_metrics = defaultdict(lambda: {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total': 0
        })
        self.class_metrics = defaultdict(lambda: {
            'correct': 0,
            'top5_correct': 0,
            'total': 0
        })
        
        # Add per-class and per-concept tracking
        self.per_class_metrics = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        self.per_concept_metrics = defaultdict(lambda: defaultdict(lambda: {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0
        }))

        if self.confusion_matrix_enabled:
            self.confusion_matrix = np.zeros((len(self.class_names), len(self.class_names)))


    def get_time_since_start(self):
        """Get time elapsed since logger initialization"""
        elapsed = datetime.now() - self.start_time
        return str(elapsed)

    def update_concept_accuracy(self, mode: str, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """Update concept prediction metrics"""
        predictions = (predictions.detach().cpu().numpy() >= 0.5)
        ground_truth = (ground_truth.detach().cpu().numpy() >= 0.5)
        
        stats = self.concept_metrics[mode]
        stats['total'] += predictions.shape[0] * predictions.shape[1]
        stats['true_positives'] += np.sum(np.logical_and(predictions == 1, ground_truth == 1))
        stats['true_negatives'] += np.sum(np.logical_and(predictions == 0, ground_truth == 0))
        stats['false_positives'] += np.sum(np.logical_and(predictions == 1, ground_truth == 0))
        stats['false_negatives'] += np.sum(np.logical_and(predictions == 0, ground_truth == 1))
        

        # Update per-concept metrics
        for idx,concept_name in enumerate(self.concept_names):
            concept_pred = predictions[:, idx]
            concept_truth = ground_truth[:, idx]
            
            stats = self.per_concept_metrics[mode][concept_name]
            stats['true_positives'] += np.sum(np.logical_and(concept_pred == 1, concept_truth == 1))
            stats['true_negatives'] += np.sum(np.logical_and(concept_pred == 0, concept_truth == 0))
            stats['false_positives'] += np.sum(np.logical_and(concept_pred == 1, concept_truth == 0))
            stats['false_negatives'] += np.sum(np.logical_and(concept_pred == 0, concept_truth == 1))

    def get_per_concept_accuracy(self, mode: str) -> Dict[str, Dict[str, float]]:
        """Calculate metrics for each concept in validation data"""
        concept_metrics = {}
        
        for concept_name in self.concept_names:
            stats = self.per_concept_metrics[mode][concept_name]
            total = (stats['true_positives'] + stats['true_negatives'] + 
                    stats['false_positives'] + stats['false_negatives'])
            
            if total > 0:
                # Calculate all metrics
                accuracy = (stats['true_positives'] + stats['true_negatives']) / total
                
                precision = (stats['true_positives'] / 
                           (stats['true_positives'] + stats['false_positives'])
                           if (stats['true_positives'] + stats['false_positives']) > 0 
                           else 0)
                
                recall = (stats['true_positives'] / 
                         (stats['true_positives'] + stats['false_negatives'])
                         if (stats['true_positives'] + stats['false_negatives']) > 0 
                         else 0)
                
                f1 = (2 * precision * recall / (precision + recall)
                      if (precision + recall) > 0 
                      else 0)
                

                
                concept_metrics[concept_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        
        return concept_metrics

    def update_class_accuracy(self, mode: str, logits: torch.Tensor, correct_label: torch.Tensor):
        """Update classification metrics"""
        logits = logits.detach().cpu().numpy()
        correct_label = correct_label.detach().cpu().numpy()
        
        top_predictions = np.argsort(logits, axis=1)[:, -5:]
        correct_classes = np.argmax(correct_label, axis=1)
        predictions = top_predictions[:, -1]
        
        self.class_metrics[mode]['total'] += logits.shape[0]
        self.class_metrics[mode]['correct'] += np.sum(predictions == correct_classes)
        self.class_metrics[mode]['top5_correct'] += np.sum([
            correct_class in top5 
            for correct_class, top5 in zip(correct_classes, top_predictions)
        ])
        
        # Update per-class metrics
        for pred, true_class in zip(predictions, correct_classes):
            if mode not in self.per_class_metrics:
                self.per_class_metrics[mode] = defaultdict(lambda: {'correct': 0, 'total': 0})
            self.per_class_metrics[mode][true_class]['total'] += 1
            if pred == true_class:
                self.per_class_metrics[mode][true_class]['correct'] += 1
        
        # Update confusion matrix
        if self.confusion_matrix_enabled:
            for pred, true_class in zip(predictions, correct_classes):
                self.confusion_matrix[true_class,pred] += 1

    def update_loss(self, mode: str, loss: float, loss_type: str = 'total'):
        """Update loss metrics with type (e.g., 'concept', 'class', 'total')"""
        self.metrics[mode][f"{loss_type}_loss"].append(loss.item())

    def get_loss_metrics(self, mode: str, loss_type: str = 'total') -> Dict[str, float]:
        """Get most recent loss for the specified mode and type"""
        return np.sum(self.metrics[mode][f"{loss_type}_loss"])

    def update_learning_rate(self, optimizer):
            """Track learning rate from optimizer"""
            self.optimizer = optimizer  # Store for later use
            
            # Don't log to wandb here anymore, just store the learning rates
            self.current_lrs = [group['lr'] for group in optimizer.param_groups]


    

    def get_class_metrics(self, mode: str) -> Dict[str, float]:
        """Calculate classification metrics"""
        stats = self.class_metrics[mode]
        if stats['total'] == 0:
            return {'top1_accuracy': 0, 'top5_accuracy': 0}
            
        return {
            'top1_accuracy': stats['correct'] / stats['total'],
            'top5_accuracy': stats['top5_correct'] / stats['total']
        }

    def get_concept_metrics(self, mode: str) -> Dict[str, float]:
        """Calculate concept prediction metrics"""
        stats = self.concept_metrics[mode]
        total = stats['true_positives'] + stats['true_negatives'] + stats['false_positives'] + stats['false_negatives']
        
        if total == 0:
            return {'accuracy': 0, 'f1': 0}
            
        accuracy = (stats['true_positives'] + stats['true_negatives']) / total
        
        precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if (stats['true_positives'] + stats['false_positives']) > 0 else 0
        recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if (stats['true_positives'] + stats['false_negatives']) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def log_metrics(self, epoch: int, optimizer):
        """Log metrics based on current training phase"""
        timestamp = self.get_time_since_start()
        wandb_dict = {}
        
        if optimizer is not None:
            wandb_dict["learning_rate"] = optimizer.param_groups[0]['lr']

        # Log concept metrics if training concepts
        if self.training_mode == 'Joint' or self.current_phase == 'concept':
            for mode in ['train', 'val']:
                if self.concept_metrics[mode]['total'] > 0:
                    metrics = self.get_concept_metrics(mode)
                    for name, value in metrics.items():
                        wandb_dict[f'concepts/{mode}/{name}'] = float(value)
                    wandb_dict[f'concepts/{mode}/loss'] = self.get_loss_metrics(mode, "concept")
            
            metrics_dict = {
                "epoch": epoch,
                "timestamp": timestamp,
                "metrics": {
                    mode: {
                        "concept_metrics": self.get_concept_metrics(mode),
                        "loss_metrics": {"loss": self.get_loss_metrics(mode, "concept")}
                    }
                    for mode in ['train', 'val']
                    if self.concept_metrics[mode]['total'] > 0
                }
            }
            self.xtoc_epochs_data.append(metrics_dict)

        # Log class metrics if training classes
        if self.training_mode == 'Joint' or self.current_phase == 'class':
            for mode in ['train', 'val']:
                if self.class_metrics[mode]['total'] > 0:
                    metrics = self.get_class_metrics(mode)
                    for name, value in metrics.items():
                        wandb_dict[f'classes/{mode}/{name}'] = float(value)
                    wandb_dict[f'classes/{mode}/loss'] = self.get_loss_metrics(mode, "class")

            metrics_dict = {
                "epoch": epoch,
                "timestamp": timestamp,
                "metrics": {
                    mode: {
                        "class_metrics": self.get_class_metrics(mode),
                        "loss_metrics": {"loss": self.get_loss_metrics(mode, "class")}
                    }
                    for mode in ['train', 'val']
                    if self.class_metrics[mode]['total'] > 0
                }
            }
            self.ctoy_epochs_data.append(metrics_dict)

        if self.training_mode == 'Joint':
            # Log total loss for joint training
            for mode in ['train', 'val']:
                wandb_dict[f'loss/total/{mode}'] = self.get_loss_metrics(mode, "total")

        if self.use_wandb:
            wandb.log(wandb_dict)

        # Save logs
        if self.xtoc_epochs_data:
            with open(os.path.join(HydraConfig.get().run.dir, "XtoC_log.json"), 'w') as f:
                json.dump(self.xtoc_epochs_data, f, indent=2)
        
        if self.ctoy_epochs_data:
            with open(os.path.join(HydraConfig.get().run.dir, "CtoY_log.json"), 'w') as f:
                json.dump(self.ctoy_epochs_data, f, indent=2)

        self.reset()

    def validate(self, dir: str,sailency_score=None):
        # Gather all metrics
        metrics = {
                "NoMajority": self.get_concept_metrics("NoMajority"),
                "Majority": self.get_concept_metrics("Majority"),
                "Class": self.get_class_metrics("test"),
                "Sailency": {"score":sailency_score}
                
            }

        #Log induvidual concept metrics
        NoMajority = self.get_per_concept_accuracy("NoMajority")
        Majority = self.get_per_concept_accuracy("Majority")

        #For each concept log both modes
        if self.training_mode != 'Standard':
            for concept in self.concept_names:
                concept_metrics = {}

                for type in ["accuracy","precision","recall","f1"]:
                    concept_metrics[f"NoMajority_{type}"] = NoMajority[concept][type]
                    concept_metrics[f"Majority_{type}"] = Majority[concept][type]
                metrics[concept] = concept_metrics
        

        # Log metrics to WandB
        if self.use_wandb:
            for mode, mode_metrics in metrics.items():
                if isinstance(mode_metrics, dict):
                    for k, v in mode_metrics.items():
                        wandb.log({f"{mode}/{k}": v}, step=0)
                else:
                    wandb.log({mode: mode_metrics}, step=0)
        




        # Save metrics locally
        output_dir = Path(dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    def finish(self):
        """Cleanup logging"""
        if self.use_wandb:
            wandb.finish()
