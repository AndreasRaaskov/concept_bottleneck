
"""
Provides a class to track training metrics and log them to a JSON file
"""

import os
import sys
import numpy as np
import torch
import json
from collections import defaultdict
from typing import Dict, Any
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score, precision_score, recall_score, balanced_accuracy_score, classification_report
from collections import defaultdict as ddict

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

'''
class Logger(object):
    """
    Log results to a file and flush() to view instant updates
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

# Legacy code below here
class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output and target are Torch tensors
    """

    output.to('cpu')
    target.to('cpu')
    
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    pred = pred.to('cpu')

    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res #about:blank#blocked

def binary_accuracy(output, target):
    """
    Computes the accuracy for multiple binary predictions
    output and target are Torch tensors
    """
    pred = output.cpu() >= 0.5
    #print(list(output.data.cpu().numpy()))
    #print(list(pred.data[0].numpy()))
    #print(list(target.data[0].numpy()))
    #print(pred.size(), target.size())
    acc = (pred.int()).eq(target.int()).sum()
    acc = acc*100 / np.prod(np.array(target.size()))
    return acc

def multiclass_metric(output, target):
    """
    Return balanced accuracy score (average of recall for each class) in case of class imbalance,
    and classification report containing precision, recall, F1 score for each class
    """
    balanced_acc = balanced_accuracy_score(target, output)
    report = classification_report(target, output)
    return balanced_acc, report
'''