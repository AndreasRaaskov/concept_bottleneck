import json
import matplotlib.pyplot as plt
import os

def save_training_metrics(log_file_path,output_dir=""):
    """
    Read training log JSON and save separate plots with subplots including loss metrics.
    """

    # Read JSON file
    with open(log_file_path, 'r') as f:
        data = json.load(f)
    
    epochs = [entry['epoch'] for entry in data]
    
    # Handle class metrics if they exist
    if 'class_metrics' in data[0]['metrics']['train']:
        class_metrics = list(data[0]['metrics']['train']['class_metrics'].keys())
        n_metrics = len(class_metrics)
        
        # Create figure with loss plot at top plus a subplot for each class metric
        fig, axes = plt.subplots(n_metrics + 1, 1, figsize=(15, 5*(n_metrics + 1)))
        if n_metrics + 1 == 1:
            axes = [axes]
            
        # Plot loss at the top
        train_loss = [entry['metrics']['train']['loss_metrics'].get('loss',0) for entry in data]
        val_loss = [entry['metrics']['val']['loss_metrics'].get('loss',0) for entry in data]
        
        axes[0].plot(epochs, train_loss, '-o', label='Train Loss')
        axes[0].plot(epochs, val_loss, '--s', label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
            
        # Plot class metrics
        for ax, metric in zip(axes[1:], class_metrics):
            train_values = [entry['metrics']['train']['class_metrics'].get(metric,0) for entry in data]
            val_values = [entry['metrics']['val']['class_metrics'].get(metric,0) for entry in data]
            
            ax.plot(epochs, train_values, '-o', label=f'Train {metric}')
            ax.plot(epochs, val_values, '--s', label=f'Val {metric}')
            ax.set_title(f'{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'class_train_plot.png'))
        plt.close()
    
    # Handle concept metrics if they exist
    if 'concept_metrics' in data[0]['metrics']['train']:
        concept_metrics = list(data[0]['metrics']['train']['concept_metrics'].keys())
        n_metrics = len(concept_metrics)
        
        # Create figure with loss plot at top plus a subplot for each concept metric
        fig, axes = plt.subplots(n_metrics + 1, 1, figsize=(15, 5*(n_metrics + 1)))
        if n_metrics + 1 == 1:
            axes = [axes]
            
        # Plot loss at the top
        train_loss = [entry['metrics']['train']['loss_metrics'].get('loss',0)for entry in data]
        val_loss = [entry['metrics']['val']['loss_metrics'].get('loss',0) for entry in data]
        
        axes[0].plot(epochs, train_loss, '-o', label='Train Loss')
        axes[0].plot(epochs, val_loss, '--s', label='Val Loss')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot concept metrics
        for ax, metric in zip(axes[1:], concept_metrics):
            train_values = [entry['metrics']['train']['concept_metrics'].get(metric,0) for entry in data]
            val_values = [entry['metrics']['val']['concept_metrics'].get(metric,0) for entry in data]
            
            ax.plot(epochs, train_values, '-o', label=f'Train {metric}')
            ax.plot(epochs, val_values, '--s', label=f'Val {metric}')
            ax.set_title(f'{metric}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir,'concept_train_plot.png'))
        plt.close()

if __name__ == "__main__":
    save_training_metrics('test_log.json')