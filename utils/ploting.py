import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_results(results, cfg, mode, plots_dir):
    """
    Plot and save the training results in a single column of 3 rows.

    Args:
        results (dict): Dictionary containing lists of losses and accuracies over epochs.
        cfg (DictConfig): Configuration object.
        mode (str): The training mode used (e.g., 'independent', 'joint').
        plots_dir (Path): Directory to save the plots.
    """
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(exist_ok=True)

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(f'{mode.capitalize()} - Training Results', fontsize=16)

    # Plot 1: Train and validation loss on class
    if results['train_losses'].get('class'):
        axs[0].plot(results['train_losses']['class'], label='Train Loss (Class)')
    if results['val_losses'].get('class'):
        axs[0].plot(results['val_losses']['class'], label='Validation Loss (Class)')
    if results['train_losses'].get('concept'):
        axs[0].plot(results['train_losses']['concept'], label='Train Loss (Concept)')
    if results['val_losses'].get('concept'):
        axs[0].plot(results['val_losses']['concept'], label='Validation Loss (Concept)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Losses')
    axs[0].legend()

    # Plot 2: Train loss on class and concept
    if results['train_losses'].get('class'):
        axs[1].plot(results['train_losses']['class'], label='Class')
    if results['train_losses'].get('concept'):
        axs[1].plot(results['train_losses']['concept'], label='Concept')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Train Loss')
    axs[1].set_title('Training Losses')
    axs[1].legend()

    # Plot 3: Validation accuracy on both concept and class
    if results['val_accuracies'].get('class'):
        axs[2].plot(results['val_accuracies']['class'], label='Class')
    if results['val_accuracies'].get('concept'):
        axs[2].plot(results['val_accuracies']['concept'], label='Concept')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Validation Accuracy')
    axs[2].set_title('Validation Accuracies')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(plots_dir / f"{mode}_training_results.png")
    plt.close()

    print(f"Plot saved in {plots_dir}")

def plot_confusion_matrix(cm, classes, output_dir):
    """
    Plot and save the confusion matrix.

    Args:
        cm (np.array): Confusion matrix.
        classes (list): List of class names.
        output_dir (Path): Directory to save the plot.
    """
    plt.figure(figsize=(30, 30))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=5)
    plt.yticks(tick_marks, classes, fontsize=5)

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() 
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    print(f"Confusion matrix saved in {output_dir}")