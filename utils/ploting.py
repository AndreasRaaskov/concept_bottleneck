import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
import seaborn as sns
from pathlib import Path

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

def normalize_rows(mat):
    """Normalize each row of a matrix so that it sums to 100 (percentages)."""
    norm_mat = np.zeros_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        s = np.sum(mat[i])
        if s > 0:
            norm_mat[i] = mat[i] / s * 100
    return norm_mat

def plot_family_confusion_matrix(cm, class_names, output_path=None):
    """
    Aggregates the confusion matrix by family and plots it.
    
    For each family, we:
      1. Aggregate the confusion counts from the original matrix.
      2. Compute a "family confusion matrix" by extracting from the original confusion matrix
         all rows and columns corresponding to classes in that family.
      3. Compute the accuracy for that family as
            Accuracy = (sum of diagonal of family submatrix) / (sum of submatrix) * 100.
         
    The aggregated matrix is then plotted (with rows normalized to percentages),
    and on the right side of each row, the accuracy (computed from the family submatrix)
    is annotated.
    
    The figure is sized 30 x 30 inches.
    """
    # --- Step 1: Parse family names and build mapping family -> list of indices ---
    families = []
    family_to_indices = {}
    for i, name in enumerate(class_names):
        # Remove numeric prefix if present (e.g. "001.")
        if '.' in name:
            name = name.split('.', 1)[1]
        tokens = name.split('_')
        # Assume the last token is the family
        family = tokens[-1] if len(tokens) > 1 else tokens[0]
        families.append(family)
        if family not in family_to_indices:
            family_to_indices[family] = []
        family_to_indices[family].append(i)
    
    # Get sorted list of unique families
    unique_fams = sorted(set(families))
    n_fams = len(unique_fams)
    family_to_index = {fam: idx for idx, fam in enumerate(unique_fams)}
    
    # --- Step 2: Create the aggregated confusion matrix ---
    # Rows: true family; Columns: predicted family
    agg_cm = np.zeros((n_fams, n_fams), dtype=cm.dtype)
    for i in range(len(class_names)):
        true_fam = families[i]
        for j in range(len(class_names)):
            pred_fam = families[j]
            agg_cm[family_to_index[true_fam], family_to_index[pred_fam]] += cm[i, j]
    
    # --- Step 3: Compute the accuracy for each family's confusion matrix ---
    # For each family, extract the submatrix (classes belonging to that family)
    # and compute accuracy = (sum(diagonal) / sum(all entries)) * 100.
    family_accuracies = {}
    for fam in unique_fams:
        indices = family_to_indices[fam]
        sub_cm = cm[np.ix_(indices, indices)]
        total = sub_cm.sum()
        correct = np.trace(sub_cm)
        if total > 0:
            acc = correct / total * 100
        else:
            acc = 0.0
        family_accuracies[fam] = acc
    
    # --- Step 4: Normalize aggregated confusion matrix rows for heatmap display ---
    norm_agg_cm = normalize_rows(agg_cm)
    
    # --- Step 5: Plotting ---
    plt.figure(figsize=(22, 22))
    ax = sns.heatmap(norm_agg_cm, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                     xticklabels=unique_fams, yticklabels=unique_fams,
                     linewidths=1, linecolor='black')
    
    # Increase font sizes for tick labels
    ax.set_xticklabels(unique_fams, rotation=90, fontsize=20)
    ax.set_yticklabels(unique_fams, rotation=0, fontsize=20)
    
    plt.xlabel("Predicted Family", fontsize=24)
    plt.ylabel("True Family", fontsize=24)
    plt.title("Family Confusion Matrix (in %)", fontsize=28)
    
    # Annotate each row on the right with the family accuracy computed from its submatrix
    for i, fam in enumerate(unique_fams):
        acc = family_accuracies[fam]
        # Place the text just to the right of the heatmap (x position = n_fams + offset)
        ax.text(n_fams + 0.3, i + 0.5, f"{acc:.0f}%", 
                va='center', ha='left', fontsize=20, color='black')
    
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    plt.show()