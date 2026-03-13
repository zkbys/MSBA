import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
    auc
)
import warnings

# Suppress warnings for zero division if any class is missing in standard calculations
warnings.filterwarnings('ignore', category=UserWarning)

def generate_latex_table(macro_auc, macro_f1, balanced_acc, class_metrics, class_names):
    """
    Generate a LaTeX formatted table for the evaluation metrics.
    """
    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += "\\caption{Evaluation Metrics for MIMIC-CXR 6-Class Classification}\n"
    latex_str += "\\label{tab:evaluation_metrics}\n"
    latex_str += "\\begin{tabular}{l c c c}\n"
    latex_str += "\\hline\n"
    latex_str += "\\textbf{Metric / Class} & \\textbf{Precision} & \\textbf{Recall (Sensitivity)} & \\textbf{AUC (OvR)} \\\\\n"
    latex_str += "\\hline\n"
    
    # Per-class metrics
    for name in class_names:
        metrics = class_metrics[name]
        latex_str += f"{name} & {metrics['precision']:.4f} & {metrics['recall']:.4f} & {metrics['auc']:.4f} \\\\\n"
    
    latex_str += "\\hline\n"
    # Overall metrics
    latex_str += f"\\textbf{{Macro Average}} & - & - & \\textbf{{{macro_auc:.4f}}} \\\\\n"
    latex_str += f"\\textbf{{Macro F1 Score}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{macro_f1:.4f}}}}} \\\\\n"
    latex_str += f"\\textbf{{Balanced Accuracy}} & \\multicolumn{{3}}{{c}}{{\\textbf{{{balanced_acc:.4f}}}}} \\\\\n"
    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}"
    
    return latex_str

def evaluate_performance(y_true, y_probs, class_names, save_roc_path="macro_roc_curve.png"):
    """
    Evaluate model performance focusing on class imbalance and output LaTeX table and ROC curves.
    
    Args:
        y_true (array-like): Ground truth labels (integer indices 0 to C-1). shape (n_samples,)
        y_probs (array-like): Predicted probabilities. shape (n_samples, n_classes)
        class_names (list): List of class names.
        save_roc_path (str): File path to save the ROC curve image.
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    n_classes = len(class_names)
    
    # Get predictions by taking the argmax of probabilities
    y_pred = np.argmax(y_probs, axis=1)
    
    # 1. Macro-AUC (One-vs-Rest)
    # y_probs must be of shape (n_samples, n_classes)
    macro_auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    
    # 2. Macro-F1 Score
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # 3. Balanced Accuracy (Arithmetic mean of recalls)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # 4. Per-class Analysis (Precision, Recall)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # Binarize labels for per-class AUC & ROC calculation
    # y_true_bin[i, c] is 1 if y_true[i] == c, else 0
    # shape: (n_samples, n_classes)
    # Manually creating one-hot encoding
    y_true_bin = np.zeros_like(y_probs)
    y_true_bin[np.arange(len(y_true)), y_true] = 1
    
    class_metrics = {}
    
    # Data structures for ROC plotting
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i, name in enumerate(class_names):
        # Calculate AUC per class for OvR
        if len(np.unique(y_true_bin[:, i])) > 1:
            class_auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
        else:
            class_auc = float('nan') # Handle edge case where a class has no positive instances in y_true
            
        class_metrics[name] = {
            'precision': report[name]['precision'],
            'recall': report[name]['recall'],
            'auc': class_auc
        }
        
        # ROC Curve points for each class
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # --- Plotting ROC Curves ---
    plt.figure(figsize=(10, 8))
    
    # Plot Macro-average ROC
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
             color='navy', linestyle=':', linewidth=4)
    
    # Plot individual class ROC curves (optional setting, can be disabled)
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i, name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'{name} (AUC = {roc_auc[i]:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) to Multi-Class (OvR)', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_roc_path, dpi=300)
    print(f"\n[INFO] Macro-average ROC curve saved to {save_roc_path}")
    plt.close()

    # --- Print Text Summary ---
    print("\n" + "="*50)
    print(" " * 15 + "EVALUATION SUMMARY")
    print("="*50)
    print(f"Macro-AUC (OvR)    : {macro_auc:.4f}")
    print(f"Macro-F1 Score     : {macro_f1:.4f}")
    print(f"Balanced Accuracy  : {balanced_acc:.4f}\n")
    
    print("Per-class Metrics:")
    print("-" * 50)
    print(f"{'Class Name':<20} | {'Precision':<9} | {'Recall':<9} | {'AUC':<9}")
    print("-" * 50)
    for name in class_names:
        metrics = class_metrics[name]
        print(f"{name:<20} | {metrics['precision']:<9.4f} | {metrics['recall']:<9.4f} | {metrics['auc']:<9.4f}")
    print("-" * 50)
    
    # Generate and print LaTeX Table
    latex_table = generate_latex_table(macro_auc, macro_f1, balanced_acc, class_metrics, class_names)
    print("\n\n" + "="*50)
    print(" " * 12 + "LATEX FORMATTED TABLE")
    print("="*50)
    print(latex_table)
    
    return {
        "macro_auc": macro_auc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
        "class_metrics": class_metrics
    }

if __name__ == "__main__":
    # --- Dummy Data Generation for Demonstration ---
    print("Generating simulated predictions for demonstration...")
    np.random.seed(42)
    n_samples = 1000
    
    class_names = [
        "No Finding", 
        "Atelectasis", 
        "Cardiomegaly", 
        "Consolidation", 
        "Edema", 
        "Pleural Effusion"
    ]
    n_classes = len(class_names)
    
    # Simulate a class imbalanced distribution
    # E.g. No Finding is majority, Consolidation is rare
    class_probabilities = [0.4, 0.15, 0.1, 0.05, 0.1, 0.2]
    y_true_dummy = np.random.choice(n_classes, size=n_samples, p=class_probabilities)
    
    # Simulate model predictions (probabilities)
    # Give the model reasonable power by adding a bias towards the true class
    y_probs_dummy = np.random.rand(n_samples, n_classes)
    for i in range(n_samples):
        true_c = y_true_dummy[i]
        y_probs_dummy[i, true_c] += np.random.rand() * 2.0  # Boost the correct class
    
    # Normalize to make them probabilities summing to 1
    y_probs_dummy = y_probs_dummy / y_probs_dummy.sum(axis=1, keepdims=True)
    
    # --- Run Evaluation ---
    evaluate_performance(
        y_true=y_true_dummy, 
        y_probs=y_probs_dummy, 
        class_names=class_names,
        save_roc_path="mimic_cxr_macro_roc.png"
    )
