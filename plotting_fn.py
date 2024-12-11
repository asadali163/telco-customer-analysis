import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrices(results, class_names):
    """
    Plots confusion matrices for multiple models in a single figure with alternating colors and better spacing.
    
    Parameters:
        results (list of dict): List of model evaluation results containing confusion matrices and model names.
        class_names (list): List of class names for the confusion matrix.
    """
    num_models = len(results)
    cols = 2
    rows = (num_models + 1) // cols 
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
    axes = axes.ravel() 
    
    color_maps = ['Blues', 'Greens', 'Oranges', 'Purples']
    for i, result in enumerate(results):
        cm = result['confusion_matrix']
        model_name = result['model']
        cmap = color_maps[i % len(color_maps)]  # Alternate colormap
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=axes[i],
                    xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[i].set_title(f"Confusion Matrix: {model_name}", fontsize=14)
        axes[i].set_xlabel("Predicted Labels", fontsize=12)
        axes[i].set_ylabel("True Labels", fontsize=12)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


def plot_testing_accuracies(results):
    """
    Plots Testing Accuracies for multiple models in a bar chart.
    
    Parameters:
        results (list of dict): List of model evaluation results containing testing accuracies.
    """
    # Extract model names and testing accuracies
    model_names = [result['model'] for result in results]
    testing_accuracies = [result['testing_acc'] for result in results]
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, testing_accuracies, color='skyblue')
    
    # Add labels and title
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Testing Accuracy', fontsize=12)
    ax.set_title('Model Performance: Testing Accuracy', fontsize=14)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_f1_scores(results):
    """
    Plots F1 Scores for multiple models in a bar chart.
    
    Parameters:
        results (list of dict): List of model evaluation results containing F1 scores.
    """
    # Extract model names and F1 scores
    model_names = [result['model'] for result in results]
    f1_scores = [result['f1_score'] for result in results]
    
    # Plot bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(model_names, f1_scores, color='salmon')
    
    # Add labels and title
    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Model Performance: F1 Score', fontsize=14)
    
    # Annotate bars with values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset text above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()


# For logistic Regression
def feature_weights(X_df, classifier, classifier_name):
    weights = pd.Series(classifier.coef_[0], index = X_df.columns.values).sort_values(ascending=False)
    
    # Let's make 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(f'{classifier_name} Feature Weights', fontsize=15, fontweight='bold')
    top_10_weights = weights.head(10)
    bottom_10_weights = weights.tail(10)
    top_10_weights.plot(kind='bar', ax=axes[0], color='skyblue')
    bottom_10_weights.plot(kind='bar', ax=axes[1], color='salmon')
    axes[0].set_title('Top 10 Features')
    axes[1].set_title('Bottom 10 Features')
    plt.show()