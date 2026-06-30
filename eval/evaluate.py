from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss

def evaluate_model(y_true, y_pred_prob):
    """
    Evaluates the model using metrics suited for imbalanced CTR data.
    """
    results = {}
    
    # 1. AUC-ROC: Ranking ability
    try:
        results['auc'] = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        results['auc'] = 0.5
        
    # 2. PR-AUC: Honest score for rare events
    try:
        results['pr_auc'] = average_precision_score(y_true, y_pred_prob)
    except ValueError:
        results['pr_auc'] = 0.0
        
    # 3. Log Loss: Probability quality
    results['log_loss'] = log_loss(y_true, y_pred_prob, labels=[0, 1])
    
    # 4. Calibration / Brier Score
    results['brier_score'] = brier_score_loss(y_true, y_pred_prob)
    
    return results

def print_evaluation(results, model_name="Model"):
    print(f"--- Evaluation for {model_name} ---")
    print(f"AUC-ROC:     {results['auc']:.4f}")
    print(f"PR-AUC:      {results['pr_auc']:.4f}")
    print(f"Log Loss:    {results['log_loss']:.4f}")
    print(f"Brier Score: {results['brier_score']:.4f}")
    print("-" * 35)

if __name__ == '__main__':
    print("Evaluation module ready.")
