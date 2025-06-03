import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def bootstrap_auc(y_true, y_pred_proba, n_bootstrap=100, random_state=42):
    """
    Calculate AUC with confidence intervals using bootstrap resampling.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    n_bootstrap : int, default=100
        Number of bootstrap iterations
    random_state : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    dict : Dictionary containing AUC statistics
        - 'auc': Original AUC score
        - 'auc_mean': Mean AUC from bootstrap samples
        - 'auc_std': Standard deviation of bootstrap AUCs
        - 'auc_ci_lower': Lower bound of 95% confidence interval
        - 'auc_ci_upper': Upper bound of 95% confidence interval
    """
    # Calculate original AUC
    original_auc = roc_auc_score(y_true, y_pred_proba)

    # Bootstrap sampling
    np.random.seed(random_state)
    bootstrap_aucs = []

    for i in range(n_bootstrap):
        # Resample with replacement
        indices = resample(
            range(len(y_true)),
            n_samples=len(y_true),
            random_state=random_state + i,
        )

        y_boot = np.array(y_true)[indices]
        y_pred_boot = np.array(y_pred_proba)[indices]

        # Calculate AUC for bootstrap sample
        try:
            boot_auc = roc_auc_score(y_boot, y_pred_boot)
            bootstrap_aucs.append(boot_auc)
        except ValueError:
            # Skip if bootstrap sample has only one class
            continue

    bootstrap_aucs = np.array(bootstrap_aucs)

    # Calculate statistics
    auc_mean = np.mean(bootstrap_aucs)
    auc_std = np.std(bootstrap_aucs)
    auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
    auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return {
        "auc": original_auc,
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
    }
