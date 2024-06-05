import numpy as np


def impurity_func(criterion, input) -> float:
    if criterion == "gini":
        return gini(*input)
    elif criterion == "log_loss":
        return log_loss(*input)
    else:
        raise ValueError("Unsupported criterion")
    
def gini(num_samples, counts):
    if num_samples == 0:
        return 0.0
    
    impurity = 1.0
    for count in counts:
        if count > 0:
            prob = count / num_samples
            impurity -= prob ** 2
    return impurity

def log_loss(num_samples, counts):
    if num_samples == 0:
        return 0.0
    
    impurity = 0.0
    total = sum(counts)
    for count in counts:
        if count > 0:
            prob = count / total
            impurity -= prob * np.log2(prob)

    return impurity