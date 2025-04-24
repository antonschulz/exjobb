from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, classification_report,  confusion_matrix

def evaluate_model(model, val_data):
    """
    A placeholder evaluation function.
    Replace this with your actual evaluation routine that returns a performance metric,
    for example, accuracy on 'val_data'.
    """
    # For example, if using scikit-learn or a PyTorch model wrapped with a scorer,
    # you might run predictions and then compute the metric.
    # Here we'll assume the model has a 'predict' method and that the validation data
    # is provided as a tuple (inputs, true_labels).
    inputs, true_labels = val_data.data, val_data.labels
    predictions = model.predict(val_data)
    metrics = {}
    
    # Subset accuracy (exact match accuracy)
    metrics["accuracy"] = accuracy_score(true_labels, predictions)
    
    # Hamming loss (fraction of wrong labels per sample)
    metrics["hamming_loss"] = hamming_loss(true_labels, predictions)
    
    # Micro-averaged metrics aggregate contributions from all labels.
    metrics["precision_micro"] = precision_score(true_labels, predictions, average='micro', zero_division=0)
    metrics["recall_micro"] = recall_score(true_labels, predictions, average='micro', zero_division=0)
    metrics["f1_micro"] = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    # Macro-averaged metrics calculate the metric for each label and average them.
    metrics["precision_macro"] = precision_score(true_labels, predictions, average='macro', zero_division=0)
    metrics["recall_macro"] = recall_score(true_labels, predictions, average='macro', zero_division=0)
    metrics["f1_macro"] = f1_score(true_labels, predictions, average='macro', zero_division=0)

        # Generate the text report
    report = classification_report(true_labels, predictions, zero_division=0, output_dict=True)
    metrics["classification_report"] = report

     # confusion matrix as nested dict
    # determine sorted labels
    labels = sorted(set(true_labels) | set(predictions))
    cm = confusion_matrix(true_labels, predictions, labels=labels)
    cm_dict = {
        int(true_label): {
            int(pred_label): int(cm[i, j])
            for j, pred_label in enumerate(labels)
        }
        for i, true_label in enumerate(labels)
    }
    metrics["confusion_matrix"] = cm_dict


    
    # Optionally, you can also compute additional metrics here.
    return metrics