import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch

### plotting the confusion matrix
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(predicted_labels, true_labels, classes):
    """
    Plot confusion matrix.
    
    Parameters:
    predicted_labels : numpy array
        Predicted labels.
    true_labels : numpy array
        True labels.
    classes : list
        List of class labels.
    """
    # Create the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add counts in the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    ### code to plot the results of training (like epoch losses and accuracy)
def plot_losses_trained_model(trainer):
    
    # Plotting losses
    train_loss = trainer.avg_train_loss_epoch
    val_loss = trainer.avg_val_loss_epoch
    plt.plot(train_loss, "o-", label="train_loss")
    plt.plot(val_loss, ">:", label="val_loss")
    plt.legend()
    plt.show()

    # Plotting train and validation accuracy
    train_acc, val_acc = trainer.train_acc_epoch, trainer.val_acc_epoch
    plt.plot(train_acc, "o-", label="train_accuracy")
    plt.plot(val_acc, ">:", label="val_accuracy")
    plt.yscale("log")
    plt.legend()
    plt.show()

    # Printing the final training and validation accuracy
    print(f"Final training accuracy is: {train_acc[-1]:0.3f}")
    print(f"Final validation accuracy is: {val_acc[-1]:0.3f}")


### function to evaluate the trained model
def evaluate_model(trained_model_input, y_test_fn, dataset_size=100, noise_level=0.003, 
                    want_confusion_matrix=False,
                    print_preds=False, range_of_value=1):
    # Generate test dataset
    x_test = np.random.uniform(-range_of_value, range_of_value, size=(dataset_size, 2)) + noise_level * np.ones(shape=(dataset_size, 2))
    y_test = y_test_fn(x_test)

    # Convert to torch tensors
    x_test, y_test = torch.tensor(x_test), torch.tensor(y_test)

    # Make predictions
    with torch.no_grad():
        preds = trained_model_input(x_test)
        if print_preds:
            print("predictions are:")
            print(preds)

    # Calculate accuracy
    acc_test = (preds.squeeze().round() == y_test).float().mean()
    print(f"Test accuracy: {acc_test.item():.4f}")
    # Plot confusion matrix
    if want_confusion_matrix:
        plot_confusion_matrix(preds.squeeze().round(), y_test, classes=["0", "1"])
