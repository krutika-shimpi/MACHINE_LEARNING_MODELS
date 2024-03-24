# Define a function to plot the loss curves
import matplotlib.pyplot as plt
def plot_loss_curves(history, regression = None, classification = None):
  """
  This function plots the loss curves by accepting an input parameter history.
  """

  plt.figure(figsize = (12,6))

  if classification == True and regression != True:

    # Find the losses and accuracies
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))

    # Plot the loss and accuracy curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label = 'Training loss')
    plt.plot(epochs, val_loss, label = 'Validation loss')
    plt.title('Training Vs. Validation loss curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label = 'Training accuracy')
    plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
    plt.title('Training Vs. Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

  elif regression == True and classification != True:
    # Find the losses and metric
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(len(history.history['loss']))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label = 'Training loss')
    plt.plot(epochs, val_loss, label = 'Validation loss')
    plt.title('Training Vs. Validation loss curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, mae, label = 'Training error')
    plt.plot(epochs, val_mae, label = 'Validation error')
    plt.title('Training Vs. Validation error')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

  plt.tight_layout()
  plt.show()