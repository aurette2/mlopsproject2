import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


# dice loss as defined above for 4 classes
def dice_coef(y_true, y_pred, smooth=1.0):
    class_num = 4
    for i in range(class_num):
        y_true_f = K.flatten(y_true[:,:,:,i])
        y_pred_f = K.flatten(y_pred[:,:,:,i])
        intersection = K.sum(y_true_f * y_pred_f)
        loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
        if i == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss
    total_loss = total_loss / class_num
    return total_loss

# Define per class evaluation of dice coef
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,1] * y_pred[:,:,:,1]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,1])) + K.sum(K.square(y_pred[:,:,:,1])) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,2] * y_pred[:,:,:,2]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,2])) + K.sum(K.square(y_pred[:,:,:,2])) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = K.sum(K.abs(y_true[:,:,:,3] * y_pred[:,:,:,3]))
    return (2. * intersection) / (K.sum(K.square(y_true[:,:,:,3])) + K.sum(K.square(y_pred[:,:,:,3])) + epsilon)


# Computing Precision
def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision


# Computing Sensitivity
def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


# Computing Specificity
def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1-y_true) * (1-y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1-y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def plot_training_history(history_df):
    """
    Plots training history including accuracy, loss, dice coefficient, and mean IoU.

    Parameters:
    - history_df: A pandas DataFrame containing the training history with columns:
                  'accuracy', 'val_accuracy', 'loss', 'val_loss', 'dice_coef',
                  'val_dice_coef', 'mean_io_u', 'val_mean_io_u'.

    Returns:
    - None
    """
    # Extract metrics
    acc = history_df['accuracy']
    val_acc = history_df['val_accuracy']
    epoch = range(len(acc))
    
    loss = history_df['loss']
    val_loss = history_df['val_loss']
    
    train_dice = history_df['dice_coef']
    val_dice = history_df['val_dice_coef']
    
    mean_io_u = history_df['mean_io_u']
    val_mean_io_u = history_df['val_mean_io_u']
    
    # Create subplots
    f, ax = plt.subplots(1, 4, figsize=(16, 8))

    # Plot accuracy
    ax[0].plot(epoch, acc, 'b', label='Training Accuracy')
    ax[0].plot(epoch, val_acc, 'r', label='Validation Accuracy')
    ax[0].legend()
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')

    # Plot loss
    ax[1].plot(epoch, loss, 'b', label='Training Loss')
    ax[1].plot(epoch, val_loss, 'r', label='Validation Loss')
    ax[1].legend()
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')

    # Plot dice coefficient
    ax[2].plot(epoch, train_dice, 'b', label='Training Dice Coefficient')
    ax[2].plot(epoch, val_dice, 'r', label='Validation Dice Coefficient')
    ax[2].legend()
    ax[2].set_title('Dice Coefficient')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Dice Coefficient')

    # Plot mean IoU
    ax[3].plot(epoch, mean_io_u, 'b', label='Training Mean IoU')
    ax[3].plot(epoch, val_mean_io_u, 'r', label='Validation Mean IoU')
    ax[3].legend()
    ax[3].set_title('Mean IoU')
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Mean IoU')

    # Show plots
    plt.tight_layout()
    plt.show()
    
    # Example usage
    # history = pd.read_csv('training.log', sep=',', engine='python')
    # plot_training_history(history)