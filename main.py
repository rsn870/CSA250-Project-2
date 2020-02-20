import numpy as np
import pdb
import os
from tqdm import tqdm





import tensorflow as tf
from tensorflow import keras


def test(model, test_images, test_labels):
    """ Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer,
    """
    
    loss, acc = model.evaluate(test_images, test_labels)
    ypred = model.predict(test_images)
    ypred = np.argmax(ypred, axis=1)

    return loss, test_labels, ypred


if __name__ == "__main__":

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255
    train_images2= train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images2 = test_images.reshape((test_images.shape[0], 28, 28, 1))
 
    neu_model = keras.models.load_model('./models/neural_net.h5')
    conv_model = keras.models.load_model('./models/conv_net.h5')


    

    loss, gt, pred = test(neu_model, test_images, test_labels)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, gt, pred = test(conv_model, test_images2, test_labels)
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
