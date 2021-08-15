from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from typing import List

################################################################################
# constants

CDIR = os.path.dirname(os.path.abspath(__file__))
NN_PLOT_DIR = os.path.join(CDIR, "nn_plots")
LR_PLOT_NAME = "lr_plot.png"

################################################################################

def lr_plotter(
    model: keras.Model, 
    x_small: List, y_small: List, 
    epochs: int = 12, 
    figure_name: str = LR_PLOT_NAME
) -> None:
    """plots the loss of a small dataset with increasing learning rate so we can 
    see the optimal learning rate"""
    
    opt = keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: 1e-4 * 10**(epoch/2)
    )
    
    history = model.fit(x=x_small, y=y_small,
                epochs=epochs,
                callbacks=[lr_schedule]
    )

    plt.semilogx(history.history['lr'], history.history['loss'])

    save_path = os.path.join(NN_PLOT_DIR, figure_name)
    plt.savefig(save_path)
    plt.close()
