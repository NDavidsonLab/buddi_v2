import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(
        loss_df, 
        panel_width=5,
        show_plot=True, 
        save_path=None
    ):
    """
    Plot the training and validation loss for each epoch.

    :param loss_df: DataFrame containing the loss data.
    :param panel_width: Width of each panel in inches.
    :param show_plot: Whether to show the plot.
    :param save_path: Path to save the plot.
    :return: None
    """
    loss_df = loss_df.copy()

    # finding columns that end with '_loss' and select for plotting
    loss_cols = [
        col for col in loss_df.columns if col.endswith('_loss')
    ]
    recon_loss_cols = [
        col for col in loss_cols if col.startswith('X_reconstruction_')
    ]
    classifier_loss_cols = [
        col for col in loss_cols if col.startswith('classifier_')
    ]
    prop_estimator_loss_cols = [
        col for col in loss_cols if col.startswith('prop_estimator_')
    ]
    plot_loss_cols = recon_loss_cols + classifier_loss_cols + prop_estimator_loss_cols

    # scale train and val loss so they are comparable on the x axis
    loss_df['n_batches'] = (
        loss_df
        .groupby(['epoch','split','type'])['batch']
        .transform('max')   # zero‐based index of last batch
        + 1                 # so max+1 = total count
    )
    loss_df['x'] = loss_df['epoch'] + loss_df['batch'] / loss_df['n_batches']

    fig, axes = plt.subplots(
        1, len(plot_loss_cols), 
        figsize=(panel_width * len(plot_loss_cols), panel_width), sharex=True)

    for ax, col in zip(axes, plot_loss_cols):
        sns.lineplot(
            data=loss_df, 
            x='x', 
            y=col, 
            hue='type', 
            style='split',
            ax=ax
        )
        ax.set_title(col)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()
