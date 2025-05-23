from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .plot_latent_space import _get_projection

def plot_data(
    X: np.ndarray,
    meta: pd.DataFrame,
    color_by: Optional[List[str]] = None,
    method: str = 'tSNE',
    ncols: int = 3,
    nrows: Optional[int] = None,
    panel_width: int = 5,
    alpha: float = 0.5,
    figsize: Optional[tuple] = None,
    title: str = '',
    palette: str = 'tab20',
    max_legend_categories: int = 20,
    save_path: Optional[str] = None,
    show_plot: bool = True,
):
    """
    """

    # Project Expression to 2D
    _proj = _get_projection(
        X,
        type=method,
    )

    # Add meta data to projection
    _proj = pd.concat([_proj, meta.reset_index(drop=True, inplace=False)], axis=1)

    if color_by is None:
        # default color_by
        color_by = ['sample_id', 'samp_type', 'stim', 'cell_prop_type', 'cell_type']
    for col in color_by:
        if col not in _proj.columns:
            raise ValueError(f"Column {col} not found in projection data.")

    if len(color_by) < ncols:
        ncols = len(color_by)
    if nrows is None:
        nrows = int(np.ceil(len(color_by) / ncols))    
    if figsize is None:
        figsize = (panel_width * ncols, panel_width * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if len(color_by) > 1 else np.array([axes])

    if title != '':
        fig.suptitle(title, fontsize=16)

    for i, hue_col in enumerate(color_by):
        ax = axes[i]

        sns.scatterplot(
            data=_proj,
            x=f'{method}_0',
            y=f'{method}_1',
            hue=hue_col,
            ax=ax,
            alpha=alpha,
            palette=palette,
        )

        ax.set_title(hue_col)
        ax.set_xticks([])
        ax.set_yticks([])

        if _proj[hue_col].nunique() > max_legend_categories:
            ax.legend_.remove()

    # also remove the empty axes
    for i in range(len(color_by), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close(fig)