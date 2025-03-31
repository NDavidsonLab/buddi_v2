import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

def _get_projection(z, type='PCA'):
    if type == 'PCA':
        proj = PCA(n_components=2)
    elif type == 'UMAP':
        proj = umap.UMAP(n_components=2)
    elif type == 'tSNE':
        proj = TSNE(n_components=2)
    else:
        raise ValueError(f"Unknown projection type {type}")
    
    proj_df = pd.DataFrame(
        proj.fit_transform(z),
        columns=[f'{type}_0', f'{type}_1']
    )

    return proj_df

def _plot_projection_marker(
        proj_df, 
        color_vec, 
        ax, 
        title="", 
        alpha=0.3, 
        legend_title="", 
        marker_vec=None,
        palette="hls"
    ):
    proj_df[legend_title] = color_vec

    if marker_vec is not None:
        proj_df['marker'] = marker_vec

    g = sns.scatterplot(
        x=proj_df.columns[0], y=proj_df.columns[1],
        data=proj_df,
        hue=legend_title,
        style='marker' if marker_vec is not None else None,
        palette=sns.color_palette(palette, len(np.unique(color_vec))),
        legend="full",
        alpha=alpha, ax= ax
    )

    ax.set_title(title)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    return g

def plot_latent_spaces_buddi4(
        unsupervised_buddi,
        X_tmp,
        meta_tmp,
        type='PCA',
        alpha=1,
        panel_width=5,
        figsize=None,
        show_plot=True,
        save_path=None,
        palette="hls"
):
    
    y_pred = unsupervised_buddi((X_tmp))
    _, z_label, z_stim, z_samp_type, z_slack, _, _, _, y_hat = y_pred

    z_label = z_label[:,:z_label.shape[1]//2]
    z_stim = z_stim[:,:z_stim.shape[1]//2]
    z_samp_type = z_samp_type[:,:z_samp_type.shape[1]//2]
    z_slack = z_slack[:,:z_slack.shape[1]//2]

    latent_spaces = [y_hat, z_label, z_stim, z_samp_type, z_slack]
    latent_space_names = ['Cell Type', 'Sample ID', 'Perturbation', 'Technology', 'Slack']

    if 'cell_type' in meta_tmp.columns:
        cell_prop_vec = meta_tmp['cell_type'].values
        marker_vec = meta_tmp['cell_prop_type'].values
    else:
        cell_prop_vec = meta_tmp['cell_prop_type'].values
        marker_vec = None
    label_vec = meta_tmp['sample_id'].values
    stim_vec = meta_tmp['stim'].values
    samp_type_vec = meta_tmp['samp_type'].values

    color_vecs = [cell_prop_vec, label_vec, stim_vec, samp_type_vec]
    color_legend_names = [
        'Cell Type',
        'Sample ID',
        'Perturbation',
        'Samp Type'
    ]

    fig, axs = plt.subplots(len(color_vecs), len(latent_spaces), figsize=figsize if figsize is not None else (panel_width*len(latent_spaces), panel_width*len(color_vecs)))

    for i, (latent_space_name, latent_space) in enumerate(zip(latent_space_names, latent_spaces)):

        proj_df = _get_projection(latent_space, type=type)

        for j, (color_legend_name, color_vec) in enumerate(zip(color_legend_names, color_vecs)):

            _plot_projection_marker(
                proj_df=proj_df,
                color_vec=color_vec,
                marker_vec=marker_vec,
                ax=axs[j, i],
                title=latent_space_name if j == 0 else "", 
                alpha=alpha, 
                legend_title=color_legend_name,
                palette=palette
            )

            # Remove legend for all but the last column in each row
            if i != len(latent_spaces) - 1 or color_legend_name == 'Sample ID': # do not display legend for the sample ids
                axs[j, i].get_legend().remove()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return fig