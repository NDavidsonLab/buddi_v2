from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

from . import utils
from .utils import CellDf

def generate_pseudo_bulk_from_counts(
        in_adata: AnnData, 
        cell_df: CellDf, 
        count_df: pd.DataFrame,
        cell_noise: Union[List[np.array], Tuple[float,float]] = None, 
        use_sample_noise: bool = True,
        sample_noise_kwargs: Dict = {}
    ) -> pd.DataFrame:
    """
    Generates pseudobulk expression profiles based on provided count data.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_df: Dictionary of cell type names mapped to subsetted AnnData objects.
    :param count_df: DataFrame containing the number of cells to sample for each cell type.
    :param cell_noise: List of noise vectors for each cell type or a tuple of (mean, std) for lognormal noise generation.
        If None, default noise is generated.
        If an empty list is provided, cell noise is disabled.
        If a tuple is provided, it must contain two float values (mean, std) for lognormal noise generation.
        If a list of numpy arrays is provided, each array must have the same length as the number of genes in in_adata.    
    :param use_sample_noise: Whether to apply additional noise to the pseudobulk profiles.
    :return: Tuple of (total proportion DataFrame, total expression DataFrame).
    """
    num_celltypes = count_df.shape[1]

    # Generate cell-specific noise if not provided
    if cell_noise is None: # Generate default noise #TODO: this is quite counter-intuitive behavior for None
        cell_noise = [np.random.lognormal(0, 0.1, in_adata.shape[1]) for _ in range(num_celltypes)]
    elif isinstance(cell_noise, List):
        if len(cell_noise) == 0:
            # Disable cell noise if an empty list is provided
            cell_noise = [np.ones(in_adata.shape[1]) for _ in range(num_celltypes)]
        elif len(cell_noise) != num_celltypes:
            raise ValueError(f"Length of cell_noise list ({len(cell_noise)}) does not match number of cell types ({num_celltypes}).")
        else:
            if not all(isinstance(noise, np.ndarray) for noise in cell_noise):
                raise ValueError("All elements in cell_noise list must be numpy arrays.")
            if not all(len(noise) == in_adata.shape[1] for noise in cell_noise):
                raise ValueError("All elements in cell_noise list must have the same length as the number of genes in in_adata.")
    elif isinstance(cell_noise, Tuple):
        if len(cell_noise) == 2 and all(isinstance(param, float) for param in cell_noise):
            # Generate noise based on provided mean and std
            cell_noise = [np.random.lognormal(cell_noise[0], cell_noise[1], in_adata.shape[1]) for _ in range(num_celltypes)]
        else:
            raise ValueError("If cell_noise is a tuple, it must contain two float values (mean, std).")
    total_expr_list = []

    for samp_idx, (_, count_profile) in enumerate(tqdm(count_df.iterrows(), total=len(count_df))):
        sum_over_cells = np.zeros((1, in_adata.shape[1]))

        for cell_idx, (cell_type, cell_count) in enumerate(count_profile.items()):

            ct_sum = utils.get_cell_type_sum(cell_df[cell_type], cell_count)
            # Apply cell-specific noise
            ct_sum = np.multiply(ct_sum, cell_noise[cell_idx])
            sum_over_cells += ct_sum

        # Apply sample noise if enabled
        if use_sample_noise:
            sum_over_cells = utils.apply_sample_wise_noise(
                sum_over_cells, **sample_noise_kwargs)

        # Convert to DataFrame
        sum_over_cells_df = pd.DataFrame(sum_over_cells)
        sum_over_cells_df.columns = in_adata.var['gene_ids']

        total_expr_list.append(sum_over_cells_df)

    # Combine into final DataFrames
    total_expr_df = pd.concat(total_expr_list, axis=0)

    return total_expr_df

def generate_pseudo_bulk_from_props(
        in_adata: AnnData, 
        cell_df: CellDf, 
        props_df: pd.DataFrame, 
        num_cells: Optional[Union[int, List[int]]] = None, 
        random_num_cell_range: Tuple[int, int] = (200, 5000),
        cell_noise: List[np.array] = None, 
        use_sample_noise: bool = True,
        sample_noise_kwargs: Dict = {}
    ) -> pd.DataFrame:
    """
    Generates pseudobulk expression profiles based on provided proportion data.
    Wrapper function that calls generate_pseudo_bulk_from_counts after converting proportions to counts.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_df: Dictionary of cell type names mapped to subsetted AnnData objects.
    :param props_df: DataFrame containing the proportion of cells to sample for each cell type.
    :param num_cells: Number of total cells to sample. Optional. 
    When not provided, samples random number of cells as a random integer between random_num_cell_range.
    :param random_num_cell_range: Tuple specifying the range of random number of cells to sample.
    :param cell_noise: List of noise vectors for each cell type.
    :param use_sample_noise: Whether to apply additional noise to the pseudobulk profiles.
    :return: Tuple of (total proportion DataFrame, total expression DataFrame).
    """
    
    count_df = utils.generate_count_from_props(props_df, num_cells, random_num_cell_range)

    return generate_pseudo_bulk_from_counts(
        in_adata=in_adata, 
        cell_df=cell_df, 
        count_df=count_df,
        cell_noise=cell_noise, 
        use_sample_noise=use_sample_noise,
        sample_noise_kwargs=sample_noise_kwargs
    )