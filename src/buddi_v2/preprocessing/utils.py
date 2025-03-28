from typing import Dict, List, Union, Tuple, Optional, Iterable
import warnings

import pandas as pd
import numpy as np
from anndata import AnnData

CellDf = Dict[str, AnnData]

"""
scRNA seq data processing utilities
"""

def get_true_proportions(
        in_adata: AnnData, 
        cell_type_col: str,
        cell_order: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
    """
    Helper function that calculates the true proportion of cell types in the given AnnData object.
    Requires a user specificed column name in in_adata.obs that contains cell type labels.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_type_col: Column name in in_adata.obs specifying cell type labels.
    :param cell_order: Optional Iterable of strings indicating all cell types to include.
    If not provided, will used the sorted unique values of in_adata.obs[cell_type_col].
    If provided, should be a superset of all unique cell types in in_adata.
    :return: DataFrame containing the true proportion of each cell type.
    """
    
    if cell_order is None:
        cell_order = sorted(in_adata.obs[cell_type_col].unique())
    elif isinstance(cell_order, Iterable):
        if all([ctype in cell_order for ctype in in_adata.obs[cell_type_col].unique()]):
            pass
        else:
            raise ValueError("cell_order must contain all unique cell types in the input data")
    else:
        raise TypeError("cell_order must be a list of strings")
        
    prop_counts = in_adata.obs[cell_type_col].value_counts(normalize=True)
    prop_counts = pd.DataFrame([prop_counts])

    return prop_counts.reindex(columns=cell_order, fill_value=0)

def subset_adata_by_cell_type(
        in_adata: AnnData, 
        cell_type_col: str,
        cell_order: Optional[Iterable[str]] = None
    ) -> CellDf:
    """
    Constructs a dictionary mapping each cell type to a subset of the AnnData object
    containing only cells of that type.
    Requires a user specified column name in in_adata.obs that contains cell type labels.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param cell_type_col: Column name in in_adata.obs specifying the cell type labels.
    :param cell_order: Optional Iterable of strings indicating all cell types to include.
    If not provided, will used the sorted unique values of in_adata.obs[cell_type_col].
    If provided, should be a superset of all unique cell types in in_adata. Cell types that 
    do not exist in in_adata will prompt this function to create an empty AnnData object to indicate absence
    of a cell type in the input adata split. 
    :return: Dictionary where keys are cell type names and values are subsetted AnnData objects.
    When a cell type is not present in the input data, the corresponding value will be an empty AnnData object.
    """

    if cell_order is None:
        cell_order = sorted(in_adata.obs[cell_type_col].unique())
    elif isinstance(cell_order, Iterable):
        if all([ctype in cell_order for ctype in in_adata.obs[cell_type_col].unique()]):
            pass
        else:
            raise ValueError("cell_order must contain all unique cell types in the input data")
    else:
        raise TypeError("cell_order must be a list of strings")
    
    cell_df = {}
    for cell_type in cell_order:
        # when cell_type does not exist in the input adata, this will be 
        # an empty AnnData object of shape (0, num_genes) which indicates absence of the cell type
        # while the number of genes will still be trackable despite has no observations.
        # This will be useful downstream
        cell_df[cell_type] = in_adata[in_adata.obs[cell_type_col] == cell_type]

        if cell_df[cell_type].shape[0] == 0:
            warnings.warn(f"Cell type '{cell_type}' not found in the input data. "
                          "Creating an empty AnnData paceholder.\n"
                          "Downstream in the workflow no pseudo-bulk samples will be generated for this cell type."
                          )

    return cell_df

"""
Pseudo-bulk sample proportion/count generation utilities
"""

def generate_random_similar_props(
        num_samp: int, 
        props_df: Union[pd.DataFrame, pd.Series],
        min_corr: float = 0.8
    ) -> pd.DataFrame:
    """
    Helper function that generates a proportion matrix where each sample's cell-type proportions correlated 
    to a given base proportion vector. 
    This is meant for generating pseudo-bulk samples similar to some ground truth dataset in terms of composition but with some variation.

    :param num_samp: Number of samples to generate.
    :param props_df: DataFrame or Series containing the base proportion vector. 
    DataFrame should have shape (1, num_celltypes) and column names should be cell type names.
    If a DataFrame has more than one row, only the first row will be considered as the prop vector.
    Series should have cell type names as index/key.
    :param min_corr: Minimum correlation threshold with the base proportion.
    :return: DataFrame with shape (num_samp, num_celltypes), where each row is a sample's cell-type proportions.
    """
    total_prop_list = []

    if isinstance(props_df, pd.DataFrame):
        props_df = props_df.iloc[0, :] # to series
    elif isinstance(props_df, pd.Series):
        pass
    else:
        raise TypeError("props_df must be a DataFrame or Series")
    
    base_prop = props_df.values
    cell_order = props_df.keys().to_list()

    ## Sample random proportion scaling factors until the scaled proportion 
    # is sufficiently correlated with the base proportion
    while len(total_prop_list) < num_samp:
        # Apply noise to base proportion vector
        noisy_prop = base_prop * np.random.lognormal(0, 1, len(base_prop))
        noisy_prop = noisy_prop / np.sum(noisy_prop)  # Normalize to sum to 1

        # Compute correlation coefficient with the base proportion
        corr_coef = np.corrcoef(noisy_prop, base_prop)[0, 1]

        if corr_coef >= min_corr:
            total_prop_list.append(noisy_prop)

    # Convert to DataFrame
    return pd.DataFrame(total_prop_list, columns=cell_order)

def generate_single_celltype_dominant_props(
        num_samp: int, 
        cell_order: Iterable[str], 
        background_prop: float = 0.01,
        present_cell_types: Optional[Iterable[str]] = None,
        return_metadata: bool = False
    ) -> pd.DataFrame:
    """
    Helper function to generate a proportion matrix where each row represents a sample in which one cell type 
    dominates while other cell types have a small background presence.

    :param num_samp: Number of samples to generate for each cell type.
    :param cell_order: Iterable of cell types (column names for the output DataFrame).
    :param background_prop: Proportion assigned to non-dominant cell types.
    :param present_cell_types: Optional list of cell types to include in the count matrix.
    If not provided, all cell types in cell_order are assumed to be present.
    If provided, will be used to identify missing cell types and zero out their counts.
    :return: DataFrame of shape (num_samp * num_celltypes, num_celltypes).
    """
    num_celltypes = len(cell_order)

    if present_cell_types is None:
        present_cell_types = cell_order

    total_prop_list = []
    cell_dominance_metadata = []

    for dominant_idx, cell_type in enumerate(cell_order):

        if cell_type not in present_cell_types:
            continue

        # Start with background levels for all cell types
        # faster memory allocation vs [background_prop] * num_celltypes
        base_prop = np.full(num_celltypes, background_prop)

        # Set dominant cell type to full proportion
        # no need to softmax here get_corr_prop_matrix handles it
        base_prop[dominant_idx] = 1
        base_prop_df = pd.DataFrame([base_prop], columns=cell_order)

        # Generate correlated proportion matrix
        prop_matrix = generate_random_similar_props(
            num_samp, base_prop_df, min_corr=0.95)
        total_prop_list.append(prop_matrix)

        cell_dominance_metadata.extend(
            [cell_type] * num_samp
        )
    
    props_df = pd.concat(total_prop_list, ignore_index=True)

    if return_metadata:
        return props_df, cell_dominance_metadata
    else:
        return props_df

def generate_prop_from_counts(
        count_df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Helper function that generates a proportion matrix based on a count matrix.

    :param count_df: DataFrame containing cell counts per cell type.
    :return: DataFrame containing cell type proportions.
    """

    return count_df.div(count_df.sum(axis=1), axis=0).reset_index(drop=True)

def generate_counts_from_props(
        prop_df: Union[pd.DataFrame, pd.Series],
        num_cells: Optional[Union[int, List[int]]] = None,
        random_num_cell_range: Tuple[int, int] = (200, 5000)
    ) -> pd.DataFrame:
    """
    Helper function that generates a count matrix based on a proportion matrix and the total number of cells.

    :param prop_df: DataFrame or Series containing cell type proportions.
    DataFrame should have shape (num_samp, num_celltypes) and column names should be cell type names.
    Series should have cell type names as index/key.
    :param num_cells: Number of total cells to sample. Optional. 
    When not provided, samples random number of cells as a random integer between random_num_cell_range.
    :param random_num_cell_range: Tuple specifying the range of random number of cells to sample.
    :return: DataFrame containing cell counts per cell type. 
    If prop_df is a dataframe, returns dataframe of shape (length(prop_df), num_celltypes).
    If prop_df is a series, returns dataframe of shape (1, num_celltypes).
    """

    if isinstance(prop_df, pd.DataFrame):
        pass
    elif isinstance(prop_df, pd.Series):
        prop_df = pd.DataFrame(prop_df).T
    else:
        raise TypeError("prop_df must be a DataFrame or Series")
    
    count_df = pd.DataFrame(columns=prop_df.columns)

    if num_cells is None:
        num_cells = np.random.randint(*random_num_cell_range, len(prop_df))
    elif isinstance(num_cells, int):
        num_cells = np.full(len(prop_df), num_cells)
    elif isinstance(num_cells, list):
        if len(num_cells) != len(prop_df):
            raise ValueError("If specified as a list, num_cells must have "
                             "the same length as the number of samples in prop_df")
        num_cells = np.array(num_cells)
    else:
        raise TypeError("num_cells must be an integer or a list of integers")

    for i, (_, prop_profile) in enumerate(prop_df.iterrows()):
        count_vec = np.ceil(prop_profile * num_cells[i]).astype(int)
        # Adjust rounding inconsistencies
        count_vec.iloc[np.argmax(count_vec)] += (num_cells[i] - count_vec.sum())

        count_df = pd.concat([count_df, pd.DataFrame(count_vec).T])

    return count_df

def generate_true_counts(
        in_adata: AnnData, 
        num_cells: int, 
        cell_type_col: str
    ) -> pd.DataFrame:
    """
    Helper function that generates a count vector based on the true cell type proportions in an AnnData object.
    Calls get_true_proportions to get the true prop df and uses generate_counts_from_props to generate the count df.

    :param in_adata: The AnnData object containing single-cell expression data.
    :param num_cells: Number of total cells to sample.
    :param cell_type_col: Column name in in_adata.obs specifying cell type labels.
    :return: pandas DataFrame containing cell counts per cell type.
    """
    true_prop_df = get_true_proportions(in_adata, cell_type_col)
    return generate_counts_from_props(true_prop_df, num_cells)

def generate_log_normal_counts(
        cell_order: Iterable[str], 
        num_cells: int, 
        num_samples: int,
        mean: float = 5.0, 
        variance_range: Tuple[float, float] = (1.0, 3.0),
        present_cell_types: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
    """
    Generates a count vector by sampling from a log-normal distribution.

    :param cell_order: List of cell type names.
    :param num_cells: Total number of cells.
    :param num_samples: Number of samples to generate.
    :param mean: Mean of the log-normal distribution.
    :param variance_range: Tuple specifying the range of variance to randomly sample from.
    :param present_cell_types: Optional list of cell types to include in the count matrix.
    If not provided, all cell types in cell_order are assumed to be present.
    If provided, will be used to identify missing cell types and zero out their counts.
    :return: DataFrame with count vectors as rows and cell types as columns.
    """

    num_celltypes = len(cell_order)

    if present_cell_types is None:
        present_cell_types = cell_order

    prop_rows = []

    for _ in range(num_samples):
        rand_variance = np.random.uniform(*variance_range)
        rand_lognorm_vec = np.random.lognormal(mean, rand_variance, num_celltypes)

        # zero out missing cell type prop
        presence_mask = np.isin(cell_order, present_cell_types)
        rand_lognorm_vec[~presence_mask] = 0

        rand_prop_vec = rand_lognorm_vec / rand_lognorm_vec.sum()
        prop_rows.append(rand_prop_vec)
    
    prop_df = pd.DataFrame(prop_rows, columns=cell_order)

    return generate_counts_from_props(prop_df, num_cells)

"""
Pseudo-bulk expression generation utilities
"""

def get_cell_type_sum(
        cell_adata: AnnData, 
        num_cells: int
    ) -> np.array:
    """
    Helper function to generate the pseudobulk gene expression for a given cell type,
    given the cell type specific subset of the AnnData object and the number of cells to sample.

    :param cell_adata: The subsetted AnnData object corresponding to the target cell type.
    :param num_cells: Number of cells to sample.
    :return: Summed gene expression vector for the selected cells of shape (num_genes,).
    """
    # Handle case where there are no cells of this type
    if cell_adata.shape[0] == 0:
        return np.zeros(cell_adata.shape[1])

    # Sample cells with replacement
    sampled_cells = cell_adata[np.random.choice(cell_adata.shape[0], num_cells, replace=True)]

    # Sum gene expression across sampled cells
    return sampled_cells.X.sum(axis=0)

"""
Noise application utilities
"""

def apply_sample_wise_noise(
        expr: np.array,
        gene_capture_efficiency_var: float = 1.0,
        gene_library_size_var: float = 0.1,
        gene_noise_var: float = 0.1,
        poisson_resample: bool = True
    ):
    """
    Helper function to apply noise to a gene expression vector.

    :param expr: Gene expression vector of shape (1, num_genes).
    :param gene_capture_efficiency_var: Variance of the log-normal distribution for gene capture efficiency.
    :param gene_library_size_var: Variance of the log-normal distribution for library size scaling.
    :param gene_noise_var: Variance of the log-normal distribution for random variability.
    :param poisson_resample: Whether to resample the expression vector as a Poisson distribution.
    :return: Gene expression vector with noise applied.
    Shape of the output is (1, num_genes).
    """
    num_genes = expr.shape[1]

    # sample specific scaling across genes
    expr *= np.random.lognormal(0, gene_capture_efficiency_var, num_genes)
    # library size scaling
    expr *= np.random.lognormal(0, gene_library_size_var, 1)[0]
    # random variability
    expr *= np.random.lognormal(0, gene_noise_var, num_genes)

    if poisson_resample:
        # re-sample as poisson with expr as mean to get integer counts
        expr = np.random.poisson(expr)

    return expr.reshape(1, -1) # back to 2D with shape[0] = 1 and shape[1] = num_genes