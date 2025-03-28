# Buddi single cell RNA-seq preprocessing

## Preprocessing Module Overview

### 1. **`generate_pseudo_bulks.py`**
Main functions for generating pseudo-bulk expression profiles:

- **`generate_pseudo_bulk_from_counts`** and **`generate_pseudo_bulk_from_props`**  
  Generate pseudo-bulk expression from:
  - `props_df` / `counts_df`: Proportion or count matrix (`pandas.DataFrame`)
  - `in_adata`: A cell-type annotated `anndata` object
  - `cell_df`: A `dict` mapping cell types to corresponding subsetted `anndata` objects (by observation)
  - `cell_noise`: *(Optional)* List of NumPy arrays for cell-type-specific expression noise
  - `use_sample_noise`: Boolean switch to enable sample-level noise
  - `sample_noise_kwargs`: *(Optional)* Keyword arguments passed to `apply_sample_wise_noise` in `utils.py`

---

### 2. **`utils.py`**
Provides utility functions and helpers used in pseudo-bulk generation.

#### Utility Functions:
- `subset_adata_by_cell_type`: Creates the `cell_df` mapping required by `generate_pseudo_bulk_from_counts` and `generate_pseudo_bulk_from_props`
- `generate_counts_from_props`: Converts a proportion matrix to a count matrix
- `generate_prop_from_counts`: Samples a count matrix from a proportion matrix given total cell count

#### Proportion & Count Matrix Generators:
- `generate_log_normal_counts`: Samples log-normal distributed cell type abundances and normalizes to a proportion matrix
- `generate_single_celltype_dominant_props`: Generates profiles dominated by a single cell type
- `get_true_proportions`: Computes true cell type proportions from an annotated `anndata` object
- `generate_random_similar_props`: Generates proportion matrices similar to a base profile, regulated by correlation. When used with `get_true_proportions`, this can produce realistic cell type distributions.

#### Helper Functions (used in `generate_pseudo_bulks.py`):
- `get_cell_type_sum`: Samples and sums single-cell expression profiles to simulate pseudo-bulk
- `apply_sample_wise_noise`: Applies sample-level noise to pseudo-bulk profiles

---

## Workflows 

Pseudobulks can be generated from the following workflows:

0. **Pre-requisite** imports and generating `cell_df`
```python
from buddi_v2.preprocessing import utils
from buddi_v2.preprocessing import generate_pseudo_bulks

# processed, annotated adata (multiple sample)
adata = sc.read_h5ad("PATH/TO/H5AD")

CELL_TYPE_COL = "[cell type column name]"

cell_types = adata.obs[CELL_TYPE_COL].unique().to_list()
cell_order = sorted(cell_types)

# subset anndata to cell type specific ones
cell_df = utils.subset_adata_by_cell_type(
    in_adata=adata, 
    cell_type_col=CELL_TYPE_COL,
    cell_order=cell_order
)
```

1. **Realistic Pseudobulk**:
```python
# subset adata to sample specific

# subset_idx = np.where(...)
subset_adata = adata[subset_idx, :]

# Compute true proportion
true_props_df = utils.get_true_proportions(
    in_adata=subset_adata, 
    cell_type_col=CELL_TYPE_COL,
    cell_order=cell_order    
)

realistic_props_df = utils.generate_random_similar_props(
    num_samp=50,
    props_df=true_props_df
)

realistic_counts_df = utils.generate_counts_from_props(
    prop_df=realistic_props_df,
    num_cells=1000, # can also be randomly sampled per sample
)

realistic_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata,
    cell_df=cell_df,
    count_df=realistic_counts_df,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = realistic_props_df, realistic_pseudobulk_df
```

2. **Random Pseudobulk**:

```python
# subset adata to sample specific

# subset_idx = np.where(...)
subset_adata = adata[subset_idx, :]

random_count_df = utils.generate_log_normal_counts(
    cell_order=cell_order, 
    num_cells=5000, # total num of cells in each random profile
    num_samples=50,
    present_cell_types=present_cell_types
)

random_prop_df = utils.generate_prop_from_counts(
    random_count_df,
)

random_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata,
    cell_df=cell_df,
    count_df=random_count_df,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = random_prop_df, random_pseudobulk_df
```

3. **Single cell type dominant Pseudobulk**:

```python
# subset adata to sample specific

# subset_idx = np.where(...)
subset_adata = adata[subset_idx, :]

sc_pb_profile = utils.generate_single_celltype_dominant_props(
    num_samp=10, # number of samples per cell type that is present
    cell_order=cell_order,
    present_cell_types=present_cell_types,
    return_metadata=True
)
# single_cell_metadata is a list of strings with length corresponding
# to the number of rows in single_cell_props_df where the string entry
# denotes the dominant cell type corresponding to that row
sc_pb_profile = single_cell_props_df, single_cell_metadata 

single_cell_counts_df = utils.generate_counts_from_props(
    single_cell_props_df,
    num_cells=5000
)

single_cell_pseudobulk_df = generate_pseudo_bulk_from_counts(
    in_adata=subset_adata,
    cell_df=cell_df,
    count_df=single_cell_counts_df,
    # cell_noise=xxx optional,
    use_sample_noise=False
)

# Final output of workflow
psuedobulk_prop, pseudobulk_expr = single_cell_props_df, single_cell_pseudobulk_df
```