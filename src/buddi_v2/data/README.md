# BuDDINData

`BuDDINData` is a generic data container class for handling bulk and single-cell deconvolution datasets. It organizes gene expression matrix (`X`), proportion matrix (`y`), encoded metadata (e.g. `label`, `stim`, `samp_type`) and metadata dataframe and enforces consistency checks, and offers flexible metadata-based filtering.

---

## Features

* **Shape & metadata validation**: Ensures feature arrays, labels, and metadata DataFrames align in sample count.
* **Internal One-hot Encoding**: Performs metadata encoding internally
* **Customizable metadata columns**: Specify column names for sample IDs, perturbations, sample types, train/test splits, and cell types.
* **Interactive filtering**: Use `query()` to filter samples by metadata values and `reset_query()` to clear filters.
* **Data retrieval**: Access filtered arrays or DataFrames via `get()` for one or multiple keys (e.g., `X`, `y`).

---

## Usage 
See examples