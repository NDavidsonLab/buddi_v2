from __future__ import annotations

import gc
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import numpy as np
import pandas as pd
import anndata as ad

#from .. import preprocessing
from buddi_v2 import preprocessing
from buddi_v2.preprocessing import utils
from buddi_v2.preprocessing import generate_pseudo_bulks

# -----------------------------------------------------------------------------
# Helper type aliases
# -----------------------------------------------------------------------------
NoiseSpec = Union[None, List[np.ndarray], Tuple[float, float]]

# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------

class SCAugmentor:
    """Generate single‑cell–derived pseudo‑bulk RNA‑seq profiles.

    Instantiates with an `AnnData` object. 
    Streamlines the augmentation of scRNA-seq data for use with BuDDI training.
    Supports 3 flavors of augmentation, each configurable with the `configure_*` setters:
    - `random`: Generates pseudo-bulks with random proportions of cell types.
    - `realistic`: Generates pseudo-bulks with realistic proportions of cell types.
    - `singlecell`: Generates pseudo-bulks with a single cell type dominating the profile.
    *note: The augmentation is by done per sample/stim/datasplit combination as this
        best facilitates model training. There is currently no alternative to this.

    The `run` method is called to generate the pseudo‑bulk profiles and writes them to 
        to a user specified directory as pkl files, with the following naming
        convention::
        `{sample};{stim};{split};prop_splits.pkl`: matrix of truth cell type proportions
        `{sample};{stim};{split};pseudo_splits.pkl`: matrix of pseudo-bulk expression
        `{sample};{stim};{split};meta_splits.pkl`: metadata for the pseudo-bulks

    Chained configuration is supported so the following is valid:

        aug = (SCAugmentor(adata)
                 .configure_random_pseudobulks(n_bulks=1000)
                 .configure_realistic_pseudobulks(n_bulks=0)
                 .configure_singlecell_pseudobulks(n_bulks=200))
        aug.run(save_path="./output", augmentation_name="my_aug")
    """

    # ---------------------------------------------------------------------
    # Construction & basic summary
    # ---------------------------------------------------------------------

    def __init__(
        self,
        adata: ad.AnnData,
        *,
        sample_col: str = "sample_id",
        stim_col: str = "stim",
        split_col: str = "datasplit",
        celltype_col: str = "cell_type",
        gene_id_col: str = "gene_id",
    ) -> None:
        """
        Initialise the augmentor with an AnnData object.

        :param adata: AnnData object containing the single‑cell data.
        :param sample_col: Column name in `adata.obs` for sample IDs.
        :param stim_col: Column name in `adata.obs` for stimulation conditions.
        :param split_col: Column name in `adata.obs` for data splits.
        :param celltype_col: Column name in `adata.obs` for cell types.
        :param gene_id_col: Column name in `adata.var` for gene IDs.
        """
        self.adata = adata
        self.sample_col = sample_col
        self.stim_col = stim_col
        self.split_col = split_col
        self.celltype_col = celltype_col
        self.gene_id_col = gene_id_col

        # Parse global values ------------------------------------------------
        self._samples = adata.obs[self.sample_col].unique().tolist()
        self._stims = adata.obs[self.stim_col].unique().tolist()
        self._splits = adata.obs[self.split_col].unique().tolist()
        self._cell_types = adata.obs[self.celltype_col].unique().tolist()
        self._cell_order = sorted(self._cell_types)

        self._n_genes = adata.var.shape[0]
        self._n_cell_types = len(self._cell_order)

        # Default configuration --------------------------------------------
        self._random_cfg: Dict[str, Any] = {
            "n_bulks": 1000,
            "n_cells": 5000,
            "mean": 5.0,
            "variance_range": (1.0, 3.0),
        }
        self._realistic_cfg: Dict[str, Any] = {
            "n_bulks": 100,
            "n_cells": 5000,
            "min_corr": 0.8,
        }
        self._single_cfg: Dict[str, Any] = {
            "n_bulks": 100,
            "n_cells": 5000,
            "background_prop": 0.01,
        }
        self._noise_cfg: Dict[str, Any] = {
            "cell_noise": [],  # see setter for options
            "use_sample_noise": False,
        }

        # Print basic summary ----------------------------------------------
        self._print_dataset_summary()

    # ---------------------------------------------------------------------
    # Configuration setters (return *self* to allow chaining)
    # ---------------------------------------------------------------------

    def configure_random_pseudobulks(
        self,
        *,
        n_bulks: Optional[int] = None,
        n_cells: Optional[int] = None,
        mean: Optional[float] = None,
        variance_range: Optional[Tuple[float, float]] = None,
    ) -> "SCAugmentor":
        """
        Configure the random pseudo‑bulk generator.
        Any parameter not specified will keep its previously configured value.

        :param n_bulks: Number of pseudo‑bulks to generate per sample/stim/datasplit condition. Default is 1000.
        :param n_cells: Number of cells to sample in each pseudo‑bulk. Default is 5000.
        :param mean: Mean of the log-normal distribution for generating counts. Default is 5.0.
        :param variance_range: Range of variance for the log-normal distribution. Default is (1.0, 3.0).
        """
        _update_dict(self._random_cfg, locals())
        return self

    def configure_realistic_pseudobulks(
        self,
        *,
        n_bulks: Optional[int] = None,
        n_cells: Optional[int] = None,
        min_corr: Optional[float] = None,
    ) -> "SCAugmentor":
        """
        Configure the realistic pseudo‑bulk generator.
        Any parameter not specified will keep its previously configured value.

        :param n_bulks: Number of pseudo‑bulks to generate per sample/stim/datasplit condition. Default is 100.
        :param n_cells: Number of cells to sample in each pseudo‑bulk. Default is 5000.
        :param min_corr: Minimum correlation for generating realistic proportions. Default is 0.8.
        """
        _update_dict(self._realistic_cfg, locals())
        return self

    def configure_singlecell_pseudobulks(
        self,
        *,
        n_bulks: Optional[int] = None,
        n_cells: Optional[int] = None,
        background_prop: Optional[float] = None,
    ) -> "SCAugmentor":
        """
        Configure the single‑cell dominant pseudo‑bulk generator.
        Any parameter not specified will keep its previously configured value.

        :param n_bulks: Number of pseudo‑bulks to generate per sample/stim/datasplit condition and per cell-type. 
            Default is 100.
        :param n_cells: Number of cells to sample in each pseudo‑bulk. Default is 5000.
        :param background_prop: Proportion of background cells in the pseudo‑bulk. Default is 0.01.
        """
        _update_dict(self._single_cfg, locals())
        return self

    def configure_pseudobulk_noise(
        self,
        *,
        cell_noise: Optional[NoiseSpec] = None,
        use_sample_noise: Optional[bool] = None,
    ) -> "SCAugmentor":
        """
        Configure the noise settings for pseudo‑bulk generation. 
        Any parameter not specified will keep its previously configured value.

        :param cell_noise: Noise specification for cell type proportions. 
            Can be a length n_celltypes list of arrays of shape (1, n_genes) of multiplicative noise
            or a tuple of (mean, sigma) for in-place randomly generated log-normal noise,
            or an empty array to indicate no noise. 
            Default is no noise.
        :param use_sample_noise: Whether to use sample noise. Default is False.
        """
        if cell_noise is not None:
            if isinstance(cell_noise, tuple):
                mean, sigma = cell_noise
                cell_noise = [
                    np.random.lognormal(mean, sigma, self._n_genes)
                    for _ in range(self._n_cell_types)
                ]
        if cell_noise is not None:
            self._noise_cfg["cell_noise"] = cell_noise
        if use_sample_noise is not None:
            self._noise_cfg["use_sample_noise"] = use_sample_noise
        return self

    # ---------------------------------------------------------------------
    # Main driver
    # ---------------------------------------------------------------------

    def run(
        self,
        *,
        save_path: os.PathLike | str,
        augmentation_name: str,
        seed: Optional[int] = None,
        overwrite: bool = True,
        write_gene_ids: bool = True
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        save_path = Path(save_path).expanduser().resolve()
        out_dir = save_path / augmentation_name
        if out_dir.exists():
            if overwrite:
                for p in out_dir.glob("*"):
                    if p.is_file():
                        p.unlink()
                    else:
                        _rm_tree(p)
            else:
                raise FileExistsError(
                    f"{out_dir} already exists; set overwrite=True to replace it.")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for provenance --------------------------------
        cfg_json = {
            "random": self._random_cfg,
            "realistic": self._realistic_cfg,
            "singlecell": self._single_cfg,
            "noise": self._serialisable_noise_cfg(),
            "columns": {
                "sample": self.sample_col,
                "stim": self.stim_col,
                "split": self.split_col,
                "celltype": self.celltype_col,
                "gene_id": self.gene_id_col,
            },
        }
        (out_dir / "config.json").write_text(json.dumps(cfg_json, indent=2))

        # write gene IDs once per augmentation
        if write_gene_ids:
            gene_out_file = out_dir / "genes.pkl"
            gene_ids = self.adata.var[self.gene_id_col]
            pickle.dump(gene_ids, open(gene_out_file, "wb"))

        # -----------------------------------------------------------------
        # Main triple loop
        # -----------------------------------------------------------------
        for _sample in self._samples:
            for _stim in self._stims:
                for _split in self._splits:
                    self._process_triple(_sample, _stim, _split, out_dir)
                    gc.collect()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    ## TODO: this is very ugly and does not account for missing stim/split
    ## should consider to refactor to a pandas groupby and iter
    def _process_triple(
            self, 
            sample: str, 
            stim: str, 
            split: str, 
            out_dir: Path) -> None:
        """Generate pseudo‑bulk profiles for a given sample/stim/split combination.
        This is the main driver for generating pseudo-bulk profiles.
        Finds the subset of adata matching the provided sample/stim/split and 
            runs all configured flavors of augmentation.

        :param sample: Sample ID.
        :param stim: Stimulation condition.
        :param split: Data split (e.g., train/test).
        :param out_dir: Output directory for saving the generated profiles pkl.
        :return: None
        """
        print(f"Generating pseudo‑bulk profiles for sample {sample}, stim {stim}, split {split} …")

        obs = self.adata.obs
        subset_idx = np.where(
            np.logical_and.reduce(
                (
                    obs[self.sample_col] == sample,
                    obs[self.stim_col] == stim,
                    obs[self.split_col] == split,
                )
            )
        )[0]
        if len(subset_idx) == 0:
            print(">No matching cells – skipping.")
            return

        sub_adata = self.adata[subset_idx, :]
        present_cell_types = sub_adata.obs[self.celltype_col].unique().tolist()

        cell_df = preprocessing.utils.subset_adata_by_cell_type(
            sub_adata,
            cell_type_col=self.celltype_col,
            cell_order=self._cell_order,
        )

        # ---------------------------------- RANDOM -----------------------
        pseudobulk_dfs, prop_dfs, meta_dfs = [], [], []

        if self._random_cfg["n_bulks"] > 0:
            print("  >Generating random‑prop pseudo‑bulks…")
            _pb, _prop, _meta = self._generate_random_pb(
                sub_adata,
                cell_df,
                present_cell_types,
                sample,
                stim,
                split,
            )
            pseudobulk_dfs.append(_pb)
            prop_dfs.append(_prop)
            meta_dfs.append(_meta)

        # ---------------------------------- REALISTIC --------------------
        if self._realistic_cfg["n_bulks"] > 0:
            print("  >Generating realistic‑prop pseudo‑bulks…")
            _pb, _prop, _meta = self._generate_realistic_pb(
                sub_adata,
                cell_df,
                sample,
                stim,
                split,
            )
            pseudobulk_dfs.append(_pb)
            prop_dfs.append(_prop)
            meta_dfs.append(_meta)

        # ---------------------------------- SINGLE CELL ------------------
        if self._single_cfg["n_bulks"] > 0:
            print("  >Generating single‑cell‑dominant pseudo‑bulks…")
            _pb, _prop, _meta = self._generate_singlecell_pb(
                sub_adata,
                cell_df,
                present_cell_types,
                sample,
                stim,
                split,
            )
            pseudobulk_dfs.append(_pb)
            prop_dfs.append(_prop)
            meta_dfs.append(_meta)

        if not pseudobulk_dfs:
            print("  >No flavours enabled – nothing to write.")
            return

        props_df = pd.concat(prop_dfs, axis=0)
        pseudobulk_df = pd.concat(pseudobulk_dfs, axis=0)
        metadata_df = pd.concat(meta_dfs, axis=0)

        # Write pickles ----------------------------------------------------
        prefix = f"{sample};{stim};{split};"
        (out_dir / f"{prefix}prop_splits.pkl").write_bytes(pickle.dumps(props_df))
        (out_dir / f"{prefix}pseudo_splits.pkl").write_bytes(pickle.dumps(pseudobulk_df))
        (out_dir / f"{prefix}meta_splits.pkl").write_bytes(pickle.dumps(metadata_df))

        print(f"  Done for {sample}, {stim}, {split}.\n")

    # ---------------------------------------------------------------------
    #  Augmentation flavour internals
    # ---------------------------------------------------------------------

    def _generate_random_pb(
            self, 
            sub_adata: ad.AnnData, 
            cell_df: pd.DataFrame, 
            present_cell_types: List[str], 
            sample: str, 
            stim: str, 
            split: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate pseudo-bulk profiles with random proportions of cell types.
        Streamlined call of the preprocessing functions to generate random pseudo-bulks.

        :param sub_adata: Subset of the AnnData object for the current sample/stim/split.
        :param cell_df: DataFrame of cell type proportions.
        :param present_cell_types: List of present cell types in the subset.
        :param sample: Sample ID. Used for metadata assembly only in this function.
        :param stim: Stimulation condition. Used for metadata assembly only in this function.
        :param split: Data split (e.g., train/test). Used for metadata assembly only in this function.
        :return: Pseudo-bulk DataFrame, proportion DataFrame, and metadata DataFrame.
        """

        cfg = self._random_cfg
        # Generate random log normal counts
        count_df = preprocessing.utils.generate_log_normal_counts(
            cell_order=self._cell_order,
            num_cells=cfg["n_cells"],
            num_samples=cfg["n_bulks"],
            present_cell_types=present_cell_types,
            mean=cfg["mean"],
            variance_range=cfg["variance_range"],
        )

        # Convert to proportions
        prop_df = preprocessing.utils.generate_prop_from_counts(count_df)

        # Generate pseudo-bulk profiles from the raw counts
        pb_df = preprocessing.generate_pseudo_bulks.generate_pseudo_bulk_from_counts(
            in_adata=sub_adata,
            cell_df=cell_df,
            count_df=count_df,
            **self._noise_cfg,
        )

        # Assemble metadata for augmentation type
        meta_df = pd.DataFrame({
            self.sample_col: [sample] * cfg["n_bulks"],
            self.stim_col: [stim] * cfg["n_bulks"],
            "cell_prop_type": ["random"] * cfg["n_bulks"],
            "cell_type": ["random"] * cfg["n_bulks"],
            "samp_type": ["sc_ref"] * cfg["n_bulks"],
            self.split_col: [split] * cfg["n_bulks"],
        })
        return pb_df, prop_df, meta_df

    def _generate_realistic_pb(
            self, 
            sub_adata: ad.AnnData, 
            cell_df: pd.DataFrame, 
            sample: str, 
            stim: str, 
            split: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate pseudo-bulk profiles with realistic proportions of cell types.
        Streamlined call of the preprocessing functions to generate realistic pseudo-bulks.

        :param sub_adata: Subset of the AnnData object for the current sample/stim/split.
        :param cell_df: DataFrame of cell type proportions.
        :param sample: Sample ID. Used for metadata assembly only in this function.
        :param stim: Stimulation condition. Used for metadata assembly only in this function.
        :param split: Data split keyword. Used for metadata assembly only in this function.
        :return: Pseudo-bulk DataFrame, proportion DataFrame, and metadata DataFrame.
        """
        cfg = self._realistic_cfg
        # Obtain true proportions from the subsetted adata
        true_props_df = preprocessing.utils.get_true_proportions(
            in_adata=sub_adata,
            cell_type_col=self.celltype_col,
            cell_order=self._cell_order,
        )

        # Generate realistic proportions similar to the true proportions controlled by min_corr
        realistic_props_df = preprocessing.utils.generate_random_similar_props(
            num_samp=cfg["n_bulks"],
            props_df=true_props_df,
            min_corr=cfg["min_corr"],
        )

        # Convert to counts
        realistic_counts_df = preprocessing.utils.generate_counts_from_props(
            prop_df=realistic_props_df,
            num_cells=cfg["n_cells"],
        )

        # Generate pseudo-bulk profiles from the counts
        pb_df = preprocessing.generate_pseudo_bulks.generate_pseudo_bulk_from_counts(
            in_adata=sub_adata,
            cell_df=cell_df,
            count_df=realistic_counts_df,
            **self._noise_cfg,
        )

        # Assemble metadata for augmentation type
        meta_df = pd.DataFrame({
            self.sample_col: [sample] * cfg["n_bulks"],
            self.stim_col: [stim] * cfg["n_bulks"],
            "cell_prop_type": ["realistic"] * cfg["n_bulks"],
            "cell_type": ["realistic"] * cfg["n_bulks"],
            "samp_type": ["sc_ref"] * cfg["n_bulks"],
            self.split_col: [split] * cfg["n_bulks"],
        })
        return pb_df, realistic_props_df, meta_df

    def _generate_singlecell_pb(
            self, 
            sub_adata: ad.AnnData, 
            cell_df: pd.DataFrame, 
            present_cell_types: List[str], 
            sample: str, 
            stim: str, 
            split: str
        ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate pseudo-bulk profiles with a single cell type dominating the profile.
        Streamlined call of the preprocessing functions to generate single-cell dominant pseudo-bulks.

        :param sub_adata: Subset of the AnnData object for the current sample/stim/split.
        :param cell_df: DataFrame of cell type proportions.
        :param present_cell_types: List of present cell types in the subset.
        :param sample: Sample ID. Used for metadata assembly only in this function.
        :param stim: Stimulation condition. Used for metadata assembly only in this function.
        :param split: Data split keyword. Used for metadata assembly only in this function.
        :return: Pseudo-bulk DataFrame, proportion DataFrame, and metadata DataFrame.
        """
        cfg = self._single_cfg
        # Generate single-cell dominant proportions
        single_props_df, single_meta = preprocessing.utils.generate_single_celltype_dominant_props(
            num_samp=cfg["n_bulks"],
            cell_order=self._cell_order,
            present_cell_types=present_cell_types,
            background_prop=cfg["background_prop"],
            return_metadata=True,
        )

        # Convert to counts
        single_counts_df = preprocessing.utils.generate_counts_from_props(
            single_props_df,
            num_cells=cfg["n_cells"],
        )

        # Generate pseudo-bulk profiles from the counts
        pb_df = preprocessing.generate_pseudo_bulks.generate_pseudo_bulk_from_counts(
            in_adata=sub_adata,
            cell_df=cell_df,
            count_df=single_counts_df,
            **self._noise_cfg,
        )

        # Assemble metadata for augmentation type
        meta_df = pd.DataFrame({
            self.sample_col: sample, # automatically expanded to match single_meta
            self.stim_col: stim,
            "cell_prop_type": "single_celltype",
            "cell_type": single_meta,
            "samp_type": "sc_ref",
            self.split_col: split,
        })
        return pb_df, single_props_df, meta_df

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------

    def _print_dataset_summary(self) -> None:
        """
        Print a summary of the dataset and configuration.

        TODO perhaps replace with a formal logger
        """
        print("-----------------------------------------------")
        print("PseudoBulkAugmentor initialised with:")
        print(f"  #Samples : {len(self._samples)}")
        print(f"  #Stims   : {len(self._stims)}")
        print(f"  #Splits  : {len(self._splits)}")
        print(f"  #Cell types : {len(self._cell_types)}")
        print("\nContingency table (sample × split):")
        ctab = pd.crosstab(self.adata.obs[self.sample_col], self.adata.obs[self.split_col])
        print(ctab)
        print("-----------------------------------------------\n")

    def _serialisable_noise_cfg(self):
        """
        String representation of noise configuration for JSON serialization.
        This is a workaround for the fact that numpy arrays are not JSON serializable.
        """
        cfg = self._noise_cfg.copy()
        if isinstance(cfg["cell_noise"], list):
            cfg["cell_noise"] = "<array_of_arrays>"  # don't dump raw arrays
        return cfg


# -----------------------------------------------------------------------------
# Helper functions outside the class
# -----------------------------------------------------------------------------

def _update_dict(dest: Dict[str, Any], local_vars: Dict[str, Any]) -> None:
    """Utility: update *dest* with non‑None values from *local_vars*."""
    for k, v in local_vars.items():
        if k in dest and v is not None: # skips None values
            dest[k] = v

def _rm_tree(path: Path) -> None:
    """Recursively delete a folder (like shutil.rmtree but without import)."""
    for p in path.glob("**/*"):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            _rm_tree(p)
    path.rmdir()

def load_sc_augmentation_dir(
  dir_path: Union[os.PathLike, str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all pseudo/prop/meta pickle splits from a single‑cell augmentation.

    The directory must contain files produced by :class:`SCAugmentor` following
    the pattern::

        {prefix}pseudo_splits.pkl
        {prefix}prop_splits.pkl
        {prefix}meta_splits.pkl

    where *prefix* uniquely identifies a (sample, stim, split) combination.  The
    function ensures that rows from the three DataFrames stay aligned by
    concatenating them in the same prefix order.

    :param dir_path: Path to the directory containing the pseudo/prop/meta files.
    :return: Tuple of three DataFrames: pseudo, prop, and meta.
    :raises FileNotFoundError: If the directory does not exist.
    :raises RuntimeError: If no *_splits.pkl files are found or if the files are incomplete.
    """
    dir_path = Path(dir_path).expanduser().resolve()
    if not dir_path.exists():
        raise FileNotFoundError(dir_path)

    # Build a mapping: prefix -> {kind: Path}
    file_map: Dict[str, Dict[str, Path]] = {}
    for pkl in dir_path.glob("*_splits.pkl"):
        name = pkl.name
        if name.endswith("pseudo_splits.pkl"):
            kind = "pseudo"
            prefix = name[: -len("pseudo_splits.pkl")]
        elif name.endswith("prop_splits.pkl"):
            kind = "prop"
            prefix = name[: -len("prop_splits.pkl")]
        elif name.endswith("meta_splits.pkl"):
            kind = "meta"
            prefix = name[: -len("meta_splits.pkl")]
        else:
            continue
        file_map.setdefault(prefix, {})[kind] = pkl

    if not file_map:
        raise RuntimeError(f"No *_splits.pkl files found in {dir_path}")

    # Sanity‑check: each prefix must have all three kinds
    missing = {pref: {k for k in ("pseudo", "prop", "meta") if k not in kinds}
               for pref, kinds in file_map.items() if len(kinds) < 3}
    if missing:
        raise RuntimeError("Incomplete split sets for prefixes: " + ", ".join(missing.keys()))

    # Deterministic order (alphabetical prefix)
    prefixes = sorted(file_map.keys())

    pseudo_dfs, prop_dfs, meta_dfs = [], [], []
    for pref in prefixes:
        paths = file_map[pref]
        with open(paths["pseudo"], "rb") as fh:
            pseudo_dfs.append(pickle.load(fh))
        with open(paths["prop"], "rb") as fh:
            prop_dfs.append(pickle.load(fh))
        with open(paths["meta"], "rb") as fh:
            meta_dfs.append(pickle.load(fh))

    pseudo_df = pd.concat(pseudo_dfs, axis=0, ignore_index=True)
    prop_df = pd.concat(prop_dfs, axis=0, ignore_index=True)
    meta_df = pd.concat(meta_dfs, axis=0, ignore_index=True)

    return pseudo_df, prop_df, meta_df