from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import warnings

from ..plotting.plot_data import plot_data

class BuDDI4Data:
    """
    Data container for BuDDI dataset, supports filtering and data retrieval.
    """
    def __init__(
        self,
        X : np.ndarray, 
        y: np.ndarray, 
        label: np.ndarray, 
        stim: np.ndarray, 
        samp_type: np.ndarray, 
        meta: pd.DataFrame,
        gene_names, cell_type_names, encode_meta,
        sample_column: str = "sample_id",
        stim_column: str = "stim",
        samp_type_column: str = "samp_type",
        split_column: str = "isTraining",
        ct_column: str = "cell_type"
    ):
        # store metadata column names (read-only)
        self._sample_column = sample_column
        self._stim_column = stim_column
        self._samp_type_column = samp_type_column
        self._split_column = split_column
        self._ct_column = ct_column

        # store raw data
        self.data = {
            "all": {
                "X": X, "Y": y,
                "label": label, "stim": stim,
                "samp_type": samp_type, "meta": meta.reset_index(drop=True, inplace=False)
            }
        }
        self._gene_names = gene_names
        self._cell_type_names = cell_type_names
        self._encode_meta = encode_meta

        # validate shapes and metadata
        self._check_shapes()
        self._check_meta_columns()

        # initialize selection (no filtering)
        self.reset_query()

        # some data properties
        self.__nx = X.shape[1]
        self.__ny = y.shape[1]
        self.__n_labels = label.shape[1]
        self.__n_stims = stim.shape[1]
        self.__n_samp_types = samp_type.shape[1]

    # ─── Input Checkers ───────────────────────────────────────────
    def _check_shapes(self):
        """Ensure all arrays and meta have matching first-dimension lengths."""
        for cond, entries in self.data.items():
            lengths = {}
            for key, arr in entries.items():
                if key == "meta":
                    if not isinstance(arr, pd.DataFrame):
                        raise TypeError(f"meta_{cond} must be a pandas DataFrame")
                    length = len(arr)
                else:
                    if not hasattr(arr, '__len__'):
                        raise TypeError(f"{key}_{cond} must be array-like with __len__")
                    length = len(arr)
                lengths[key] = length
            # ensure all lengths equal
            dims = set(lengths.values())
            if len(dims) != 1:
                raise ValueError(f"Length mismatch in '{cond}' data: {lengths}")

    def _check_meta_columns(self):
        """Ensure required metadata columns exist for both conditions."""
        for cond in self.data:
            meta = self.data[cond]['meta']
            for attr in ['sample_column', 'stim_column', 'samp_type_column', 'split_column', 'ct_column']:
                col_name = getattr(self, f"_{attr}")
                if col_name not in meta.columns:
                    raise ValueError(f"Column '{col_name}' not found in metadata of '{cond}'")

    # ─── Exposed Properties ───────────────────────────────────────────

    @property
    def gene_names(self):
        return self._gene_names

    @property
    def cell_type_names(self):
        return self._cell_type_names

    @property
    def encode_meta(self):
        return self._encode_meta

    @property
    def sample_column(self):
        return self._sample_column

    @property
    def stim_column(self):
        return self._stim_column

    @property
    def samp_type_column(self):
        return self._samp_type_column

    @property
    def split_column(self):
        return self._split_column

    @property
    def ct_column(self):
        return self._ct_column
    
    @property
    def nx(self):
        """Number of features (genes) in the data."""
        return self.__nx
    @property
    def ny(self):
        """Number of features (cell types) in the data."""
        return self.__ny
    @property
    def n_labels(self):
        """Number of sample labels in the data."""
        return self.__n_labels
    @property
    def n_stims(self):
        """Number of stimulation conditions in the data."""
        return self.__n_stims
    @property
    def n_samp_types(self):
        """Number of sample types in the data."""
        return self.__n_samp_types

    # ─── Override Class Methods ───────────────────────────────────────────
    def __repr__(self):

        return (
            f"BuDDI4Data(total_samples={len(self)}, "
            f"genes={len(self.gene_names)}, cell_types={len(self.cell_type_names)})"
        )

    def __len__(self):
        """Number of samples currently selected (after filtering)."""

        counts = []
        for cond, _ in self.data.items():
            counts.append(
                int(self._selection[cond].sum())
            )

        total = sum(counts)

        return total

    # ─── Query Methods ───────────────────────────────────────────

    def reset_query(self):
        """
        Reset any filters; selects all samples for both conditions.
        """
        self._selection = {}
        for cond, entries in self.data.items():
            n = len(entries['X'])
            self._selection[cond] = np.ones(n, dtype=bool)
        return self

    def query(self, conditions=None, **filters):
        """
        Filter samples based on metadata columns.

        :param conditions: 'unkp', 'kp', or list of those. Defaults to both.
        :param filters: keyword filters on metadata columns (e.g., stim='Dox', isTraining=True).
        """
        # determine target conditions
        if conditions is None:
            conds = list(self.data.keys())
        else:
            conds = [conditions] if isinstance(conditions, str) else conditions
        conds = [c for c in conds if c in self.data]
        if not conds:
            raise ValueError(f"No valid conditions in query: {conditions}")

        # check filter values against combined metadata
        for key, val in filters.items():
            if key not in [
                self._sample_column,
                self._stim_column,
                self._samp_type_column,
                self._split_column,
                self._ct_column
            ]:
                raise KeyError(f"Filter '{key}' is not a valid metadata column")
            values = val if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)) else [val]
            # gather all unique values across both conditions
            all_vals = set()
            for cond in self.data:
                df = self.data[cond]['meta']
                all_vals.update(df[key].unique().tolist())
            missing = set(values) - all_vals
            if missing:
                warnings.warn(
                    f"Filter value(s) {missing} for column '{key}' not found in any metadata",
                    UserWarning
                )

        # apply filtering: set selection for each condition
        for cond in self.data:
            df = self.data[cond]['meta']
            if cond in conds:
                mask = pd.Series(True, index=df.index)
                for key, val in filters.items():
                    values = (
                        val
                        if isinstance(val, (list, tuple, set, np.ndarray, pd.Series))
                        else [val]
                    )
                    mask &= df[key].isin(values)
                self._selection[cond] = mask.values
            else:
                # conditions not targeted get no selection
                self._selection[cond] = np.zeros(len(df), dtype=bool)
        return self

    # ─── Data Retrieval ───────────────────────────────────────────

    def get(self, 
            keys: List[str] = [
                'X', 'Y', 'label', 'stim', 'samp_type', 'meta'
                ],
            n_samples: Optional[int] = None,
            replace: bool = False,
            random_state: Optional[int] = None,
        ):
        """
        Retrieve data arrays or DataFrames for current selection.

        :param keys: single key or list of keys (e.g., 'X', ['X','Y']).
            By default, retrieves all data.
        :param n_samples: number of samples to return (default all).
        :param replace: whether to sample with replacement (default False).
        :param random_state: random seed for reproducibility (default None).
        :returns: array/DataFrame or tuple thereof.
        """
        key_list = [keys] if isinstance(keys, str) else keys
        results = []
        for k in key_list:
            parts = []
            for cond, mask in self._selection.items():
                data = self.data[cond].get(k)
                if data is None:
                    raise KeyError(f"Key '{k}' not found in condition '{cond}'")
                if isinstance(data, pd.DataFrame):
                    part = data.loc[mask].reset_index(drop=True)
                else:
                    part = data[mask]
                parts.append(part)
            # concatenate across conditions
            first = parts[0]
            if isinstance(first, pd.DataFrame):
                concat = pd.concat(parts, ignore_index=True)
            else:
                concat = np.concatenate(parts, axis=0)
            results.append(concat)

        results_sampled = []
        if n_samples is not None:
            if n_samples > len(self) and not replace:
                raise ValueError(f"Cannot sample {n_samples} samples from {len(self)} total samples")
            if n_samples <= 0:
                raise ValueError(f"n_samples must be positive, got {n_samples}")            
            if random_state is not None:
                rng = np.random.RandomState(random_state)
            else:
                rng = np.random.default_rng()

            idx = np.arange(len(results[0]))
            idx_sample = rng.choice(
                idx,
                size=n_samples,
                replace=replace
            )
                
            for result in results:
                if isinstance(result, pd.DataFrame):
                    results_sampled.append(
                        result.iloc[idx_sample].reset_index(drop=True)
                    )
                else:
                    results_sampled.append(
                        result[idx_sample]
                    )
        else:
            results_sampled = results
        
        return results_sampled[0] if len(results_sampled) == 1 else tuple(results_sampled)

    # ─── Filter value hint function ───────────────────────────────────────────

    def unique_values(self, column, conditions=None):
        """
        Get unique metadata values for a column across specified conditions.

        :param column: metadata column name.
        :param conditions: condition or list (default all).
        :returns: sorted list of unique values.
        """
        if column not in [self._sample_column, self._stim_column, self._samp_type_column, self._split_column, self._ct_column]:
            raise KeyError(f"Column '{column}' is not a valid metadata column")
        if conditions is None:
            conds = list(self.data.keys())
        else:
            conds = [conditions] if isinstance(conditions, str) else conditions
        values = set()
        for cond in conds:
            df = self.data[cond]['meta']
            if column not in df.columns:
                raise KeyError(f"Column '{column}' not found in metadata of '{cond}'")
            values.update(df[column].unique().tolist())
        return sorted(values)
    
    # ─── Plotting ───────────────────────────────────────────
    def plot(
            self,
            color_by: Optional[List[str]]=None,
            filters: Optional[Dict]=None,
            **kwargs,
        ):
        """
        Plot the data using the plot_data function.
        """
        # Get the data
        if filters is not None:
            X, meta = self.query(**filters).get(['X', 'meta'])
        else:
            X, meta = self.reset_query().get(['X', 'meta'])
        
        if color_by is None:
            # default color_by plots all metadata columns
            color_by = [
                self._sample_column,
                self._samp_type_column,
                self._stim_column,
                self._split_column,
                self._ct_column
            ]
        color_by = [
            col for col in color_by if col in meta.columns
        ]

        if len(color_by) == 0:
            raise ValueError("No valid columns found in color_by")

        # Call the plot_data function
        plot_data(
            X=X,
            meta=meta,
            color_by=color_by,
            **kwargs
        )