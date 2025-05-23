from typing import Optional, List, Dict, Union, Any
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from ..plotting.plot_data import plot_data

class BuDDINData:
    """
    Generic container supporting any number of one-hot encoded metadata fields.
    """
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,        
        gene_names: List[str], 
        cell_type_names: List[str],
        encode_fields: Optional[List[str]] = None,
        split_column: str = "isTraining",
        ct_column: str = "cell_type"
    ):
        """
        Initializes a BuDDINData object by storing the input 
        expression and proportion matrix as well as metadata. 
        The user specifies through the `encode_fields` argument 
        which metadata columns are needed for training and should 
        be one-hot encoded.         
        """

        # 1) Check if the input data are of correct format
        encode_fields = encode_fields if encode_fields is not None else []
        meta = meta.reset_index(drop=True, inplace=False)
        self.__check_inputs(
            X=X,
            y=y,
            meta=meta,
            split_column=split_column,
            ct_column=ct_column,
            encode_fields=encode_fields
        )

        self._split_column = split_column
        self._ct_column = ct_column
        self._encode_fields = encode_fields
        self._gene_names = gene_names
        self._cell_type_names = cell_type_names

        # 2) Perform one-hot encoding of the metadata fields
        self._encode_data: Dict[str, pd.DataFrame] = {}
        self._encoded_arrays: Dict[str, np.ndarray] = {}
        for field in encode_fields:
            enc = OneHotEncoder(dtype=int)
            arr = enc.fit_transform(meta[[field]]).toarray()
            feat_names = enc.get_feature_names_out([field]).tolist()
            self._encoded_arrays[field] = arr
            self._encode_data[field] = pd.DataFrame(
                arr, 
                columns=feat_names, 
                index=meta.index)

        # 3) Define data
        self.data = {
            'X': X,
            'y': y,
            'meta': meta,
            **self._encoded_arrays
        }

        # 4) Some data properties
        self._n = X.shape[0]
        self._shape = {
            'X': X.shape[1],
            'y': y.shape[1],
            **{
                f'n_{field}': arr.shape[1]
                for field, arr in self._encoded_arrays.items()
            }
        }

        # 5) Placeholder for selection
        self._selection: Optional[np.ndarray] = None
        _ = self.reset_query()

    # ─── Query Methods ───────────────────────────────────────────
    def reset_query(self):
        """
        Reset the query to the original data.
        """
        self._selection = np.ones(self._n, dtype=bool)
        
        return self # for chaining
        
    def query(self, **filters):
        """
        Filter samples based on metadata columns.
        """
        
        mask = pd.Series(self._selection, index=self.data['meta'].index)
        for col_name, filter_values in filters.items():
            if col_name not in self.data['meta'].columns:
                raise KeyError(f"Column '{col_name}' not found in metadata")
            
            # wrap filter_values in a list if it is not already
            _values = filter_values if isinstance(
                filter_values, 
                (list, tuple, set, np.ndarray, pd.Series)) else [filter_values]

            # check if filtering condition includes values not present in the data
            # and warn the user if so
            all_vals = set()
            all_vals.update(self.data['meta'].loc[:, col_name].unique().tolist())
            missing = set(_values) - all_vals
            if missing:
                warnings.warn(
                    f"Some values in {col_name} are not present in the data: {missing}"
                )

            # actually logical and filter with existing selection (mask)
            mask &= self.data['meta'].loc[:, col_name].isin(_values)
            
        self._selection = mask.values

        return self # for chaining

    # ─── Data Retrieval ───────────────────────────────────────────

    def __subsample(
        self,
        retrievals: List[Any],
        n_samples: Optional[int] = None,
        replace: bool = False,
        random_state: Optional[int] = None
    ):
        if n_samples is None:
            return retrievals
        
        if n_samples > len(self) and not replace:
            warnings.warn(
                f"Requested {n_samples} samples, but only {len(self)} available. "
                "Returning all available samples."
            )
            return retrievals
        
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than 0")
        
        retrievals_sampled = []
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random.default_rng()
        
        idx = np.arange(len(self))
        idx_sample = rng.choice(
            idx, 
            size=n_samples, 
            replace=replace
        )

        for retrieval in retrievals:
            retrievals_sampled.append(
                retrieval[idx_sample, :].copy() if isinstance(retrieval, np.ndarray) 
                else retrieval.iloc[idx_sample, :].copy()
            )

        return retrievals_sampled

    def get(
            self, 
            keys: Optional[List[str]] = None,
            n_samples: Optional[int] = None,
            replace: bool = False,
            random_state: Optional[int] = None,
        ):
        """
        Retrieve data arrays or DataFrames for current selection.
        """

        if keys is None:
            keys = list(self.data.keys())
        else:
            keys = [keys] if isinstance(keys, str) else keys
            for key in keys:
                if key not in self.data.keys():
                    raise KeyError(f"Key '{key}' not found in data")
        returns = []
        for key in keys:
            _data = self.data[key]
            returns.append(
                _data[self._selection, :].copy() if isinstance(_data, np.ndarray) 
                else _data.iloc[self._selection, :].copy()
            )

        returns = self.__subsample(
            returns,
            n_samples=n_samples,
            replace=replace,
            random_state=random_state
        )

        return returns[0] if len(returns) == 1 else tuple(returns)
    
    # ─── Plotting ───────────────────────────────────────────
    def plot(
        self,
        color_by: Optional[List[str]]=None,
        **kwargs,
    ):
        
        if len(self) == 0:
            raise ValueError(
                "No samples selected. "
                "Please use the query method to select samples or reset_query()"
                )
        
        X, meta = self.get(['X', 'meta'])
        if color_by is None:
            color_by = self._encode_fields + [
                self._split_column,
                self._ct_column
            ]
        else:
            color_by = [
                col for col in color_by if col in meta.columns
            ]        
            if len(color_by) == 0:
                raise ValueError("No valid columns to color by")
            
        # Call the plot_data function
        plot_data(
            X=X,
            meta=meta,
            color_by=color_by,
            **kwargs
        )
    
    # ─── Exposed Properties ───────────────────────────────────────────
    @property
    def keys(self):
        return list(self.data.keys())
    
    @property
    def gene_names(self):
        return self._gene_names

    @property
    def cell_type_names(self):
        return self._cell_type_names
    
    @property
    def encoded(self):
        return self._encode_data
    
    @property
    def shape(self):
        return self._shape

    ## Maybe the below 2 are not needed
    @property
    def split_column(self):
        return self._split_column
    
    @property
    def ct_column(self):
        return self._ct_column
    
    # ─── Overridden Methods ───────────────────────────────────────────
    def __len__(self):
        """Number of samples currently selected (after filtering)."""
        return int(self._selection.sum())
    
    def __repr__(self):
        return (
            f"BuDDINData(selected samples={len(self)}/{self._n}, "
            f"genes={len(self.gene_names)}, cell_types={len(self.cell_type_names)})"
        )
    
    # ─── Input Checkers ───────────────────────────────────────────
    def __check_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta: pd.DataFrame,
        gene_names: Optional[List[str]] = None,
        cell_type_names: Optional[List[str]] = None,
        split_column: str = "isTraining",
        ct_column: str = "cell_type",
        encode_fields: Optional[List[str]] = None,
    ):
        """
        Check if the input data is a numpy array and if the input pandas DataFrame contains the required columns.
        """
        self.__check_x(X, gene_names)
        self.__check_y(y, cell_type_names)
        self.__check_metadata(
            meta,
            split_column=split_column,
            ct_column=ct_column,
            encode_fields=encode_fields
        )
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] != meta.shape[0]:
            raise ValueError("X and meta must have the same number of rows")
        return None
    
    def __check_x(
            self, 
            X: np.ndarray,
            gene_names: Optional[List[str]] = None
        ) -> None:
        """
        Check if the input data is a numpy array.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")
        if gene_names is not None:
            if len(gene_names) != X.shape[1]:
                raise ValueError("gene_names must have the same length as the number of columns in X")
        return None
    
    def __check_y(
            self, 
            y: np.ndarray,
            cell_type_names: Optional[List[str]] = None
        ) -> None:
        """
        Check if the input data is a numpy array.
        """
        if not isinstance(y, np.ndarray):
            raise TypeError("y must be a numpy array")
        if cell_type_names is not None:
            if len(cell_type_names) != y.shape[1]:
                raise ValueError("cell_type_names must have the same length as the number of columns in y")
        return None
    
    def __check_metadata(
            self, 
            meta: pd.DataFrame,
            split_column: str,
            ct_column: str,
            encode_fields: Optional[List[str]] = None
        ) -> None:
        """
        Check if the input data is a pandas DataFrame.
        """
        if not isinstance(meta, pd.DataFrame):
            raise TypeError("meta must be a pandas DataFrame")
        if split_column not in meta.columns:
            raise ValueError(f"split_column '{split_column}' not found in meta columns")
        if ct_column not in meta.columns:
            raise ValueError(f"ct_column '{ct_column}' not found in meta columns")
        if encode_fields is not None:
            for field in encode_fields:
                if field not in meta.columns:
                    raise ValueError(f"encode_field '{field}' not found in meta columns")   

        return None
    
