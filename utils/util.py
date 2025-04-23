# util for insects-species-dataset
import glob
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
"""
Specification:
- Pre-split
Drop invalid trajectories (< X valid data points)
- Post-split
Imputation/interpolation of missing values
Dropping trailing rows
Normalization
"""


def remove_columns_except_xy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame containing only the columns whose names start with 'x0' or 'y0'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with various columns.

    Returns:
        pd.DataFrame: DataFrame containing only columns starting with 'x0' or 'y0'.
    """
    cols = [col for col in df.columns if col.startswith("x0") or col.startswith("y0")]
    return df[cols]
    

def interpolate_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Interpolates missing values in a DataFrame assumed to have columns starting with 'x0' and 'y0'.
    Linear interpolation is applied in both directions (forward and backward) to fill missing data.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with numeric columns (e.g., x0, y0) that may contain missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values interpolated.
    """
    df_interp = df.copy()
    # check the number of missing values
    num_missing = df_interp.isna().sum().sum()
    # Apply linear interpolation to fill missing values along the index.
    df_interp = df_interp.interpolate(method='linear', limit_direction='both')
    return df_interp, num_missing

def read_dataset(
    dirpath: str,
    interpolate: bool,
    normalize: bool,
    screen_width: int = 1920,
    screen_height: int = 1080,
) -> Tuple[Dict[str, Dict], Optional[MinMaxScaler]]:
    """
    Reads CSV files from the specified directory and returns a dictionary of track dictionaries.
    Each track dictionary contains:
        - 'filename': the name of the file.
        - 'label': extracted from the filename (using the first character).
        - 'dataframe': the processed DataFrame.
        
    Parameters:
        dirpath (str): Directory containing CSV files.
        interpolate (bool): Whether to interpolate missing values.
        normalize (bool): Whether to normalize x and y columns.
        screen_width (int): Width for normalization (x values).
        screen_height (int): Height for normalization (y values).
        
    Returns:
        Dict[str, Dict]: Dictionary with each key as a filename and value as a dict containing
                         the filename, label, and processed DataFrame.
    """
    dataset = {}
    # Search for CSV files in the provided directory.
    for filepath in glob.glob(os.path.join(dirpath, "*.csv")):
        filename = os.path.basename(filepath)
        # Use the first character of the filename as the label (adjust as needed)
        label = filename[0].lower()

        # Read the CSV file into a DataFrame.
        df = pd.read_csv(filepath)
        
        # Dynamically find the x and y columns (assuming they start with 'x0' and 'y0')
        x_col = next((col for col in df.columns if col.startswith("x0")), None)
        y_col = next((col for col in df.columns if col.startswith("y0")), None)
        frame_col = next((col for col in df.columns if col.startswith("nframe")), None)
        
        if x_col is None or y_col is None or frame_col is None:
            print(f"Warning: Could not find required columns in {filename}. Skipping file.")
            continue

        # Copy the dataframe as a starting point
        clean_df = df.copy()

        if interpolate:
            # Interpolate missing values linearly, applying the change to clean_df.
            clean_df = clean_df.interpolate(method='linear', limit_direction='both')

        if normalize:
            # Normalize the x and y columns.
            normalized_df = clean_df.copy()
            normalized_df.loc[:, normalized_df.columns.str.startswith('x0')] /= screen_width
            normalized_df.loc[:, normalized_df.columns.str.startswith('y0')] /= screen_height
            clean_df = normalized_df.copy()

        # Drop all columns except the ones starting with 'x0' and 'y0',
        # then rename these columns to 'x' and 'y'
        clean_df = clean_df[[x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})

        

        # Add the processed file to the dictionary.
        dataset[filename] = {
            "filename": filename,
            "label": label,
            "dataframe": clean_df
        }



    return dataset

from typing import Dict, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    PowerTransformer,
)

def compute_and_scale_deltas(
    dataset: Dict[str, Dict],
    scaler: Optional[MinMaxScaler] = None,
    scaler_str: str = None,
) -> MinMaxScaler:
    """
    For each track in `dataset` (with keys 'filename', and values containing
    at least 'dataframe': a DataFrame with columns ['x','y']), this function:
    
      1. Replaces the DataFrame with its first differences (Δx, Δy), filling NaN with 0.
      2. Fits a MinMaxScaler (feature_range=(0,1)) on the concatenated deltas across all tracks
         if no `scaler` is provided.
      3. Transforms each track's delta-DataFrame in-place to the [0,1] range per feature.
    
    Args:
        dataset: mapping filename -> {"dataframe": DataFrame[['x','y']], "label": ...}
        scaler: Optional pre-fitted MinMaxScaler. If None, a new one is fit on this dataset.
    
    Returns:
        The MinMaxScaler used to transform the data.
    """
    # 1) Compute first differences for each track
    diff_frames = []
    for info in dataset.values():
        df = info['dataframe']
        # Compute Δx, Δy and replace
        df_diff = df.diff().fillna(0.0)
        info['dataframe'] = df_diff
        diff_frames.append(df_diff)

    if not diff_frames:
        raise ValueError("Dataset is empty; no dataframes to process.")

    # 2) Stack all diffs to fit scaler if needed
    all_deltas = pd.concat(diff_frames, axis=0)[['x','y']].values  # shape (sum_T, 2)

    scalers = {
        "standard": StandardScaler(),
        "minmax":    MinMaxScaler(),
        "robust":    RobustScaler(),
        "maxabs":    MaxAbsScaler(),
        "quantile":  QuantileTransformer(output_distribution="uniform"),
        "power":     PowerTransformer(method="yeo-johnson"),
    }
    if scaler is None:
        if scaler_str is None:
            scaler_str = 'standard'
        scaler = scalers[scaler_str]
        scaler.fit(all_deltas)

    # 3) Transform each track in-place
    for info in dataset.values():
        df_diff = info['dataframe']
        scaled = scaler.transform(df_diff[['x','y']].values)
        # put back into a DataFrame with same index
        info['dataframe'] = pd.DataFrame(
            scaled,
            columns=['x','y'],
            index=df_diff.index
        )

    return scaler
