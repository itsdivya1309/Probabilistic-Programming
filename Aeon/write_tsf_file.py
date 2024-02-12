import os
import textwrap
import numpy as np
import pandas as pd

def write_to_tsf_file(
    X, path, y=None, problem_name="sample_data.tsf", header=None, horizon=0
):
    """Write an aeon collection of time series to text file in .tsf format.

    Write metadata and data stored in aeon compatible data set to file.
    A description of the tsf format is in examples/load_data.ipynb.

    Note that this file is structured to still support the

    Parameters
    ----------
    X : pd.DataFrame, each cell a pd.Series
        Collection of time series: univariate, multivariate, equal or unequal length.
    path : string.
        Location of the directory to write file
    y: None or pd.Series, default = None
        Response variable, discrete for classification, continuous for regression
        None if clustering.
    problem_name : string, default = "sample_data"
        The file is written to <path>/<problem_name>/<problem_name>.tsf
    header: string, default = None
        Optional text at the top of the file that is ignored when loading.
    """
    if not (
        isinstance(X, pd.DataFrame)
    ):
        raise TypeError(
            f" Wrong input data type {type(X)} convert to pd.DataFrame"
        )

    # See if passed file name contains .tsf extension or not
    split = problem_name.split(".")
    if split[-1] != "tsf":
        problem_name = problem_name + ".tsf"

    _write_dataframe_to_tsf_file(
        X, 
        path, 
        problem_name, 
        y=y,
        horizon=horizon,
        comment=header
    )

def _write_dataframe_to_tsf_file(
    X, path, problem_name="sample_data", y=None, horizon=0, comment=None
):
    # ensure data provided is a dataframe
    if not isinstance(X, pd.DataFrame):
        raise ValueError(f"Data provided must be a DataFrame, passed a {type(X)}")
    # See if passed file name contains .tsf extension or not
    split = problem_name.split(".")
    if split[-1] != "tsf":
        problem_name = problem_name + ".tsf"
    equal_length = not X.isnull().values.any()
    missing = X.isnull().values.any()
    columns = X.dtypes.to_dict()
    for i in columns.keys():
        if columns[i]=='float64' or columns[i]=='int32' or columns[i]=='int64':
            columns[i]='numeric'
        if columns[i]=='datetime64[ns]':
            columns[i]='date'
        else:
            columns[i]='string'
    file = _write_header_tsf(
        path,
        problem_name,
        attribute=columns,
        equal_length=equal_length,
        frequency=calculate_frequency(X),
        horizon=horizon,
        missing=missing,
        comment=comment,
        suffix=None,
    )
    n_cases, n_channels = X.shape
    
    for j in range(0, n_channels):
        column_name = X.columns[j]
        file.write(f"{column_name}:")
        
        for i in range(0, n_cases):
            series = X.iloc[i, j]
            # Check if the value is NaN
            if pd.notna(series):
                series_str = str(series)
            else:
                series_str = '?'  # Replace NaN with a ?
                
            # Write the series string to the file
            file.write(f"{series_str},")
        # Check if y is not None before accessing its elements
        if y is not None:
            file.write(f"{y[i]}\n")
        else:
            file.write("\n")  # Write a newline if y is None
    file.close()

def calculate_frequency(df):
    # Convert timestamps to DateTime format
    df['Timestamp'] = pd.to_datetime(df.index)

    # Calculate time differences
    time_diffs = df['Timestamp'].diff().dropna()

    # Calculate median time difference
    median_diff = time_diffs.median()

    # Determine frequency based on median time difference
    if median_diff <= pd.Timedelta(days=1):
        frequency = "daily"
    elif median_diff <= pd.Timedelta(weeks=1):
        frequency = "weekly"
    elif median_diff <= pd.Timedelta(days=30):
        frequency = "monthly"
    elif median_diff <= pd.Timedelta(days=365):
        frequency = "yearly"
    else:
        frequency = "other"  # You can define more granular frequencies as needed
    df.drop('Timestamp', axis=1, inplace=True)

    return frequency

def _write_header_tsf(
    path,
    problem_name,
    attribute={'col':'data_type'},
    equal_length=True,
    frequency="weekly",
    horizon=0,
    missing=False,
    comment=None,
    suffix=None,
):
    if not os.path.exists(path):
        os.makedirs(path)
    if suffix is not None:
        load_path = load_path + suffix   
    # See if passed file name contains .tsf extension or not
    split = problem_name.split(".")
    if split[-1] != "tsf":
        problem_name = problem_name + ".tsf"       
    load_path = f"{path}/{problem_name}"
    
    file = open(load_path, "w")

    if comment is not None:
        file.write("\n# ".join(textwrap.wrap("# " + comment)))
        file.write("\n")
        
    file.write(f"@relation {str(split[0]).lower()}\n")
    # Write attribute metadata for each column
    if attribute is not None:
        for attr in attribute:
            file.write(f"@attribute {str(attr)} {str(attribute[attr])}\n")
    file.write(f"@frequency {str(frequency).lower()}\n")
    file.write(f"@horizon {str(horizon).lower()}\n")
    file.write(f"@missing {str(missing).lower()}\n")
    file.write(f"@equallength {str(equal_length).lower()}\n")
    file.write("@data\n")

    return file
