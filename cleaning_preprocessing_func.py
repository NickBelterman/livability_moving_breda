import numpy as np
import pandas as pd
import re
import logging

logging.basicConfig(level=logging.INFO)

def load_csvs(path=list, col_drop=list):
    """
    Load multiple CSV files into a list of pandas DataFrames.

    Args:
        path (list): A list of file paths or tuples containing file path and separator.
        col_drop (list): A list of columns to drop from each DataFrame (optional).

    Returns:
        list: A list of pandas DataFrames containing the loaded CSV data.
    """
    data = []
    for idx in range(0, len(path)):
        if isinstance(path[idx], tuple):
            separator = path[idx][1]
            logging.debug(f'if_tuple: PATH:{path[idx][0]} SEPARATOR:{separator}')
            data.append(pd.read_csv(path[idx][0], encoding='utf-8', encoding_errors='ignore', sep=separator))
        else:
            logging.debug(f'PATH:{path[idx]}')
            data.append(pd.read_csv(path[idx], encoding='utf-8', encoding_errors='ignore'))
    return data

def fillna_all(df):
    """
    Fill missing values in numeric columns of a DataFrame with their column means.

    Args:
        df (pandas.DataFrame): The DataFrame to fill missing values.

    Returns:
        pandas.DataFrame: The DataFrame with missing values filled.
    """
    for column in df.select_dtypes(include=[np.number]).columns:
        logging.debug(f'Filling missing values in column: {column}')
        df[column] = df[column].fillna(np.mean(df[column]))
    return df
    
def check_data_pd(X, value_count=None, dtype=False):
    """
    Analyze the data in a pandas DataFrame.

    Args:
        X (pandas.DataFrame): The DataFrame to analyze.
        value_count (int or str, optional): Value to count occurrences (optional).
        dtype (bool, optional): Flag to display column data types (optional).

    Returns:
        None
    """
    X_isna = X.isna().sum()
    X_nunique = X.nunique()

    if value_count:
        counts = {}
        for column in X.columns:
            counts[column] = X[column].value_counts().get(value_count, 0)
            logging.info(f'{column}:{counts[column]}')

    if dtype:
        for column in X.columns:
            logging.info(f'column:{column}:dtype:{type(X[column].loc[0])}')

    logging.info('======================')
    logging.info(f'Is na sum:\n{X_isna}')
    logging.info('======================')
    logging.info(f'Number unique:\n{X_nunique}')
    logging.info('======================')
    logging.info(len(X))

def clean_csv(df, fill_mean=False, col_drop_threshold=None, value=None, dropna=False, col_dropna=False, specified_col=None, to_drop=None, check_data=False, string_lower=False):
    """
    Cleans the CSV data by performing various operations such as dropping columns with missing values, filling missing
    values with mean, dropping rows with missing values, dropping specified columns, and converting strings to lowercase.

    Args:
        df (DataFrame): The input pandas DataFrame containing the CSV data.
        fill_mean (bool, optional): Whether to fill missing values with the column mean. Defaults to False.
        col_drop_threshold (float, optional): The threshold for dropping columns with a higher percentage of missing values.
            Columns with missing value percentage greater than or equal to col_drop_threshold will be dropped. Defaults to None.
        value (object, optional): The value to be replaced with column mean if fill_mean is True and specified for a particular column. Defaults to None.
        dropna (bool, optional): Whether to drop rows with missing values. Defaults to False.
        col_dropna (bool, optional): Whether to drop columns with missing values in the specified column. Defaults to False.
        specified_col (str, optional): The column to consider when dropping columns using col_dropna. Defaults to None.
        to_drop (list, optional): List of column names to be dropped from the DataFrame. Defaults to None.
        check_data (bool, optional): Whether to print the DataFrame after each operation. Defaults to False.

    Returns:
        DataFrame: The cleaned pandas DataFrame.
    """
    len_ = len(df)
    drop_cols = []    
    logging.debug(f'len:{len_}')
    df_isna = df.isna().sum()

    if check_data:
        print(f'no_mod:{check_data_pd(df)}')

    if dropna:
        df.dropna(inplace=True)

    if col_dropna:
        df = df.dropna(subset=[specified_col])

    if check_data:
        print(f'col_dropna:{check_data_pd(df)}')

    if to_drop is not None:
        for i in to_drop:
            if i in df.columns:
                df = df.drop(to_drop, axis=1)
                if check_data:
                    print(f'drop_to_drop:{check_data_pd(df)}')
    
    if col_drop_threshold is not None and isinstance(col_drop_threshold, float):    
        logging.debug(f'if_col_drop_threshold')
        for idx, col in enumerate(df.columns):
            logging.debug(f'col:{col}:df_isna[col]:{df_isna[col]}')
            if (df_isna[col] / len_) >= col_drop_threshold:
                logging.debug(f'if>=:col:{col}:df_isna[col]:{df_isna[col]}')
                drop_cols.append(col)
            if value:
                count = (df[col] == value).sum()
                # logging.debug(f'count:{count}')
                if count / len_ >= col_drop_threshold:
                    logging.debug(f'if>=:col:{col}:count:{count}')
                    drop_cols.append(col)
        df = df.drop(drop_cols, axis=1)

    if check_data:
        print(f'drop_drop_cols:{check_data_pd(df)}')
    
    if fill_mean:
        logging.debug(f'if_fill_mean:')
        if value:
            logging.debug(f'if_fill_mean:value:')
            for i in df.select_dtypes(include=[np.number]).columns:
                col_mean = df[df[i] != value][i].mean()
                df[i] = df[i].replace(value, col_mean)
                if check_data:
                    print(f'replace_value_mean:{check_data_pd(df)}')
        else:
            logging.debug(f'if_fill_mean:else:')
            for i in df.select_dtypes(include=[np.number]).columns:
                logging.debug(f'fillna:col:{i}')
                df.loc[df[i].isnull(), i] = df[i].mean()
                if check_data:
                    print(f'fillna_mean:{check_data_pd(df)}')  

    if string_lower:
        logging.debug(f'if_string_lower:')
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        if check_data:
            print(f'string_lower:{check_data_pd(df)}')

    return df

def loc_isna(df=pd.DataFrame, column=str, return_index=False):
    """
    Returns the rows or indices where the specified column in the DataFrame contains missing values (NaN).

    Args:
        df (DataFrame, optional): The input pandas DataFrame. Defaults to pd.DataFrame.
        column (str): The column to check for missing values.
        return_index (bool, optional): Whether to return the indices of rows with missing values. Defaults to False.

    Returns:
        DataFrame or Index: The rows or indices where the specified column contains missing values.
    """
    if return_index:
        logging.debug(f'return_index: column={column}')
        return df.loc[pd.isna(df[column]), :].index
    if isinstance(df, pd.core.series.Series):
        logging.debug(f'isinstance: column={column}')
        return df.loc[pd.isna(df)]
    else:
        logging.debug(f'else: column={column}')
        return df.loc[pd.isna(df[column]), :]
    
def find_difference(set_1=set(), set_2=set()):
    """
    Finds the elements that are unique to set_1 and set_2.

    Args:
        set_1 (set): The first set. Defaults to an empty set.
        set_2 (set): The second set. Defaults to an empty set.

    Returns:
        tuple: A tuple containing two sets: elements_unique_to_set1 and elements_unique_to_set2.
    """
    set1 = set(set_1)
    set2 = set(set_2)

    elements_unique_to_set1 = set1 - set2
    elements_unique_to_set2 = set2 - set1

    return elements_unique_to_set1, elements_unique_to_set2

def isin_series(series1, series2):
    """
    Returns a list of indices where the elements of series1 are equal to or present in series2.

    Args:
        series1 (Series): The first pandas Series.
        series2 (Series): The second pandas Series.

    Returns:
        list: A list of indices where the elements of series1 are equal to or present in series2.
    """
    idx_bool = []
    for idx, item in enumerate(series1):
        if isinstance(item, str):
            try:
                if item == series2[idx]:
                    idx_bool.append(idx)
            except IndexError:
                break
        if item in series2:
            idx_bool.append(idx)
    return idx_bool

def lower_string(df, axis=0, exclude=None):
    """
    Converts string values in the DataFrame to lowercase.

    Args:
        df (DataFrame): The input pandas DataFrame.
        axis (int, optional): The axis along which to convert string values to lowercase. 0 for columns, 1 for rows. Defaults to 0.
        exclude (list, optional): List of column names to exclude from lowercase conversion. Defaults to None.

    Returns:
        DataFrame: The pandas DataFrame with string values converted to lowercase.
    """
    if axis == 0:
        for column in df.columns:
            if exclude:
                if df[column].dtype == object and column not in exclude:
                    logging.debug(f'column:{column}:object:not in exclude')
                    df[column] = df[column].apply(lambda x: x.lower() if isinstance(x, str) else x)

    return df

def replace_value(df, old_value, new_value, replace_str_to_int=False):
    """
    Replaces occurrences of the old_value with the new_value in the DataFrame.

    Args:
        df (DataFrame): The input pandas DataFrame.
        old_value (object): The value to be replaced.
        new_value (object): The new value to replace the old_value with.
        replace_str_to_int (bool, optional): Whether to replace string values containing old_value with new_value and convert them to integers. Defaults to False.

    Returns:
        DataFrame: The pandas DataFrame with replaced values.
    """
    if replace_str_to_int:
        for column in df.select_dtypes(include=[object]):
            if df[column].str.contains(old_value).any():
                logging.debug(f'if_df[column]_contains_old_value')
                df[column] = df[column].str.replace(old_value, new_value).astype(int)
        return df
    else:
        logging.debug(f'else:')
        return df.replace(old_value, new_value)
      
def replace_comma(df):
    """
    Replaces commas with periods in numeric string values of the DataFrame.

    Args:
        df (DataFrame): The input pandas DataFrame.

    Returns:
        DataFrame: The pandas DataFrame with commas replaced by periods in numeric string values.
    """
    for column in df.select_dtypes(include=[object]).columns:
        if df[column].str.isnumeric().all():
            logging.warning(f"{column} contains numeric strings.")
            df[column] = df[column].str.replace(',', '.').astype(float)
        else:
            logging.warning(f"{column} does not contain numeric strings.")
            try:
                df[column] = df[column].str.replace(',', '.').astype(float)
            except ValueError:
                pass
    return df
        
def fill_values_moving(df):
    """Fill the missing values

    Parameters:
        df (dataframe): the dataframe to be filled

    Returns:
        dataframe: The filled dataframe
    """
    # Fill missing values with 0 and drop duplicates 
    df = df.fillna(0).drop_duplicates().reset_index(drop=True)
    # Drop unnamed columns if any
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def preprocess_moving_data(df, lookup, y_columns, the_dataset):
    """Order the data and clean/preprocess it

    Parameters:
        df (Dataframe): Dataframe to be processed
        lookup (Dataframe): lookup table
        y_columns (list): list of wanted columns
        the_dataset (Dataframe): the dataframe with data

    Returns:
        Dataframe: The preprocessed dataframe
    """
    # Merge the dataframes 
    df = pd.merge(lookup[['cbs_grid_code', 'neighborhood_code', 'neighborhood_name']], 
                  df[[
                      'cbs_grid_code', 
                      'moving_inside_gridcell', 
                      'leaving_gridcell', 
                      'leaving_gridcell_outside_breda', 
                      'leaving_gridcell_in_breda']], 
                  on='cbs_grid_code', 
                  how='outer')
    # Fill the missing values
    df = fill_values_moving(df)
    # Merge to order the values based on CBS grid code
    df = pd.merge(the_dataset['cbs_grid_code'], df, on='cbs_grid_code', how='left')
    # Keep only the wanted columns
    df = df[y_columns]
    return df

def create_y(df1, df2, df3, y_columns):
    """This function creates the y dataset for the model

    Parameters:
        df1 (Dataframe): Dataframe to use
        df2 (Dataframe): Dataframe to use
        df3 (Dataframe): Dataframe to use
        y_columns (list): The list of columns we wish to use in our model

    Returns:
        Dataframe: The y dataframe to use in the model
    """
    # Create the y Dataframe
    y = pd.DataFrame()
    # Calculate the mean cell value based on the three dataframes
    for i in range(0, len(y_columns)):
        y[y_columns[i]] = round((df1[y_columns[i]] + df2[y_columns[i]] + df3[y_columns[i]]) / 3)
    return y

def fill_values_based_on_neighborhood(value, df):
    """Fill the specified/unusual value with the average of the column grouped by neighborhood 

    Parameters:
        value (float): The value to replace
        df (Dataframe): The dataframe in which to apply 

    Returns:
        Dataframe: The updated dataframe
    """
    # Replace the value with the mean of the column based on neighborhood
    df = df.mask(df == value, df.groupby('neighborhood_code').transform(lambda x: x[x != value].mean()))
    # Drop the first column 
    df = df.iloc[:, 1:]
    # Round the values
    df = df.round()
    # Replace the missing values for neighborhoods that do not have any data with the column mean
    df = df.fillna(value)
    df = df.replace(value, df.apply(lambda x: round(np.mean(x[x != value]))))
    # ROund the values
    df = df.round()
    return df