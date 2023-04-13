import graphviz
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import sklearn.linear_model as linear_model
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold
import sklearn.preprocessing as preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from termcolor import colored, cprint
import typing
from yellowbrick.features import PCA as yellowPCA
feature_engineering = typing.TypeVar('rabbitml.feature_engineering')

class rabbitml:
    """
    An automl library designed for tabular data

    @Taran Sean Marley 
    https://www.kaggle.com/taranmarley
    """

    class feature_engineering:
        """
        A class intended to move through and improve the features of a dataset.
        """

        def auto_casefold(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            Take a dataframe, find the string columns and convert them all to lower case through casefold

            Parameters
            ----------
            dataframe : pd.DataFrame
                The dataframe to casefold over to convert to lower case

            Returns
            -------
            pd.DataFrame
                The same dataframe given with the new lorrrwer case values if applied

            """
            for col in df.columns:
                if self.is_string_type(df[col]):
                    df[col] = df[col].astype(str).str.casefold()
            return df

        def break_up_by_string(self, df_temp: pd.DataFrame, splitting_string: str) -> pd.DataFrame:
            """
            Break up columns by string to create new columns from each split.

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to start splitting up object columns
            splitting_string : str
                String to split up columns by


            Returns
            -------
            pd.DataFrame
                modified dataframe with extra columns containing split up values
            """
            obj_cols = df_temp.select_dtypes(include=[object])
            for col in obj_cols:
                if df_temp[col].str.contains(splitting_string).sum() > 0:
                    df2 = df_temp[col].str.split(splitting_string, expand=True)
                    rename_dict = {}
                    for rename_col in df2.columns:
                        if splitting_string != ' ':
                            rename_dict[rename_col] = col + splitting_string + str(rename_col)
                        else:
                            rename_dict[rename_col] = col + str(rename_col)
                    df2 = df2.rename(columns=rename_dict)
                    df2 = df2.fillna(0)
                    df_temp = pd.concat([df_temp, df2], axis=1)
            return df_temp

        def compare_object_columns(self, df_temp: pd.DataFrame, df_temp_2: pd.DataFrame, silent=False, replace=False) -> None:
            """
            Compare object columns and print out the if there is a difference between them. This helps determining the differences between a test dataframe and a training dataframe
            
            Parameters
            ----------
            df_temp : pd.DataFrame
                First dataframe to compare columns with
            df_temp_2 : pd.DataFrame
                Second dataframe to compare columns with
            silent : bool
                Print the results or not
            replace : bool
                Replace bad values in df_temp with NaN values
            """
            for col in df_temp.select_dtypes(include='object').columns:
                if col in df_temp_2.columns:
                    unique_df_list = df_temp[col].unique().tolist()
                    test_df_list = df_temp_2[col].unique().tolist()
                    if set(unique_df_list) != set(test_df_list):
                        unique_df_list = ['nan' if x is np.nan else x for x in unique_df_list]
                        test_df_list = ['nan' if x is np.nan else x for x in test_df_list]
                        unique_df_list.sort()
                        test_df_list.sort()
                        if not silent:
                            print('***', col)
                            print(unique_df_list)
                            print(test_df_list)
                        for x in unique_df_list:
                            if x not in test_df_list:
                                df_temp[col].replace({x: np.nan})

        def create_interactions(self, df_temp: pd.DataFrame, column_list: typing.List) -> pd.DataFrame:
            """
            Create interactions by totalling and multiplying columns within a dataframe

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to create interactions in
            column_list : typing.List
                List of columns to create interactions from

            Returns
            ----------
            pd.DataFrame
                Dataframe with interactions added
            """
            for x in itertools.combinations(column_list, 2):
                df_temp[x[0] + '_X_' + x[1]] = df_temp[x[0]] * df_temp[x[1]]
                df_temp = df_temp.copy()
            iterative_total = 0
            i = 0
            for j in column_list:
                iterative_total = iterative_total + df_temp[j]
                if i > 0:
                    df_temp['A' + str(i) + '_iter_score'] = iterative_total
                    df_temp = df_temp.copy()
                i = i + 1
            return df_temp

        def detect_continous_columns(self, df_temp: pd.DataFrame, ratio: float=0.05, continous_columns: typing.List=[]) -> typing.List:
            """
            Detect the continous columns in a dataframe. Columns that have more than the given ratio by total length of dataframe will be considered continous.

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to detect continous columns in. This is assumed to already be encoded to a numerical format
            ratio : float / int
                Ratio of the total length of dataframe that will be used to cull continous from discrete data, if given as an int then this is consider to be a discrete number instead of a ratio
            continous_columns : typing.List
                Continous columns that can be given to the function without checking

            Returns
            ----------
            typing.List
                List of columns found
            """
            continous_cutoff: int = round(ratio * len(df_temp))
            if ratio > 1:
                continous_cutoff = ratio
            for col in df_temp.columns:
                if not self.is_string_type(df_temp[col]):
                    if col not in continous_columns:
                        if df_temp[col].nunique() > continous_cutoff:
                            continous_columns.append(col)
            return continous_columns

        def detect_duplicates(self, df_temp: pd.DataFrame, silent: bool=False, id_cols: typing.List=[]) -> None:
            """
            Detect duplicates in data and return the columns in which duplicates where detected.

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to detect duplicates in
            silent : bool
                Whether to run print statements 
            id_cols : typing.List
                Given id cols that aren't auto detected - Useful if there is an obvious ID column that also wants to be detected for duplication
            """
            cols_to_use = []
            for col in df_temp.columns:
                if len(df_temp[col].unique()) != len(df_temp[col]):
                    cols_to_use.append(col)
                elif col not in id_cols:
                    id_cols.append(col)
            id_temp = df_temp.copy()[id_cols]
            df_temp = df_temp.copy()[cols_to_use]
            count_dupes = df_temp.duplicated().sum()
            count_dupes_in_ID = id_temp.duplicated().sum()
            if not silent:
                print('Duplicates in data: ', str(count_dupes))
                print('Duplicates in id columns: ', str(count_dupes_in_ID))
                print('When filtering out id columns: ', str(id_cols))

        def detect_nans(self, df_temp: pd.DataFrame, name='', silent: bool=False, plot: bool=True) -> typing.List:
            """
            Detect NaNs in a provided dataframe and return the columns that NaNs were detected in     

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to detect NaN values in
            name : str
                Name of the dataframe which helps give a more descriptive read out
            silent : bool
                Whether the print statements should fire
            plot : bool
                Whether to return a plot of the counts of NaNs in the data

            Returns
            -------
            typing.List
                List of columns in the provided dataframe that contain NaN values
            """
            plt.rcParams['figure.figsize'] = (9, 9)
            count_nulls = df_temp.isnull().sum().sum()
            columns_with_NaNs = []
            if count_nulls > 0:
                for col in df_temp.columns:
                    if df_temp[col].isnull().sum().sum() > 0:
                        columns_with_NaNs.append(col)
            if not silent:
                if name != '':
                    print('******')
                    cprint('Detecting NaNs in ' + str(name), attrs=['bold'])
                    print('******')
                print('NaNs in data:', count_nulls)
                if count_nulls > 0:
                    print('******')
                    for col in columns_with_NaNs:
                        print('NaNs in', col + ': ', df_temp[col].isnull().sum().sum())
                    print('******')
            print('')
            if plot and count_nulls > 0:
                sns.barplot(y=df_temp[columns_with_NaNs].isnull().sum().index, x=df_temp[columns_with_NaNs].isnull().sum().values).set_title(str(name) + ' NaNs')
            return columns_with_NaNs

        def detect_id_columns(self, df_temp: pd.DataFrame) -> typing.List:
            """
            Detect which columns are ID columns, those for which one unique value exists for each row.

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to detect ID columns

            Returns
            -------
            typing.List
                List of Identity columns that were detected
            """
            id_cols = []
            for col in df_temp.columns:
                if len(df_temp[col].unique()) == len(df_temp[col]):
                    id_cols.append(col)
            return id_cols

        def drop_unshared_columns(self, df_temp: pd.DataFrame, df_temp_2: pd.DataFrame, exclude_columns: typing.List) -> None:
            """
            Detect which columns are not shared between the two dataframes excepting for a target_col if provided.
            Delete in place.

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to check for shared columns        
            df_temp_2 : pd.DataFrame
                Second dataframe to check for shared columns
            exclude_columns : typing.List
                Columns not to remove in this process
            """
            drop_cols: typing.List = []
            for col in df_temp_2.columns:
                if col not in df_temp.columns:
                    if col not in exclude_columns:
                        drop_cols.append(col)
            df_temp_2 = df_temp_2.drop(columns=drop_cols, axis=1, inplace=False)
            drop_cols: typing.List = []
            for col in df_temp.columns:
                if col not in df_temp_2.columns:
                    if col not in exclude_columns:
                        drop_cols.append(col)
            df_temp = df_temp.drop(columns=drop_cols, axis=1, inplace=False)

        def encode_binary_object(self, series: pd.Series) -> pd.Series:
            """
            Encode a binary object series

            Parameters
            ----------
            series : pd.Series
                The series to be encoded. 

            Returns
            -------
            pd.Series
                The encoded series
            """
            map_dict = {}
            series_list = series.unique().tolist()
            series_list.sort()
            for (i, x) in enumerate(series_list):
                map_dict[x] = i
            series = series.map(map_dict)
            return series

        def encode_columns(self, df: pd.DataFrame, columns: pd.Series, test_df: pd.DataFrame=None, cutoff: int=12) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
            """
            Encode columns based on the number of unique values in each column

            Parameters
            ----------
            df_temp : pd.DataFrame
                Dataframe to encode columns in 
            columns : pd.Series
                Columns to encode
            test_df : pd.DataFrame
                Test dataframe to encode based on classes in the Dataframe
            cut_off : int
                The cut off number of classes to choose between label encoding and get dummies. This keeps the dimensionality under control

            Returns
            -------
            (pd.DataFrame, pd.DataFrame)
                Original dataframe and the test dataframe
            """
            for col in columns:
                le = preprocessing.LabelEncoder()
                classes_to_encode = df[col].astype(str).unique().tolist()
                classes_to_encode.sort()
                classes_to_encode.append('None')