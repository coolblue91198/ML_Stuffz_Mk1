# Imports
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import typing
import random

# Define Class for Pre-Analysis
class PreAnalysis:
    """
    Class for Pre-Analysis of data for Machine Learning Processes
    """

    def __init__(self, data: pd.DataFrame, label: str = None) -> None:
        """Return an PreAnalysis object whose data is *data* and label is *label*"""

        self.data = data
        self.label = label

    def data_layout(self) -> None:
        """ Print data.head() and data.describe() in better format """

        print('\n\n')
        print("Data Preview:")
        print('===================================================================')
        print(self.data.head())
        print('\n')
        print("Statistical Info:")
        print('===================================================================')
        print(self.data.describe())
        print('\n\n')

    def var_choice(self, col: typing.List[int]) -> typing.List[str]:
        """
        var_choice(col): Choose variables to narrow down data

        col: Index of Columns of data to analyze

        """

        if len(col) <= len(self.data.columns):
            col_list = list(self.data.columns)  # Store list of variables/columns

            var_list: typing.List[str] = []  # List of columns to perform analysis on

            for arg in col:
                var_list.append(col_list[arg])

            self.data = self.data[var_list]

            return var_list

        else:
            raise RuntimeError("Number of columns passed greater than available columns")

    def find_outliers(self, variables: typing.List[str], remove_out=False) -> typing.List[int]:
        """
        find_outliers(data, variables): Finds outliers in given variables

        variables: List of variables to find outliers for

        remove_out: Determines whether or not outliers are removed from dataset

        """

        dinfo = self.data.describe()  # Statistics on data

        iqr = {}  # Inter Quartile Range

        ul = {}  # Upper Limit

        ll = {}  # Lower Limit

        outliers = {}  # Outliers (boolean)

        outlier_index = {}  # Outlier Indices (int)

        for var in variables:
            if var not in list(self.data.columns):
                my_str = var + ' is not a valid variable'
                raise Exception(my_str)

            # IQR Calcs
            iqr_key = 'iqr_' + var

            iqr_value = dinfo[var][6] - dinfo[var][4]  # IQR = Q3 - Q1

            iqr[iqr_key] = iqr_value

            # Upper Limit Calcs
            ul_key = 'ul_' + var

            ul_value = dinfo[var][6] + (1.5 * iqr_value)  # UL = Q3 + 1.5(IQR)

            ul[ul_key] = ul_value

            # Lower Limit Calcs
            ll_key = 'll_' + var

            ll_value = dinfo[var][4] - (1.5 * iqr_value)  # LL = Q1 - 1.5(IQR)

            ll[ll_key] = ll_value

            # Find Outliers
            out_key = 'out_' + var

            # Outlier = data greater than upper limit or data less than lower limit
            out_value = (self.data[var] >= ul_value) | (self.data[var] <= ll_value)

            outliers[out_key] = out_value

            # Convert boolean outliers to dataframe indices
            outidx_key = 'out_index_' + var

            outidx_value = list(self.data[out_value].index.values)

            outlier_index[outidx_key] = outidx_value

        unique_outliers = self.unique_values(outlier_index)

        # Remove Outliers
        if remove_out is True:
            self.data = self.data.drop(unique_outliers).reset_index(drop=True)

        return unique_outliers

    @staticmethod
    def unique_values(dictionary: typing.Dict) -> typing.List[int]:
        """Returns list of unique values in given dictionary"""

        temp = list(dictionary.values())

        unique: typing.List[int] = []

        for i in temp:
            if type(i) is list:
                for j in i:
                    if j not in unique:
                        unique.append(j)
            else:
                if i not in unique:
                    unique.append(i)

        return unique

    def separate_data(self, split: float = 0.80) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separates data into training and testing sets
        :param split: percentage of data to train on
        :return: Returns Two Pandas Dataframes
        """
        len_data = len(self.data)  # Length of Data

        self.data = self.data.sample(frac=1).reset_index(drop=True)  # Shuffle Data

        train_idx = round(len_data * split)  # Last index of training data

        train_data = self.data[:train_idx].reset_index(drop=True)  # Training Data

        test_data = self.data[train_idx:].reset_index(drop=True)  # Testing Data

        return test_data, train_data


# Define Class for Training Data
class TrainAlgo:
    """
    Class for Training data for Linear Regression Machine Learning Processes
    """

    def __init__(self, data: pd.DataFrame, label: str = None) -> None:
        """Return a Train Algo object whose data is *data* and label is *label*"""
        self.data = data
        self.label = label
        self.num_points = len(data)  # Length of Data i.e Total number of points
        self.x_points = data[data.columns[0]]  # X variable
        self.y_points = data[data.columns[1]]  # Y variable

    def error_calcs(self, m: float, b: float) -> float:
        """
    	Calculates Error (Least Squares Method)
    	:param m: Slope
    	:param b: Y-Intercept
    	:return: Sum of squared error
    	"""
        error_sum = 0  # Initialize sum of error

        for i in range(self.num_points):
            error_sum += (self.y_points[i] - m * self.x_points[i] - b) ** 2

        return float(error_sum / self.num_points)

    def gradient_step(self, m_current: float, b_current: float,
                      learn_rate: float) -> typing.Tuple[float, float]:
        """
    	Calculates the step needed to take for gradient descent
    	:param m_current: Current Slope
    	:param b_current: Current Y-Intercept
    	:param learn_rate: Learning rate of the function
    	:return: New slope (m) and intercept (b) in direction of steepest descent
    	"""

        grad_m = 0  # Initialize gradient of m

        grad_b = 0  # Initialize gradient of b

        # Calculate and sum up gradients
        for i in range(self.num_points):
            grad_m += -(2 / self.num_points) * self.x_points[i] \
                      * (self.y_points[i] - m_current * self.x_points[i] - b_current)
            grad_b += -(2 / self.num_points) * (self.y_points[i] - m_current * self.x_points[i]
                                                - b_current)
        # Calculate new m and b
        m_new = m_current - (learn_rate * grad_m)
        b_new = b_current - (learn_rate * grad_b)

        return m_new, b_new

    def grad_descent(self, starting_m: float, starting_b: float, learn_rate: float,
                     num_iter: int) -> typing.Tuple[float, float]:
        """
        Applies Gradient Descent Method for specified number of iterations
        :param starting_m: Starting value for the slope, m
        :param starting_b: Starting value for the intercept, b
        :param learn_rate: Learning Rate for the gradient step
        :param num_iter: Number of times to iterate the method
        :return: The best fit values for m and b
        """
        m = starting_m
        b = starting_b
        for i in range(num_iter):
            m, b = self.gradient_step(m, b, learn_rate)
        return m, b


def pre_anl():
    """Loads and Prepares Data for Training and Testing Algorithms"""

    data = sns.load_dataset("tips")  # Load *tips* dataset from seaborn

    pa = PreAnalysis(data, 'Tips Data')  # Create *PreAnalysis* object

    print("Original Data")

    pa.data_layout()  # Take a look at the raw, original data

    # Plot Original Data
    plot1 = sns.relplot(x='total_bill', y='tip', data=data)
    plot1.set(xlabel='Total Bill [$]', ylabel='Tips [$]', title='Original Data')
    plt.show()

    pa.var_choice([0, 1])  # Reduce Data to *total_bill* and *tip* columns

    pa.find_outliers(['total_bill', 'tip'], True)  # Find and remove outliers

    print("Refined Data")

    pa.data_layout()  # Look at refined data

    ref_data = pa.data

    # Plot Refined Data
    plot2 = sns.relplot(x='total_bill', y='tip', data=ref_data)
    plot2.set(xlabel='Total Bill [$]', ylabel='Tips [$]', title='Refined Data')
    plt.show()

    # Separate Data
    train_data, test_data = pa.separate_data()

    # Store Separated Data
    fid1 = open('train_data', 'wb')
    pickle.dump(train_data, fid1)
    fid1.close()
    fid2 = open('test_data', 'wb')
    pickle.dump(test_data, fid2)
    fid2.close()


def train_alg():
    """Trains Linear Regression Algorithm"""

    # Obtain Data From Pickle
    fid_o1 = open('train_data', 'rb')
    data_train = pickle.load(fid_o1)
    fid_o1.close()

    # Create TrainAlgo object
    ta = TrainAlgo(data_train, 'Seaborn Training Data')

    # Define Constants
    learning_rate = 0.0001      # Learn Rate
    initial_m = random.random()     # Choose random m value
    initial_b = random.random()     # Choose random b value
    num_iterations = 1000       # Number of times to iterate

    # Calculate Initial Error
    init_error = ta.error_calcs(initial_m, initial_b)

    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {init_error} ")
    print("Running...")

    final_m, final_b = ta.grad_descent(initial_m, initial_b, learning_rate, num_iterations)

    # Calculate Final Error
    final_error = ta.error_calcs(final_m, final_b)

    print(f"After {num_iterations} iterations: b = {final_b}, m = {final_m}, error = {final_error}")

    # Plot Results
    plot1 = sns.relplot(x='total_bill', y='tip', )



if __name__ == '__main__':
    pre_anl()
    train_alg()
