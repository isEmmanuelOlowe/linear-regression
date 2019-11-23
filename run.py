import pandas as pd
import matplotlib.pyplot as plt


def LinearRegression(data):
    """
    Creates a Regression function for a data frame.

    :param data: the datafram being modelled for.
    :return: a regression function
    """

    # Finding the averags
    x_mean = data["Age"].mean()
    y_mean = data["Blood Pressure"].mean()

    # Finding the standard Deviation
    x_std = data["Age"].std()
    y_std = data["Blood Pressure"].std()

    # Calculating gradient between y and x
    gradient = y_std / x_std

    # Calcuating the y intercept
    constant = y_mean - gradient * x_mean

    def predict(x):
        """
        Predicts the value of y given x

        :param x: the feature being predicted for.
        :return: prediction.
        """
        return x * gradient + constant
    return predict


if __name__ == "__main__":
    # Reads the csv file
    df = pd.read_csv("data.csv")
    # Scatter plot of Age against Blood Pressure
    plt.scatter(df["Age"], df["Blood Pressure"])

    # Generates the prediction model for the dataframe
    predictionModel = LinearRegression(df)

    # Plots the Regression line
    x = [i for i in range(16, 70)]
    y = [predictionModel(i) for i in range(16, 70)]
    plt.plot(x, y)

    # labelling Graph
    plt.xlabel("Age")
    plt.ylabel("Blood Pressure (BP)")
    plt.title("Correlation Between Age and Blood Pressure")

    plt.show()
