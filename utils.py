"""
Author: Vlad Carare
Date: 14.04.2023
Description:
This package contains various methods used in the simulation.
"""

import matplotlib.pyplot as plt
import numpy as np
import string
import random
from matplotlib.lines import Line2D


def multiply_vector_matrix(vector, matrix):
    """
    Multiply a vector by a matrix element-wise.
    And performs a test to make sure the input is a vector and the output is a matrix.

    :param vector: a numpy array of shape (n,)
    :param matrix: a numpy array of shape (n, m)
    :return: a numpy array of shape (n, m)
    """
    if not len(vector.shape) == 1:
        raise TypeError('Vector must be a 1D array.')
    if not len(matrix.shape) == 2:
        raise TypeError('Matrix must be a 2D array.')
    if not vector.shape[0] == matrix.shape[0]:
        raise ValueError(
            'The number of rows in the vector must be equal to the number of rows in the matrix.')
    return np.einsum('i,ij->ij', vector, matrix)


def read_data(file_name, n_months=12, n_currencies=4):
    """
    Read in the data from the file.

    :param file_name: the name of the file to read in
    :param n_months: the number of months in the data
    :param n_currencies: the number of currencies in the data
    :return: a numpy array of shape (n_months, n_currencies, n_currencies)
    """

    # Open the file and delete any new line characters
    with open(file_name, 'r') as f:
        data = f.read().replace('\n', '').replace(
            '[', '').replace(']', '').replace(' ', '').split(',')
        # Convert the data to a numpy array of floats
        data = np.array(data, dtype=float)[:(n_months * n_currencies * n_currencies)]
        # Reshape the data to the correct shape
        data = data.reshape((n_months, n_currencies, n_currencies))
        assert data.shape[1:][0] == n_currencies, 'The shape of the exchange rates matrix is not consistent with the number of currencies.'
        assert data.shape[1:][1] == n_currencies, 'The shape of the exchange rates matrix is not consistent with the number of currencies.'
        # Return the data
        return data

def print_transaction_history_given_nodes_list(transaction_path, exchange_rates, amounts,currency_order):
    """
    Print the transaction history given a list of nodes.
    :param transaction_path: a list of nodes
    :param exchange_rates: a numpy array of shape (n_months, n_currencies, n_currencies)
    :param amounts: a numpy array of shape (n_months, n_currencies)
    :param currency_order: a list of currencies
    :return: transaction history
    """
    transaction_history = []
    print('Month'.ljust(6), 'From'.ljust(5), 'To'.ljust(
        5), 'Exchange Rate'.ljust(14), 'Amount'.ljust(19))
    for i, node in enumerate(transaction_path[:-1]):
        month = i
        source_currency = currency_order[node]
        target_currency = currency_order[transaction_path[i + 1]]
        exchange_rate = exchange_rates[month]
        amount = amounts[month]

        transaction_entry = (month, source_currency, target_currency, exchange_rate, amount)
        transaction_history.append(transaction_entry)

        print(str(month).ljust(6), source_currency.ljust(5), target_currency.ljust(
            5), str(exchange_rate).ljust(14), str(amount).ljust(19))
    
    return transaction_history

def generate_exchange_rate_matrices(n_currencies, n_months, enforce_reversibility=False):
    """
    Generate a list of random exchange rate matrices.
    :param n_currencies: the number of currencies
    :param n_months: the number of months
    :return: a numpy array of shape (n_months, n_currencies, n_currencies)
    """
    exchange_rates = np.random.normal(1, 0.1, (n_months, n_currencies, n_currencies))
    [np.fill_diagonal(exchange_rates[i], 1) for i in range(n_months)]
    assert exchange_rates[0][0][0] == 1 and exchange_rates[0][0][1] != 1, 'Diagonal elements are not 1 as they should.'

    if enforce_reversibility:
        # We enforce that doing the same transaction in reverse gives the original result.
        for i in range(n_months):
            for j in range(n_currencies):
                for k in range(j+1, n_currencies):
                    exchange_rates[i, j, k] = 1 / exchange_rates[i, k, j]
    return exchange_rates


def generate_strings(n_currencies):
    """
    Generate a list of non-repeating n_currencies random strings.
    :param n_currencies: the number of strings
    :return: a list of n_currencies strings
    """
    # Create a list of random currencies
    result = {
        # For each string, join 3 random upper case letters
        ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
        for _ in range(n_currencies-1)
    }
    result = ['GBP'] + list(result)
    # Check if the length of the set is equal to n_currencies
    if len(result) == n_currencies:
        return result
    else:
        # Repeat the process until it is
        return generate_strings(n_currencies)
    
def print_transaction_history(transaction_history):
    """
    Print the transaction history in a nice format, with a header explaining the different terms. We allocate 3 spaces for the month, 7 spaces for the source currency, 7 spaces for the final currency, 12 spaces for the exchange rate, and 19 spaces for the amount.
    :param transaction_history: a list of tuples, where each tuple is of the form (month, source_currency, target_currency, exchange_rate, amount)
    :return: None
    """
    print('Month'.ljust(6), 'From'.ljust(5), 'To'.ljust(5), 'Exchange Rate'.ljust(14), 'Amount'.ljust(19))
    for transaction in transaction_history:
        print(str(transaction[0]).ljust(6), transaction[1].ljust(5), transaction[2].ljust(5), str(transaction[3]).ljust(14), str(transaction[4]).ljust(19))


def create_plot_of_transation_histories(transaction_histories, currency_order, legend_labels = None):
    """
    Here we are plotting transaction paths given the history.
    This is only an experimental function, and only works when we have 12 months, with 4 currencies, as the positions of the nodes are hardcoded.
    :param currency_holdings: a list of CurrencyHolding objects
    :return: a plot of the top the transaction paths
    """
    # Get paths from transaction history, where each path is a list of size 12, where each i-th entry is the node it touches at layer i. 
    # Nodes are specified by their index in the currency_order list. 'GBP' should map to 0, 'USD' to 1, 'EUR' to 2, 'JPY' to 3.

    if not isinstance(transaction_histories,list):
        transaction_histories = [transaction_histories]

    transaction_histories = transaction_histories[::-1]
    legend_labels = legend_labels[::-1]

    print()
    print()
    print('Making plot of transaction histories')
    print()

    # Go through transaction histories and get the paths
    paths = []
    amounts = []
    for transaction_history in transaction_histories:
        paths.append(get_path(transaction_history, currency_order)[0])
        amounts.append(get_path(transaction_history, currency_order)[1])

    print(paths)

    x_range = np.linspace(0.05,0.95,13)
    # Convert the elements of a path list such that they matches the positions of the nodes in the plot
    def convert_path(path):
        new_path = []
        for i in range(len(path)):
            if i == 12:
                new_path.append(45)
                continue
            new_path.append(i%12 + path[i]*11)
        return new_path
    # Convert the paths to match the positions of the nodes in the plot
    new_paths = []
    for path in paths:
        new_paths.append(convert_path(path))

    amounts_labels = [item for sublist in amounts for item in sublist]
    new_paths_labels = [item for sublist in new_paths for item in sublist]
    amounts_labels = list(zip(amounts_labels, new_paths_labels))


    # Define the coordinates of the nodes - these are hardcoded
    x_range = np.linspace(0.05, 0.95, 13)
    x = np.concatenate([[x_range[0]], np.tile(x_range[1:-1], 4), [x_range[-1]]])
    y = np.array([0.5] + [0.5]*11 + [0.633]*11 + [0.766]*11 + [0.9]*11 + [0.5])

    # Define the labels of the nodes - their indices
    labels = [str(i) for i in range(len(x))]

    # Define the paths as lists of node indices
    paths = new_paths

    # Define a color map for the paths
    colors = [  "yellow", "red", "blue", "green"]

    # Define the line style for the paths
    line_styles = [(0, (5, 10)),
                   (0, (3, 5, 1, 5, 1, 5)),
                   '--','-'][::-1]
    
    # Define the decreasing line width for the paths
    line_widths = [0.01, 0.0065, 0.003, 0.0017]
    head_widths = [0.019, 0.016, 0.013, 0.01]

    # Draw the graph with matplotlib
    plt.figure(figsize=(14, 5))

    # Draw the nodes as circles
    plt.scatter(x, y, s=10)

    # Draw the node labels, but for node 45 change the position every time to avoid overlapping
    # Note for some exchange rates it may happen that paths meet before the last node (node 45), in that that case the labels will overlap, this needs fixing

    node_45_iter_displacement = (x for x in np.linspace(-0.05, 0.05, 4))
    for item in amounts_labels:
        # print(item)
        label, i = item
        # print(label, i)

        label = str(round(label, 3))
        if i == 45:
            displacement = next(node_45_iter_displacement)
            plt.text(x[i]+0.02, y[i]+displacement+0.02, s=label, fontsize=10)
            continue
        plt.text(x[i]-0.02, y[i]+0.02, s=label, fontsize=10)


    # Draw the node labels
    # for i, label in enumerate(labels):
    #     plt.text(x[i]+0.02, y[i]+0.02, s=label
    #              , fontsize=10)


    # remove frame
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    # mask the x-axis but keep the ticks
    plt.gca().spines['bottom'].set_visible(False)

    # set x ticks to be at the nodes, with labels of 1 to 12
    plt.xticks(x_range, [str(i) for i in range(13)])
    # set y ticks to be at the nodes, with labels from currency_order
    plt.yticks([0.5, 0.633, 0.766, 0.9], currency_order)


    #remove x ticks keeping the labels
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=True, pad=20)
    plt.xlabel('Month', fontsize=15)

    #remove y ticks keeping the labels
    plt.tick_params(axis='y', which='both', left=False,
                    right=False, labelleft=True, pad=20)
    plt.ylabel('Currency', fontsize=15)

    # Draw the edges as arrows
    for i in range(len(paths)):
        for j in range(len(paths[i]) - 1):
            plt.arrow(x[paths[i][j]], y[paths[i][j]],  # Start point
                    x[paths[i][j+1]] - x[paths[i][j]], y[paths[i]
                                                        [j+1]] - y[paths[i][j]],  # End point
                    width=line_widths[i],
                    length_includes_head=True,
                    head_width=head_widths[i],
                    head_length=head_widths[i]*2,
                    color=colors[i],
                    ls=line_styles[i])

    plt.title('Best transaction paths according to different algorithms', fontsize=20, pad=20)
    if legend_labels is not None:
        # Draw the legend with the line styles and colors using Line2D
        legend_elements = [Line2D([0], [0], color=colors[i], lw=3, label=legend_labels[i]) for i in range(len(legend_labels))]
        # Place the legend in the upper right corner outside the plot
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=15)


    # Save the plot such that the labels are not cut off
    plt.savefig('OptimalVsGreedyVsBruteForceTransactionPaths.png', bbox_inches='tight')


def get_path(transaction_history, currency_order, number_months=12):
    """
    Given a transaction history, n a list of paths and a list of amounts for each path.
    :param transaction_history: a list of tuples (month, source_currency, target_currency, exchange_rate, amount)
    :param currency_order: a list of currencies in the order they appear in the graph
    :return: the path and the list of amounts
    """
    path = []
    amounts = [1] # the first amount is 1
    for month, source_currency, target_currency, exchange_rate, amount in transaction_history:
        path.append(currency_order.index(source_currency))
        amounts.append(amount)
        if month==(number_months-1):
            path.append(0) # we end with GBP
    return path, amounts