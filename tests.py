"""
Author: Vlad Carare
Date: 14.04.2023
Description:
This class contains various tests to be run with pytest.
"""
import utils
from trading_simulation import Simulation
import numpy as np
import pytest

@pytest.mark.parametrize("is_greedy", [False, True])
def test_transaction_history_consistency(is_greedy):
    """
    Tests that the transaction history is consistent with the current holdings
    :param currency_holdings: a list of currency holdings
    :return: None
    """

    n_months = 8  # feel free to modify this, but bear in mind the brute force algorithm will take a long time for n_months > 8
    n_currencies = 4
    exchange_rates = utils.generate_exchange_rate_matrices(n_currencies, n_months)
    currency_order = utils.generate_strings(n_currencies)

    sim = Simulation(exchange_rates, currency_order, is_greedy=is_greedy)
    sim.run()

    currency_holdings = sim.currency_holdings
    # If the simulation is greedy, all the non-profitable holdings were set to 0, so we need to remove them from the list of currency holdings
    if is_greedy:
        currency_holdings = [currency_holding for currency_holding in sim.currency_holdings if currency_holding.amount != 0]

    # now check that following the transaction histories, the current holdings are correct. check against the exchange rates matrices
    for currency_holding in currency_holdings:
        checked_amount = 1
        for transaction in currency_holding.transaction_history:
            month, source_currency, final_currency, exchange_rate, amount_in_history = transaction
            checked_amount *= exchange_rate
            assert checked_amount == amount_in_history, 'The amount in the transaction history does not equal source currency * exchange rate, as it should.'
        print(currency_holding.transaction_history)
        print(currency_holding.currency)
        print(amount_in_history)
        print(currency_holding.amount)
        print(checked_amount)
        assert amount_in_history == currency_holding.amount
        assert checked_amount == currency_holding.amount


def test_multiply_vector_matrix():
    """
    Test the multiply_vector_matrix function.
    """
    a = np.array([1.1, 0.9])
    b = np.array([[1, 1.1], [1.5, 1]])
    result = np.array([[1.1, 1.21], [1.35, 0.9]])
    # check 2 2D arrays are equal element-wise within machine precision
    assert np.allclose(utils.multiply_vector_matrix(a, b), result)


@pytest.mark.parametrize("n_currencies", [2, 3, 4, 5])
def test_simulation(n_currencies):
    """
    Test the simulation.
    :param n_currencies: the number of currencies
    :return: None
    """
    n_months = 8 # feel free to modify this, but bear in mind the brute force algorithm will take a long time for n_months > 8
    exchange_rates_list = utils.generate_exchange_rate_matrices(n_currencies, n_months)
    currency_order = utils.generate_strings(n_currencies)

    optimal_sim = Simulation(exchange_rates_list, currency_order)
    optimal_sim.run()
    optimal_holdings = optimal_sim.current_holdings
    maximum_GBP_amount_from_optimal_algorithm = max(optimal_holdings)

    greedy_sim = Simulation(exchange_rates_list, currency_order, is_greedy=True)
    greedy_sim.run()
    greedy_holdings = greedy_sim.current_holdings
    maximum_GBP_amount_from_greedy_algorithm = max(greedy_holdings)

    _, maximum_GBP_amount_from_brute_force_algorithm = Simulation.brute_force_best_transaction_path(exchange_rates_list, currency_order, n_months)

    # check that the solutions of the optimal and brute force algorithms are the same and an upper bound on the greedy algorithm
    assert maximum_GBP_amount_from_optimal_algorithm == maximum_GBP_amount_from_brute_force_algorithm, 'The optimal and brute force algorithms do not give the same result.'
    assert maximum_GBP_amount_from_brute_force_algorithm >= maximum_GBP_amount_from_greedy_algorithm, 'The greedy algorithm should not give a better result than the brute force approach.'

