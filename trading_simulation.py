"""
Author: Vlad Carare
Date: 14.04.2023
Description:
This class represents a simulation of the evolving (as a result of transactions) system of a set of currency holding accounts, one of each currency.
"""

import itertools
import numpy as np
# Import the CurrencyHolding class from currency_holding.py
from currency_holding import CurrencyHolding
import utils

class Simulation:
    """
    The simulation class represents a simulation of the evolving (as a result of transactions) system of a set of currency holding accounts, one of each currency.
    """
    def __init__(self, exchange_rates_list, currency_order, is_greedy=False):
        """
        Initialises the simulation.
        :param exchange_rates_list: a list of exchange rate matrices, one for each month
        :param currency_order: a list of currencies in the order they appear in the exchange rate matrices
        :param is_greedy: a boolean indicating whether the simulation is greedy or optimal
        """

        if is_greedy:
            print('Initialising greedy simulation')
        else:
            print('Initialising optimal simulation')
        # We have as many currency holdings as there are currencies
        self.currency_holdings = [CurrencyHolding(
            currency, order, 1 if currency == 'GBP' else 0, []) for order, currency in enumerate(currency_order)]
        self.number_months = len(exchange_rates_list)
        self.exchange_rates_list = exchange_rates_list
        self.month = 0
        self.current_exchange_rate = exchange_rates_list[0]
        # Compile a numpy list of current holding amounts in order of currency_order
        self.current_holdings = np.array(
            [currency_holding.amount for currency_holding in self.currency_holdings])
        self.is_greedy = is_greedy

        print(self.current_holdings)





    def step(self):
        """
        This method performs a single step of the simulation, i.e. it performs the 16 transactions for the current month.
        """

        # Get the transaction results
        transaction_results = utils.multiply_vector_matrix(
            self.current_holdings, self.exchange_rates_list[self.month])
        print('Transaction results:\n', transaction_results)


        # The crucial step is that after the 16 transactions take place, we are only interested in the ones that have the maximum possible holding for each currency
        # Get the maximum possible holding for each currency, i.e. the maximum along each column
        max_holdings = np.max(transaction_results, axis=0)
        # Get the indices of the maximum possible holding for each currency, i.e. for each column, get the index of the maximum - that gives you the currency holding the maximum came from
        max_indices = np.argmax(transaction_results, axis=0)
        # Now that we have the indices in the currency_order list, we can get the holding account the maximum came from. The holding account is defined by the currency.
        source_currency_holdings = [self.currency_holdings[max_indices[i]]
                                    for i in range(len(max_indices))]
        print('Maximum possible holdings:', max_holdings)
        print('Source holdings of maximum possible current holdings:', [
              currency_holding.currency for currency_holding in source_currency_holdings])
        print()


        # We save a copy of the current state of the transaction histories
        previous_transation_histories = [list(
            currency_holding.transaction_history) for currency_holding in self.currency_holdings]
        # print('Previous transaction histories', previous_transation_histories)



        # Update the currency holdings.
        """
        For each currency holding, after each transaction round there are 2 posibilities:
            1. The most currency was attained by not doing an exchange, i.e. keeping the same currency.
                In this case, we just append the new transaction to the transaction history of the current holding.

            2. The most currency was attained by doing an exchange from another, source, currency.
                In this case, we delete the transaction history of the current holding and instead branch off from the transaction history of the source currency holding.

        This is because, for a given currency holding, we are only interested in the transaction history that resulted in the maximum amount of currency.
        """
        for current_currency_holding, new_amount, source_currency_holding in zip(self.currency_holdings, max_holdings, source_currency_holdings):
            print('Source currency holding:', source_currency_holding.currency)
            print('Current currency holding:',
                  current_currency_holding.currency)
            print(f'Maximum {current_currency_holding.currency} holding this month:', new_amount, current_currency_holding.currency,',')
            print(f'achieved by converting from {source_currency_holding.currency} at a rate of:', self.exchange_rates_list[self.month][source_currency_holding.order, current_currency_holding.order])
            print(f'Source currency holding had {source_currency_holding.amount} {source_currency_holding.currency} in the previous month.')
            
            # Update the amount of currency held
            current_currency_holding.amount = new_amount

            # Create the new transaction entry for the transaction history
            new_transaction = (self.month, source_currency_holding.currency, current_currency_holding.currency,
                               self.exchange_rates_list[self.month][source_currency_holding.order, current_currency_holding.order], new_amount)

            # If the transaction was an exchange between currencies, then we branch off the transaction history of the source currency holding
            if current_currency_holding.currency != source_currency_holding.currency:
                current_currency_holding.transaction_history = list(previous_transation_histories[
                    source_currency_holding.order])

            # Update the transaction history with the new transaction
            current_currency_holding.transaction_history.append(
                new_transaction)
            print('Transaction history:\n',
                  current_currency_holding.transaction_history)
            print()

        # Update the current holdings
        self.current_holdings = np.array(
            [currency_holding.amount for currency_holding in self.currency_holdings])




    def last_step(self):
        """
        At the end of the simulation, all the currency holdings are converted to GBP and the best performing one is picked.
        """

        print()
        currency_order = [currency_holding.currency for currency_holding in self.currency_holdings]
        # Get the maximum potential holdings in GBP
        max_holdings_in_gbp = self.current_holdings * \
            self.exchange_rates_list[self.month][:,
                                                 currency_order.index('GBP')]

        for source_currency_holding, max_GBP_amount in zip(self.currency_holdings, max_holdings_in_gbp):
            print('Source currency holding:',
                  source_currency_holding.currency)
            print('Maximum amount of GBP attainable after conversion:', max_GBP_amount)

            # Update the transaction histories
            source_currency_holding.amount = max_GBP_amount
            new_transaction = (self.month, source_currency_holding.currency, 'GBP',
                               self.exchange_rates_list[self.month][source_currency_holding.order, currency_order.index('GBP')], max_GBP_amount)
            source_currency_holding.transaction_history.append(new_transaction)

            print('Final transaction history:\n',
                  source_currency_holding.transaction_history)
            print()

        # Update the current holdings
        self.current_holdings = np.array(
            [currency_holding.amount for currency_holding in self.currency_holdings])


        holding_with_best_transaction_path = max(
            self.currency_holdings, key=lambda x: x.amount)
        print('')
        print('The best transaction path we could taken given the exchange rates is: ')
        holding_with_best_transaction_path.print_transaction_history()
            #   [str(transaction) for transaction in holding_with_best_transaction_path.transaction_history])
        print('The final maximum amount of GBP we would have is using optimal algorithm: ',
              holding_with_best_transaction_path.amount)
        
        print()
        



    def greedy_step(self):
        """
        This method performs a greedy step of the simulation. i.e one where we select the largest exchange rate possible at each step.
        """
        
        currency_order = [currency_holding.currency for currency_holding in self.currency_holdings]
        # Among the currency_holdings, find the currency holding with non-zero amount
        source_currency_holding = next(
            currency_holding for currency_holding in self.currency_holdings if currency_holding.amount != 0)
        

        source_currency = source_currency_holding.currency
        source_currency_index = currency_order.index(source_currency)
        print('Starting currency:', source_currency)
        print('Starting amount:', source_currency_holding.amount, source_currency)

        # Get the transaction results
        transaction_results = utils.multiply_vector_matrix(
            self.current_holdings, self.exchange_rates_list[self.month])
        # Since we only have a single currency at one time, we only consider the row of the transaction results that corresponds to the single currency
        transaction_results = transaction_results[source_currency_index, :]

        print('Possible transaction results:\n', transaction_results)

        # Get the maximum amount of currency attainable given our source currency holdings
        max_holdings = np.max(transaction_results)
        # Get the index of the currency that gives the maximum amount of currency
        max_currency_holding_index = np.argmax(transaction_results)
        
        # If it's the last step, then we convert all the currency to GBP
        if self.month == self.number_months-1:
            max_currency_holding_index = currency_order.index('GBP')
            max_holdings = transaction_results[max_currency_holding_index]

        current_currency_holding = self.currency_holdings[max_currency_holding_index]

        print('Maximum possible amount:', max_holdings, currency_order[max_currency_holding_index])
        print(source_currency_holding.currency, '->', current_currency_holding.currency)



        print('Old holdings:', self.current_holdings)
        # We transfer all the money into the currency acount that has the largest exchange rate
        source_currency_holding.amount = 0
        current_currency_holding.amount = max_holdings
        # Create the new transaction entry for the transaction history
        new_transaction = (self.month, source_currency_holding.currency, current_currency_holding.currency,
                           self.exchange_rates_list[self.month][source_currency_index, max_currency_holding_index], max_holdings)
        
        # Update the transaction history with the new transaction
        current_currency_holding.transaction_history = list(source_currency_holding.transaction_history) + [new_transaction]
        

        # Update the current holdings
        self.current_holdings = np.array(
            [currency_holding.amount for currency_holding in self.currency_holdings])
        print('Current holdings:', self.current_holdings)

        # If last step, then we print the final transaction history and the final amount of currency held
        if self.month == self.number_months-1:

            print(
                'The transaction path of the sub-optimal greedy algorithm is: ')
            current_currency_holding.print_transaction_history()
            print('The final maximum amount of GBP we would have using the greedy algorithm:',
                  current_currency_holding.amount)
            



    @staticmethod
    def brute_force_best_transaction_path(exchange_rates_list, currency_order, number_months):
        """
        Find the best transaction path using brute force.
        :param exchange_rates_list: A list of exchange rates matrices
        :param currency_order: A list of the currencies in the order they appear in the exchange rates matrices
        :param number_months: The number of months in the simulation
        """

        print('Finding the best transaction path using brute force.')

        numbers = list(range(len(currency_order)))

        # Define a list of all possible transaction paths
        transaction_paths = [path for path in itertools.product(
            numbers, repeat=number_months+1) if path[0] == 0 and path[-1] == 0]

        best_transaction_path = []
        best_transaction_path_exchange_rates = []
        best_transaction_path_amounts = []
        best_amount = 0


        for transaction_path in transaction_paths:
            # Define a variable to store the current amount of GBP, we always start with 1
            current_amount = 1
            transaction_path_exchange_rates = []
            transaction_path_amounts = []

            for i in range(len(exchange_rates_list)):

                # Perform transaction and update holdings
                current_exchange_rates_matrix = exchange_rates_list[i]
                old_currency = transaction_path[i]
                new_currency = transaction_path[i+1]
                exchange_rate = current_exchange_rates_matrix[old_currency][new_currency]
                current_amount *= exchange_rate
                transaction_path_exchange_rates.append(exchange_rate)
                transaction_path_amounts.append(current_amount)

            # Update the best amount of GBP and best transaction path
            if current_amount > best_amount:
                best_amount = current_amount
                best_transaction_path = transaction_path
                best_transaction_path_exchange_rates = list(
                    transaction_path_exchange_rates)
                best_transaction_path_amounts = list(transaction_path_amounts)

        print(
            'The transaction path of the brute-force algorithm is: ')
        transaction_history = utils.print_transaction_history_given_nodes_list(
            best_transaction_path, best_transaction_path_exchange_rates, best_transaction_path_amounts, currency_order)
        print('The final maximum amount of GBP we would have using the brute-force algorithm:', best_amount)
        print(f"---------------------------------------The end of the brute force algorithm.---------------------------------------")
        print('\n\n\n\n')

        return transaction_history, best_amount




    # Define a method to run the simulation
    def run(self):
        """
        Run the simulation.
        """

        # Perform multiple iterations of the simulation
        for self.month in range(self.number_months-1):

            # Print the current month, current holdings, and exchange rates
            print('Month:', self.month)
            print('Current holdings:', self.current_holdings)
            print('Exchange rates:\n', self.exchange_rates_list[self.month])
            
            # Call step method
            if self.is_greedy:
                self.greedy_step()
            else:
                self.step()
            # Print current month
            print()
            print(
                f"-----------------------------Month {self.month} complete.-----------------------------")
            print()


        self.month += 1          
        print('Month:', self.month)
        print('Current holdings:', self.current_holdings)
        print('Exchange rates:\n', self.exchange_rates_list[self.month])
        print() 

        # Run the last step
        if self.is_greedy:
            self.greedy_step()
        else:
            self.last_step()


        print()
        print(
            f"-----------------------------Month {self.number_months-1} complete.-----------------------------")
        print(f"---------------------------------------The end.---------------------------------------")
        print('\n\n\n\n')


        # Get the transaction history of the holding with the largest amount of GBP from the current algorithm
        holding_with_best_transaction_path = max(
            self.currency_holdings, key=lambda x: x.amount)
        best_transaction_history = holding_with_best_transaction_path.transaction_history
        best_amount = holding_with_best_transaction_path.amount

        return best_transaction_history, best_amount