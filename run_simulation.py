"""
Author: Vlad Carare
Date: 14.04.2023
Description:
This is my solution for the coding challenge for a position of Research Scientist. 
The task is to find the best strategy for a currency trader to maxime profit in GBP, given a retrospective list of exchange rates for a number of currencies.
"""
import os
from trading_simulation import Simulation
# Import all methods from the utils.py file
import utils

if __name__ == '__main__':

    # Define the default order of the currencies, data file name, number of months and n currencies. The user will be asked if this is correct.
    currency_order = ['GBP', 'USD', 'EUR', 'JPY']
    data_file = 'currency_data.txt'
    number_months = 12
    n_currencies = 4
    # Get inputs from the user: the data file, the number of months to simulate, and the order of the currencies.
    while True:
        try:
            use_default_data_file = input(
                f'Is the data file you want to use located in the current folder and named "{data_file}"? (y/n)\n')
            assert use_default_data_file.strip() in ['y','n'], 'Please only reply with y or n. Try again.'
            if use_default_data_file.strip() == 'n':
                data_file = input(
            'Please enter the name (or path) of the data file containing the exchange rate matrices in the format of the example.\n')            
            assert os.path.exists(
                data_file), 'The data file does not exist. Try again.'

            # Get the number of months from the user and make sure it can be converted to an integer.
            number_months = input(
                'Please enter the number of months you would like to simulate. It has be to an integer.\n')
            number_months = int(number_months)

            # Ask the user if the order of the currencies is correct.
            is_default_order_correct = input(
                f'Are these the currencies in the data file, in order? (y/n) \n{currency_order}\n')
            assert is_default_order_correct.strip() in ['y','n'], 'Please only reply with y or n. Try again.'
            if is_default_order_correct.strip() == 'n':
                # If the default order is not correct, ask the user to input the the number and correct order of currencies.
                n_currencies = input(
                    'Please enter the number of currencies you want to simulate. This should correspond to the data file.\n')
                n_currencies = int(n_currencies)
                currency_order = input(
                    'Please enter the correct order of the currencies in the data file, with no space, separated by commas. NOTE: one of them should be GBP.\n').split(',')
                assert len(currency_order) == n_currencies, 'The specified number of currencies does not match the currency order provided. Try again.'
                assert 'GBP' in currency_order, 'GBP must be one of the currencies. Try again. Also maybe make sure the input follows the format specified.'

            # Ask user if they want to run brute-force algorithm
            run_brute_force = input(
                'Do you want to run the brute-force algorithm? NOTE: For 4 currencies this takes only few seconds for less then 10 months, and about 4 minutes for 12 months. The result will be the same as the optimal one, so this is meant only as a test. (y/n)\n')
            assert run_brute_force.strip() in ['y','n'], 'Please only reply with y or n. Try again.'
            if run_brute_force.strip() == 'y':
                run_brute_force = True
            else:
                run_brute_force = False
            break
        except AssertionError as e:
            print(e)
            continue
        except ValueError as e:
            print("The value provided must be an integer. Try again.")
            continue

    print(n_currencies, number_months, currency_order, data_file)

    # Read in exchange rates matries from the data file
    exchange_rates_list = utils.read_data(data_file, n_months=number_months, n_currencies=n_currencies)

    # Create a simulation instance
    optimal_sim = Simulation(exchange_rates_list, currency_order)
    # Run the simulation
    optimal_transaction_path, optimal_amount = optimal_sim.run()

    # Now create a greedy simulation 
    greedy_sim = Simulation(exchange_rates_list, currency_order, is_greedy=True)
    # Run the greedy simulation
    greedy_transaction_path, greedy_amount = greedy_sim.run()

    # Now run the brute-force algorithm 
    # It can be simulated fast for 10 months. For 12 months, it takes 4 minutes.
    if run_brute_force:
        brute_force_transaction_path, brute_force_amount = Simulation.brute_force_best_transaction_path(exchange_rates_list, currency_order, number_months)

    # Print the results
    print('\n Results:')
    print('Optimal algorithm:')
    utils.print_transaction_history(optimal_transaction_path)
    print(f'Optimal return of GBP: {(optimal_amount-1)*100:.4f}%')
    print('Greedy algorithm:')
    utils.print_transaction_history(greedy_transaction_path)
    print(f'Greedy return of GBP: {(greedy_amount-1)*100:.4f}%')
    print(f'Optimal return of GBP: {(optimal_amount-1)*100:.4f}%')
    if run_brute_force:
        print('Brute-force algorithm:')
        utils.print_transaction_history(brute_force_transaction_path)
        print(f'Brute-force return of GBP: {(brute_force_amount-1)*100:.4f}%')
        print(f'Greedy return of GBP: {(greedy_amount-1)*100:.4f}%')
    

    # Create the plot for the transaction history of the greedy and optimal algorithms
    plot_results = False
    if plot_results:
        if run_brute_force:
            list_of_transaction_paths = [optimal_transaction_path, greedy_transaction_path, brute_force_transaction_path]
            legend_labels = ['Optimal','Greedy','Brute-force']
        else:
            list_of_transaction_paths = [optimal_transaction_path, greedy_transaction_path]
            legend_labels = ['Optimal','Greedy']
        utils.create_plot_of_transation_histories(list_of_transaction_paths, currency_order, legend_labels=legend_labels)
