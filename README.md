# Author: Vlad Carare
# Date: 14.04.2023
# Purpose:
The goal of this project is the offer the optimal solution to a coding challenge I completed.
The problem is the following:


Imagine that you are a currency trader. You would like to determine what would have been the maximum profit you could have achieved in the previous year. You trade with the following conditions, you may only trade at the beginning of each month and trades are free from any fees.

We will provide you with monthly returns for the available currency pairs (i.e. for currencies X and Y, if you bought currency Y at the start of the month having started with currency X, how much return would you have attained in terms of currency X). You must start and finish the year with all of your portfolio in GBP. We provide further explanation of a good/bad trade alongside example data below.

We would like you to write an algorithm to calculate the maximum possible return over the year.


# Code aims

The code implements 3 solutions and compares the results:
1. The optimal solution based on the arguments below
2. The greedy solution, where at each month the largest exchange rate is selected
3. The brute force method - which also results in the right answer, but at a much larger cost, so be weary if you want to run this.

# The 3 main assumptions

## Exchange rates are uncorrelated over time - which is empirically true.
This means there is no predefined path we should take. 

## Holding all money in one currency is always more profitable than diversifying.
There is absolutely no benefit in a mixed strategy even if you were allowed one (in this ideal case of no risk & fees). To see this, imagine that an optimal solution actually decides decide to branch out  at some time step and change it’s only holding (say in currency Ca) into two currencies, Cb and Cc, with some fraction f of the total Ca you had at that point going into Cb and the residual fraction (1 - f) going to Cc. Notice that this could happen at the first step. Then you can imagine this as starting two new subproblems with a smaller time horizon where one subproblem starts with currency Cb and needs to end in GBP and the other subproblem starts with currency Cc and needs to end in GBP. If one of this subproblems yields better GBP results than the other, say starting with Cb, then changing all of our original Ca to Cb (I.e. f = 1) will yield better end returns, violating the fact that the strategy is optimal. Therefore, both subproblems must yield exactly the same returns (when scaled by f and (1-f)). This implies that changing everything originally to one of the two currencies (say Cb by setting f = 1) will also be an optimal strategy. Hence there must exist an optimal strategy that has holdings on a single currency at all times and therefore there are no real benefits in mixing things rather than just making things more complicated. 

Additionally, by writing down the equation for a 3 month trade using only 2 currencies one can see that the partial derivatives with respect to the fractions of mixing of currencies do not depend on the corresponding fractions, and also do not change sign. This suggests optimal results are on the corners of the [0,1]^(number of mixing fractions) domain, i.e. that we either exchange all or none of the holdings at each month. 

Should it not be true, the approach I would suggest is to write the problem as a neural network with 4 nodes per layer, where the learneable weights are the mixing fractions and where they are multiplied by the corresponding exchange rates. Additionally, one would need to constrain the mixing fractions of the exchanges starting from the same currency to sum up to 1. This poses some difficulties, but supposing it could be done one would then apply optimisation techniques from the ML community to maximise the objective function: the end profit. 

## At each month, the best 4 paths are those that maximise the holding in each currency.
This builds on the previous assumption. Now assume that your potential transactions paths form a directed tree. At the first month you have 4 branches, as you start from only one currency. At the second month you'll have 4^2 branches. At third 4^3 branches and so on. At the n-th month you'll have 4^n branches. These branches can be split into 4 categories, according to which currency they end up in. Each category will contain 4^(n-1) branches. All branches in a category will have the same currency, but various amounts of it. Since we are interested in finding the path that maximises profit, we are naturally only interested in the branch within that category, that has the largest amount. There's absolutely no reason to keep track of the other ones and we are not losing out on anything, since the future transactions available to a branch in a category are the same for all branches in that category. As a result, for each category (each currency), at each month, we only keep track of the 4 paths that result in the maximum holdings for each currency, and discard the rest. 


# Code design

There are 2 main objects in this project: 
    Simulation - which contains the algorithms implemented and keeps track of the evolution. We create a different Simulation object for the optimal algorithm, for the greedy one and for the brute force one.
    CurrencyHolding - which represents an account holding currency X, with amount, and transaction histry.

The "run_simulation.py" file is used to start the simulation, and "utils.py" is used to stack various methods needed.
Finally, "tests.py" contains useful tests.

# Installation

conda create --name currencyChallenge


conda activate currencyChallenge


conda install matplotlib


conda install pytest

Alternatively use your favourite environment manager and do:

pip install matplotlib


pip install pytest

# Test your code

pytest tests.py 

This tests that the function for calculating transaction results works as expected against a know result, further it tests that transaction histories are consistent and finally that, for some randomly generated matrices, the optimal algorithm returns the same result as the brute-force, and that the greedy algorithm is never more efficient. 

# Inputs

The code will ask whether the user wishes to load the default file: "currency_data.txt", otherwise asks for the path to the file. 
The file should contain data in the format given in currency_data.txt, but a variety of input formats is accepted, as the code strips away '[]\n ' characters, splits by ',' and reshapes to the number of months and currencies selected by the user. 

# Run the code

python run_simulation.py

# Extras

You can get the nice looking plot in this folder by going to the line ~89 of the run_simulation.py code, and set plot_results=True.
NOTE: this only works when you have 12 months and 4 currencies as node positions are hardcoded, which is why I made it harder to change.

