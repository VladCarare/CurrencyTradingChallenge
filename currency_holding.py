import utils
class CurrencyHolding:
    """
    The currency holding class represents a currency holding for a given month.
    We keep the transaction history in a list of tuples (from_currency, to_currency, rate, amount).
    """

    def __init__(self, currency, order, amount, transaction_history):
        """
        The constructor for the currency holding class.
        """
        self._order = order
        self.currency = currency
        self.amount = amount
        self.transaction_history = transaction_history

    def __str__(self):
        """
        The string representation of the currency holding.
        """
        return f'Currency {self.currency} Amount {self.amount} \n Transaction History {self.transaction_history}'

    def __repr__(self):
        """
        The string representation of the currency holding.
        """
        return f'CurrencyHolding(currency={self.currency}, amount={self.amount}, transaction_history={self.transaction_history})'

    @property
    def currency(self):
        """
        Get the currency.
        """
        return self._currency

    @currency.setter
    def currency(self, currency):
        """
        Set the currency.
        """
        self._currency = currency

    @property
    def order(self):
        """
        Get the order.
        """
        return self._order

    @order.setter
    def order(self, order):
        """
        Set the order.
        """
        raise ValueError(f'Order is {order} and cannot be changed.')

    @property
    def amount(self):
        """
        Get the amount.
        """
        return self._amount

    @amount.setter
    def amount(self, amount):
        """
        Set the amount, making sure amount is convertible to float and positive.
        """
        try:
            amount = float(amount)
        except ValueError:
            raise ValueError('Amount must be convertible to float.')
        if not amount >= 0:
            raise ValueError('Amount must be positive.')
        self._amount = amount

    @property
    def transaction_history(self):
        """
        Get the transaction history.
        """
        return self._transaction_history

    @transaction_history.setter
    def transaction_history(self, transaction_history):
        """
        Set the transaction history.
        """
        self._transaction_history = transaction_history

    def print_transaction_history(self):
        """
        Print the transaction history in a nice format, with a header explaining the different terms. We allocate 3 spaces for the month, 7 spaces for the source currency, 7 spaces for the final currency, 12 spaces for the exchange rate, and 19 spaces for the amount.
        """
        utils.print_transaction_history(self.transaction_history)