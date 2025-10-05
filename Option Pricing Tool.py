import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from datetime import datetime


class MarketData:
    def __init__(self, rate: float, volatility: float):
        """
        rate: risk-free rate (continuous compounding, annualized)
        volatility: flat volatility (annualized)
        """
        self.rate = rate
        self.volatility = volatility


class Derivative:
    def __init__(self,
                 underlying_asset: str,
                 contract_size: float,
                 expiration_date: str,
                 strike_price: float,
                 premium: float = None):

        #expiration_date: string in format 'YYYY-MM-DD'

        self.underlying_asset = underlying_asset
        self.contract_size = contract_size
        self.expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        self.strike_price = strike_price
        self.premium = premium


class Option(Derivative):
    def __init__(self, option_type: str, style: str, spot_price: float, **kwargs):
        """
        option_type: "call" or "put"
        style: "european" or "american"
        spot_price: current underlying spot
        """
        super().__init__(**kwargs)
        self.option_type = option_type.lower()
        self.style = style.lower()
        self.spot_price = spot_price

    def payoff(self, spot_at_expiry: float):
        """ Payoff at maturity, not discounted. """
        if self.option_type == "call":
            return max(0, spot_at_expiry - self.strike_price) * self.contract_size
        elif self.option_type == "put":
            return max(0, self.strike_price - spot_at_expiry) * self.contract_size
        else:
            raise ValueError("Unknown option type")


class GetDate:
    def __init__(self, current_date: str):
        self.current_date = datetime.strptime(current_date, "%Y-%m-%d")

class Heston:
    def __init__(self, kappa, theta, sigma, rho, v0, m):
        self.call_price = None
        self.put_price = None
        self.kappa = kappa #Mean reversion rate
        self.theta = theta #Long term average volatility
        self.sigma = sigma #Vol of vol
        self.rho = rho #Correlation between asset price and volatility
        self.v0 = v0 #Initial volatility
        self.m = m #Shifts the values in the integral to be able to be evaluated and does not affect option price

    def phi(self, u, T, r, S0):
        kappa, theta, sigma, rho, v0 = self.kappa, self.theta, self.sigma, self.rho, self.v0

        d = np.sqrt((rho * sigma * 1j * u - kappa) ** 2 + (sigma ** 2) * (1j * u + u ** 2))
        g = (kappa - rho * sigma * 1j * u - d) / (kappa - rho * sigma * 1j * u + d)

        C = (kappa * theta / sigma ** 2) * ((kappa - rho * sigma * 1j * u - d) * T
                                            - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g)))
        D = ((kappa - rho * sigma * 1j * u - d) / sigma ** 2) * ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))

        return np.exp(C + D * v0 + 1j * u * (np.log(S0) + r * T))

    def price(self, option: Option, market_data: MarketData, date: GetDate):
        K = option.strike_price
        u = np.log(K)
        mu = 1j * u + u ** 2
        alpha = 2
        S0 = option.spot_price
        r = market_data.rate
        T = (option.expiration_date - date.current_date).days / 365.0
        if option.option_type == "call":
            integrand = lambda u: np.real(np.exp(-1j*u*np.log(K))*(self.phi(u - 1j*(alpha + 1), T, r, S0 )) / ((1j*np.log(K)+ alpha)*(1j*np.log(K) + alpha + 1)))
            integral, error= np.real(quad(integrand, 0, np.inf, limit=200))
            self.call_price = np.exp(-r * T) * S0 * np.exp(-alpha*np.log(K)) / np.pi * integral
            price = self.call_price
        else:
            raise ValueError("Invalid option type")

        return price

class BlackScholes:
    def __init__(self):
        self.call_price = None
        self.put_price = None
        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega = None
        self.rho = None

    def price(self, option: Option, market_data: MarketData, date: GetDate):
        """ Price European call/put and compute Greeks using Black-Scholes. """

        # time to expiry in years
        T = (option.expiration_date - date.current_date).days / 365.0
        if T <= 0:
            # expired option = intrinsic value
            return option.payoff(option.spot_price)

        S0 = option.spot_price
        K = option.strike_price
        r = market_data.rate
        sigma = market_data.volatility

        # d1, d2
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # option prices
        if option.option_type == "call":
            self.call_price = norm.cdf(d1) * S0 - norm.cdf(d2) * K * np.exp(-r * T)
            price = self.call_price
        elif option.option_type == "put":
            self.put_price = norm.cdf(-d2) * K * np.exp(-r * T) - norm.cdf(-d1) * S0
            price = self.put_price
        else:
            raise ValueError("Invalid option type")

        # Greeks
        self.gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        self.vega = S0 * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% change
        if option.option_type == "call":
            self.theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            self.rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            self.delta = norm.cdf(d1)
        else:
            self.theta = (-S0 * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                          + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            self.rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            self.delta (norm.cdf(d1) - 1)

        return price

option = Option(option_type="call",
                style="european",
                spot_price=100,
                underlying_asset="Sample",
                contract_size=150,
                expiration_date="2025-12-31",
                strike_price=100)

market = MarketData(rate=0.05, volatility=0.2)
today = GetDate("2025-01-01")

# Price with Black-Scholes
bs = BlackScholes()
price = bs.price(option, market, today)

print("Price:", price)
print("Delta:", bs.delta, "Gamma:", bs.gamma, "Vega:", bs.vega,
      "Theta:", bs.theta, "Rho:", bs.rho)

#Inaccurate but allows for calculation
h = Heston(kappa = 2,
           theta = 0.2,
           sigma = 0.4,
           rho = 0.6,
           v0 = 0.3,
           m = 0)

T = (option.expiration_date - today.current_date).days / 365.0
price2 = h.price(option, date = today, market_data=market)
price2 = price2/option.contract_size

print("Heston Price:", price2)

