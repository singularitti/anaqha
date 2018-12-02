from abc import abstractmethod

import numpy as np
import scipy.optimize as optimize

# What to export?
__all__ = [
    'EOS',
    'Birch',
    'Murnaghan',
    'BirchMurnaghan2nd',
    'BirchMurnaghan3rd',
    'BirchMurnaghan4th'
]


class EOS:
    def __init__(self, *init_params):
        self._params = init_params

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        self._params = value

    @staticmethod
    @abstractmethod
    def _free_energy_at(v, *params):
        ...

    @staticmethod
    @abstractmethod
    def _pressure_at(v, *params):
        ...

    def free_energy_at(self, v):
        if self._params is not None:
            return self._free_energy_at(np.array(v, float), *self._params)
        raise ValueError("The initial parameters of the eos is not set! Please set before use!")

    def pressure_at(self, v):
        if self._params is not None:
            return self._pressure_at(np.array(v, float), *self._params)
        raise ValueError("The initial parameters of the eos is not set! Please set before use!")

    def fit(self, xdata, ydata, option, maxiter=1):
        dispatch = {'f': self._free_energy_at, 'p': self._pressure_at}
        try:
            f = dispatch[option]
        except KeyError:
            raise ValueError("Option '{0}' not recognized!".format(option))

        for _ in range(maxiter):
            popt, pcov = optimize.curve_fit(f, xdata, ydata, p0=self._params)
            self._params = popt

        return popt, pcov


class Birch(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, bp0, f0=0):
        x = (v0 / v) ** (2 / 3) - 1
        xi = 9 / 16 * b0 * v0 * x ** 2
        return f0 + 2 * xi + (bp0 - 4) * xi * x

    @staticmethod
    def _pressure_at(v, v0, b0, bp0):
        x = v0 / v
        xi = x ** (2 / 3) - 1
        return 3 / 8 * b0 * x ** (5 / 3) * xi * (4 + 3 * (bp0 - 4) * xi)


class Murnaghan(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, bp0, f0=0):
        x = bp0 - 1
        y = (v0 / v) ** bp0
        return f0 + b0 / bp0 * v * (y / x + 1) - v0 * b0 / x

    @staticmethod
    def _pressure_at(v, v0, b0, bp0):
        return b0 / bp0 * ((v0 / v) ** bp0 - 1)


class BirchMurnaghan2nd(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, f0=0):
        f = ((v0 / v) ** (2 / 3) - 1) / 2
        return f0 + 9 / 2 * b0 * v0 * f ** 2

    @staticmethod
    def _pressure_at(v, v0, b0):
        f = ((v0 / v) ** (2 / 3) - 1) / 2
        return 3 * b0 * f * (1 + 2 * f) ** (5 / 2)


class BirchMurnaghan3rd(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, bp0, f0=0):
        eta = (v0 / v) ** (1 / 3)
        xi = eta ** 2 - 1
        return f0 + 9 / 16 * b0 * v0 * xi ** 2 * (6 + bp0 * xi - 4 * eta ** 2)

    @staticmethod
    def _pressure_at(v, v0, b0, bp0):
        eta = (v0 / v) ** (1 / 3)
        return 3 / 2 * b0 * (eta ** 7 - eta ** 5) * (1 + 3 / 4 * (bp0 - 4) * (eta ** 2 - 1))


class BirchMurnaghan4th(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, bp0, bpp0, f0=0):
        f = ((v0 / v) ** (2 / 3) - 1) / 2
        h = b0 * bpp0 + bp0 ** 2
        return f0 + 3 / 8 * v0 * b0 * f ** 2 * ((9 * h - 63 * bp0 + 143) * f ** 2 + 12 * (bp0 - 4) * f + 12)

    @staticmethod
    def _pressure_at(v, v0, b0, bpp0, bp0):
        f = ((v0 / v) ** (2 / 3) - 1) / 2
        h = b0 * bpp0 + bp0 ** 2
        return 1 / 2 * b0 * (2 * f + 1) ** (5 / 2) * ((9 * h - 63 * bp0 + 143) * f ** 2 + 9 * (bp0 - 4) * f + 6)


class Vinet(EOS):
    @staticmethod
    def _free_energy_at(v, v0, b0, bp0, f0=0):
        x = (v / v0) ** (1 / 3)
        xi = 3 / 2 * (bp0 - 1)
        return f0 + 9 * b0 * v0 / xi ** 2 * (1 + (xi * (1 - x) - 1) * np.exp(xi * (1 - x)))

    @staticmethod
    def _pressure_at(v, v0, b0, bp0):
        x = (v / v0) ** (1 / 3)
        xi = 3 / 2 * (bp0 - 1)
        return 3 * b0 / x ** 2 * (1 - x) * np.exp(xi * (1 - x))
