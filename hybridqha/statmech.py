"""

"""

import numpy as np
from scipy.constants import hbar, Boltzmann

__all__ = [
    'bose_einstein_distribution',
    'subsystem_partition_function',
    'subsystem_free_energy',
    'subsystem_internal_energy',
    'subsystem_entropy',
    'subsystem_volumetric_specific_heat'
]


def _validate_frequency(frequency):
    if frequency < 0:
        raise ValueError("Negative frequency is not proper for QHA calculation!")


def bose_einstein_distribution(temperature, frequency):
    return 1 / (np.exp(hbar * frequency / (Boltzmann * temperature)) - 1)


def subsystem_partition_function(temperature, frequency):
    """
    Calculate the subsystem partition function of a single harmonic oscillator at a specific temperature.
    This is a vectorized function so the argument *frequency* can be an array.
    :param temperature: The temperature, in unit 'Kelvin'. Zero-temperature is allowed.
    :param frequency: The frequency of the harmonic oscillator, in SI unit. If the *frequency*
        is less than or equal to :math:`0`, directly return ``1`` as its subsystem partition function value.
    :return: The subsystem partition function of the harmonic oscillator.
    """
    _validate_frequency(frequency)

    if frequency == 0:
        return 1

    x = hbar * frequency / (Boltzmann * temperature)
    return np.exp(x / 2) / (np.exp(x) - 1)


def subsystem_free_energy(temperature, frequency):
    """
    Calculate Helmholtz free energy of a single harmonic oscillator at a specific temperature.
    This is a vectorized function so the argument *frequency* can be an array.
    :param temperature: The temperature, in unit 'Kelvin'. Zero-temperature is allowed.
    :param frequency: The frequency of the harmonic oscillator, in SI unit. If the *frequency*
        is less than or equal to :math:`0`, directly return ``0`` as its free energy.
    :return: Helmholtz free energy of the harmonic oscillator, with unit 'joule'.
    """
    _validate_frequency(frequency)

    if frequency == 0:
        return 0

    hw = hbar * frequency
    kt = Boltzmann * temperature
    return hw / 2 + kt * np.log(1 - np.exp(-hw / kt))


def subsystem_internal_energy(temperature, frequency):
    _validate_frequency(frequency)

    if frequency == 0:
        return Boltzmann * temperature

    hw = hbar * frequency
    return hw / 2 / np.tanh(hw / (2 * Boltzmann * temperature))


def subsystem_entropy(temperature, frequency):
    _validate_frequency(frequency)

    n = bose_einstein_distribution(temperature, frequency)
    return Boltzmann * ((1 + n) * np.log(1 + n) - n * np.log(n))


def subsystem_volumetric_specific_heat(temperature, frequency):
    _validate_frequency(frequency)

    if frequency == 0:
        return Boltzmann

    hw = hbar * frequency
    kt = 2 * Boltzmann * temperature
    return Boltzmann * (hw / np.sinh(hw / kt) / kt) ** 2
