__version__ = "1.0"

""" all in mks """
import numpy as np
import sympy as sp
# ###################
# ## Common Constants

# use numpy
pi = np.pi
# planck's constant ?
h = 6.6260700e-34
# reduced planck's constant ?
hbar = h / (2 * pi)
# Boltzmann's constant ?
k_B = 1.3806488e-23
# speed of light (exact)
c = 299792458
# Stephan-Boltzmann constant, ?
sigma = 5.6704e-8
# atomic mass unit, ?
amu = 1.6605390e-27
# gravity acceleration near earth, inexact
g = 9.80665
# gravitational constant (inexact) (nist)
G = 6.67408e-11
# earth mass (inexact)
m_Earth = 5.972e24
r_Earth = 6.371e6
# fundamental charge (charge of electron & proton), in coulombs, inexact
qe = 1.6021766208e-19
# Bohr Radius, in m
a0 = 0.52917721067e-10
# Electric constant, vacuum permittivity, in Farads per meter, exact
epsilon0 = 8.854187817e-12
# Magnetic Constant, vacuum permeability, in Henrys / meter or newtons / Amp^2, exact
mu0 = 4e-7 * pi

# Pauli Matrices
X = sigma_x = sp.Matrix([[0, 1], [1, 0]])
Y = sigma_y = sp.Matrix([[0, -1j], [1j, 0]])
Z = sigma_z = sp.Matrix([[1, 0], [0, -1]])
# Hadamard
H = hadamard = sp.Matrix([[1, 1], [1, -1]])
# Phase Gate
S = phaseGate = sp.Matrix([[1, 0], [0, 1j]])

def phaseShiftGate(phi):
    return sp.Matrix([[1, 0], [[0, sp.exp(1j * phi)]]])

# ######################
# ### Rubidium Constants

# Most of these come from Steck's Rb87 paper.

# rubidium 87 mass (inexact)
Rb87_M = 86.909180527 * amu
# Measured, inexact
Rb87_GroundStateSplitting = 6.83468261090429e9
Rb87_GroundStateSplitting_Uncertainty = 9e-5
Rb87_Ground_ToF2 = 2.56300597908911e9
Rb87_Ground_ToF2_Uncertainty = 4e-5
Rb87_Ground_ToF1 = Rb87_Ground_ToF2 - Rb87_GroundStateSplitting
Rb87_Ground_ToF1_Uncertainty = 6e-5


def Rb87_Ground_State_Shift(F):
    """
    Based on the ground state splitting.
    """
    if F==1:
        return Rb87_Ground_ToF1
    elif F==2:
        return Rb87_Ground_ToF2
    else:
        raise ValueError("Invalid argument for ground state manifold")


# Approximate Lande` G-factors (g_F) in Hz/Gauss
Rb87_Ground_F2_g_F = 0.70e6
Rb87_Ground_F1_g_F = -0.70e6
# lifetimes, in s
Rb87_D1_Lifetime = 27.70e-9
Rb87_D1_Lifetime_Uncertainty = 0.04e-9
Rb87_D2_Lifetime = 26.24e-9
Rb87_D2_Lifetime_Uncertainty = 0.04e-9
# linewidths, in s^-1
Rb87_D1Gamma = 36.10e6
Rb87_D1Gamma_Uncertainty = 0.05e6
Rb87_D2Gamma = 38.11e6
Rb87_D2Gamma_Uncertainty = 0.06e6
# splittings of the excited 5^2 P_(3/2) state (D2 Line), in Hz
Rb87_5P32_ToF3 = 193.7408e6
Rb87_5P32_ToF3_Uncertainty = 4.6e3
Rb87_5P32_ToF2 = -72.9113e6
Rb87_5P32_ToF2_Uncertainty = 3.2e3
Rb87_5P32_ToF1 = -229.8518e6
Rb87_5P32_ToF1_Uncertainty = 5.6e3
Rb87_5P32_ToF0 = -302.0738e6
Rb87_5P32_ToF0_Uncertainty = 8.8e3

def Rb87_D2_Excited_State_Shift(Fp):
    """
    Shifts based on the excited state splittings
    """
    if Fp == 3:
        return Rb87_5P32_ToF3
    elif Fp == 2:
        return Rb87_5P32_ToF2
    elif Fp == 1:
        return Rb87_5P32_ToF1
    elif Fp == 0:
        return Rb87_5P32_ToF0
    else:
        raise ValueError("Invalid argument for D2 excited state manifold.")
        
Rb87_5P12_ToF2 = 306.246e6
Rb87_5P12_ToF2_Uncertainty = 11e3
Rb87_5P12_ToF1 = -510.410e6
Rb87_5P12_ToF1_Uncertainty = 19e3

def Rb87_D1_Excited_State_Shift(Fp):
    """
    Shifts based on the excited state splittings
    """
    if Fp == 2:
        return Rb87_5P12_ToF2
    elif Fp == 1:
        return Rb87_5P12_ToF1
    else:
        raise ValueError("Invalid argument for D1 excited state manifold. (Fp=1 or Fp=2).")

# for far-detuned approximations only.
# strictly, I should probably weight by Clebsch-Gordon coefficients or something to get
# a better far-detuned approximation.
Rb87_AvgGamma = (Rb87_D1Gamma + Rb87_D2Gamma)/2
# in mW/cm^2, 2-3', resonant & isotropic light.
Rb87_I_Sat_ResonantIsotropic_2_to_3 = 3.57713
Rb87_I_Sat_ResonantIsotropic_2_to_3_Uncertainty = 0.00074
Rb87_I_Sat_FarDetunedD2Pi = 2.50399
Rb87_I_Sat_FarDetunedD2Pi_Uncertainty = 0.00052
Rb87_I_Sat_FarDetunedD1Pi = 4.4876
Rb87_I_Sat_FarDetunedD1Pi_Uncertainty = 0.0031
# for cycling light specifically.
Rb87_I_Sat_ResonantSigmaP_22_to_33 = 1.66933
Rb87_I_Sat_ResonantSigmaP_22_to_33_Uncertainty = 0.00035

# wavelengths are in vacuum.
# in m
Rb87_D2LineWavelength = 780.241209686e-9
Rb87_D2LineWavelengthUncertainty = 1.3e-17
# in Hz (1/s)
Rb87_D2LineFrequency = 384.2304844685e12
Rb87_D2LineFrequencyUncertainty = 6.2e3
Rb87_D2_F2toFp3 = Rb87_D2LineFrequency - Rb87_Ground_ToF2 + Rb87_5P32_ToF3
Rb87_D2_F1toFp0 = Rb87_D2LineFrequency - Rb87_Ground_ToF1 + Rb87_5P32_ToF0

def Rb87_D2_Transition_Freq(F, Fp):
    return Rb87_D2LineFrequency - Rb87_Ground_State_Shift(F) + Rb87_D2_Excited_State_Shift(Fp)                

# etc.
Rb87_D1LineWavelength = 794.9788509e-9
Rb87_D1LineWavelengthUncertainty = 8e-16
Rb87_D1LineFrequency = 377.1074635e12
Rb87_D1LineFrequencyUncertainty = 0.4e6

# Note that the reduced dipole matrix values are taken from the various tables in:  
# Arora and Sahoo, Phys. Rev. A 86 033416 (2012)
# rdme = reduced dipole matrix element, the raw number is in atomic units (i.e. SI value / (e*a0)),
# Evaluated value is in SI units. First index is the initial state, second is the final state.
Rb87_Transition_rdme = {
    '5S12':{
        '5P12': 4.227 * qe * a0,
        '6P12': 0.342 * qe * a0,
        '7P12': 0.118 * qe * a0,
        '8P12': 0.061 * qe * a0,
        '9P12': 0.046 * qe * a0,

        '5P32': 5.977 * qe * a0,
        '6P32': 0.553 * qe * a0,
        '7P32': 0.207 * qe * a0,
        '8P32': 0.114 * qe * a0,
        '9P32': 0.074 * qe * a0,
    },
    '5P12':{
        '5S12': 4.227 * qe * a0,
        '6S12': 4.144 * qe * a0,
        '7S12': 0.962 * qe * a0,
        '8S12': 0.507 * qe * a0,
        '9S12': 0.333 * qe * a0,
        #'10S12': 0.235 * mc.qe * mc.a0,

        '4D32': 8.069 * qe * a0,
        '5D32': 1.184 * qe * a0,
        '6D32': 1.002 * qe * a0,
        '7D32': 0.75 * qe * a0,
        '8D32': 0.58 * qe * a0,
        '9D32': 0.45 * qe * a0,
    },
    '5P32':{
        '5S12': 5.977 * qe * a0,
        '6S12': 6.048 * qe * a0,
        '7S12': 1.363 * qe * a0,
        '8S12': 0.714 * qe * a0,
        '9S12': 0.468 * qe * a0,
        #'10S12': 0.330 * mc.qe * mc.a0,

        '4D32': 3.65 * qe * a0,
        '5D32': 0.59 * qe * a0,
        '6D32': 0.48 * qe * a0,
        '7D32': 0.355 * qe * a0,
        '8D32': 0.272 * qe * a0,
        '9D32': 0.212 * qe * a0,

        '4D52': 10.89 * qe * a0,
        '5D52': 1.76 * qe * a0,
        '6D52': 1.42 * qe * a0,
        '7D52': 1.06 * qe * a0,
        '8D52': 0.81 * qe * a0,
        '9D52': 0.593 * qe * a0,
    },
}


# The level energies are taken from table 1, the NIST column, in:  
# M Safronova and U Safronova, Phys. Rev. A 83 052508 (2011)
# Note that the latter paper also has a lot of dipole matrix values for higher-state->higher-state transitions.

# e2 on all numbers here is to convert from cm^-1 to m^1. Original source reported in cm^-1 for some bad reason.
# E=0 = single-ionization energy. Raw numbers are in m^-1, evaluated value is in joules.
# Note that I'm somewhat confused about exactly where the source is getting their exact values. I'm having
# trouble getting the specific Rb87 data from the nist website...
# Index like 5P12 refers to state 5P_(1/2).
Rb87_Energies = {
    '5S12':-33691e2 * h * c,
    '6S12':-13557e2 * h * c,
    '7S12':-7379e2 * h * c,
    '8S12':-4644e2 * h * c,
    '9S12':-3192e2 * h * c,
    
    '5P12':-21112e2 * h * c,
    '6P12':-9976e2 * h * c,
    '7P12':-5856e2 * h * c,
    '8P12':-3856e2 * h * c,
    '9P12':-2732e2 * h * c,

    '5P32':-20874e2 * h * c,
    '6P32':-9898e2 * h * c,
    '7P32':-5821e2 * h * c,
    '8P32':-3837e2 * h * c,
    '9P32':-2721e2 * h * c,
    
    '4D32':-14335e2 * h * c,
    '5D32':-7990e2 * h * c,
    '6D32':-5004e2 * h * c,
    '7D32':-3411e2 * h * c,
    '8D32':-2469e2 * h * c,
    '9D32':-1869e2 * h * c,
    
    '4D52':-14336e2 * h * c,
    '5D52':-7987e2 * h * c,
    '6D52':-5001e2 * h * c,
    '7D52':-3409e2 * h * c,
    '8D52':-2468e2 * h * c,
    '9D52':-1868e2 * h * c,
}

# Scatteing lengths, numbers refer to F levels.
# From paper "Measurement of s-wave scattering lengths in a two-component Bose-Einstein condensate", M. Egorov et all 2013
Rb87_a22 = 95.44 * a0
Rb87_a11 = 98.006 * a0
Rb87_a12 = 97.66 * a0

# Rubidium Molecule Dissociation Constants (from the paper 
# "Dispoersion forces and long-range electronic transition dipole moments of alkali-metal dimer excited states" by Marinescu et al.)
# the number given is in atomic units, once multiplying by Eh and a0 this is in SI units. 
Eh = 4.35974381e-18
Rb87_C3 = 9.202*Eh*a0**3
Rb87_C6Sigma = 12.05e3*Eh*a0**6
Rb87_C6Pi = 8.047e3*Eh*a0**6
Rb87_C8Sigma_1 = 2.805e6*Eh*a0**8
Rb87_C8Sigma_m1 = 9.462e6*Eh*a0**8
Rb87_C8Pi_1 = 11.32e5*Eh*a0**8
Rb87_C8Pi_m1 = 4.203e5*Eh*a0**8

# #################
# ### Lab Constants

opBeamDacToVoltageConversionConstants = [8.5, -22.532, -1.9323, -0.35142]
# pixel sizes for various cameras we have
baslerScoutPixelSize = 7.4e-6
baslerAcePixelSize = 4.8e-6
andorPixelSize = 16e-6
dataRayPixelSize = 4.4e-6

# basler conversion... joules incident per grey-scale count.
# number from theory of camera operation
cameraConversion = 117e-18
# number from measurement. I suspect this is a little high because I think I underestimated the attenuation
# of the signal by the 780nm filter.
# C = 161*10**-18

# note: I don't know what the limiting aperture is, but I believe that it's a bit larger than either of these. (what?!?!?!?)
# This parameter could probably be extracted from Zeemax calculations.
# (in meters)

# ... what?!?!?!? these are both too big... I don't know why these are set to these numbers...
____sillLensInputApertureDiameter____ = 40e-3
____sillLensExitApertureDiameter____ = 40e-3

# need to look these up.
# tweezerBeamGaussianWaist = 4
# probeBeamGaussianWaist = ???

# ########################
# ### Trap Characteristics

# in nm
trapWavelength = 852e-9
# ( 0 for Pi-polarized trapping light (E along quantization axis) *)
EpsilonTrap = 0
# this is the complex spherical tensor representation of the trap polarization at the atom, the first
# entry is the sigma plus polarization component, the second is the pi-polarized component, and
# the third is the sigma minus polarization component.
uTrap = [0, 1, 0]

