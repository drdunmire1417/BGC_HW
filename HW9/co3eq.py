import numpy as np
def co3eq(temp, s, z, alk, dic):
    """
    Calculates pCO2, pH, H2CO3*, HCO3, and CO3 concentrations.

    NOTE: Currently you have to do this for a single point at a time. Due to
    dependence on `np.roots`, it isn't extremely easy to vectorize, although
    I imagine it is possible if one spent a little time on it.

    Contact: Riley.Brady@colorado.edu


    Parameters
    ----------
    temp : float
        Temperature at location (degC)
    s : float
        Salinity at location (ppt)
    z : float
        Depth at location (m)
    alk : float
        Alkalinity at location (umol/kg)
    dic : float
        DIC at location (umol/kg)


    Returns
    -------
    pCO2 (uatm), pH, CO2 (umol/kg), HCO3 (umol/kg), CO3 (umol/kg) as floats.


    Reference
    ---------
    Emerson and Hedges 2008: Chemical Oceanography and the Marine Carbon Cycle.
    (4A 1.2)


    Example
    -------
    pco2, pH, CO2, HCO3, CO3 = co3eq(15, 35, 10, 2300, 2100)
    """
    # Conversions
    t = temp + 273.15
    Pr = z/10
    alk = alk * 1e-6
    dic = dic * 1e-6
    R = 83.131

    # Calculate total borate from chlorinity
    tbor = .000416 * s / 35

    # Calculate Henry's Law coefficient, K0 (Weiss, 1974)
    U1 = -60.2409 + 93.4517 * (100/t) + 23.3585*np.log(t/100)
    U2 = s * (.023517 - .023656 * (t/100) + .0047036 * (t/100)**2)
    KH = np.exp(U1 + U2)

    # Calculate KB from temp and salinity (Dickson, 1990)
    KB = np.exp((-8966.9 - 2890.53 * s**0.5 - 77.942 * s + 1.728 * s**1.5
                 - 0.0996 * s**2)/t + 148.0248 + 137.1942 * s**0.5 + 1.62142 * s
                 - (24.4344 + 25.085 * s**0.5 + 0.2474 * s) * np.log(t)
                 + 0.053105 * s**0.5 * t);

    # Calculate K1 and K2 (Luecker et al., 2000)
    K1 = 10**(-(3633.86/t - 61.2172 + 9.67770 * np.log(t) - 0.011555 * s
                + 0.0001152 * s**2))
    K2 = 10**(-(471.78/t + 25.92990 - 3.16967 * np.log(t) - 0.01781 * s
                + 0.0001122 * s**2))

    # Pressure variation of K1, K2, and KB (Millero, 1995)
    dvB = -29.48 + 0.1622 * temp - .002608 * (temp)**2
    dv1 = -25.50 + 0.1271 * temp
    dv2 = -15.82 - 0.0219 * temp
    dkB = -.00284
    dk1 = -.00308 + 0.0000877 * temp
    dk2 = .00113 - .0001475 * temp
    KB  = (np.exp(-(dvB / (R * t)) * Pr + (0.5 * dkB / (R * t)) * Pr**2)) * KB
    K1  = (np.exp(-(dv1 / (R * t)) * Pr + (0.5 * dk1 / (R * t)) * Pr**2)) * K1
    K2  = (np.exp(-(dv2 / (R * t)) * Pr + (0.5 * dk2 / (R * t)) * Pr**2)) * K2

    # Temperature dependence of KW (DOE, 1994)
    KW1 = 148.96502 - 13847.26 / t - 23.65218 * np.log(t)
    KW2 = (118.67 / t - 5.977 + 1.0495 * np.log(t)) * s**.5 - 0.01615 * s
    KW  = np.exp(KW1 + KW2)

    # solve for H ion (Zeebe and Wolf-Gladrow, 2000)
    a1 = 1
    a2 = (alk + KB + K1)
    a3 = (alk * KB - KB * tbor - KW + alk * K1 + K1 * KB + K1 * K2 - dic * K1)
    a4 = (-KW  * KB + alk * KB * K1 - KB * tbor * K1 - KW * K1 + alk * K1 * K2
          + KB * K1 * K2 - dic * KB * K1 - 2 * dic * K1 * K2)
    a5 = (-KW * KB * K1 + alk * KB * K1 * K2 - KW * K1 * K2 - KB * tbor * K1
          * K2 - 2 * dic * KB * K1 * K2)
    a6 = -KB * KW * K1 * K2
    p = [a1, a2, a3, a4, a5, a6]
    r = np.roots(p)
    h = np.max(np.real(r))

    # Calculate bicarbonate, carbonate, and aqueous CO2 usin DIC, Alk, and H+
    hco3 = dic / (1 + h/K1 + K2/h) * 1e6
    co3 = dic / (1 + h/K2 + h * h / (K1 * K2)) * 1e6
    co2 = dic / (1 + K1/h + K1 * K2 / (h * h)) * 1e6
    pco2 = co2 / KH
    pH = -np.log10(h)

    # Calculate B(OH)4 and OH
    BOH4 = KB * tbor / (h + KB)
    OH = KW / h

    # recalculate DIC and Alk to check calculations
    Ct = (hco3 + co3 + co2) * 1e6
    At = (hco3 + 2*co3 + BOH4 + OH - h) * 1e6

    return pco2, pH, co2, hco3, co3
