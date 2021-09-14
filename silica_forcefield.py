"""
Force Field Parameters

"""

# Parameters for atoms types
# type : [charge, sigma(ang), epsilon(kcal/mol), mass]
data_atomstype = {
    "SIbulk": [1.1, 4.15, 0.093, 28.08000],
    "Obulk": [-0.55, 3.47, 0.054, 15.99940],
    "Osurf": [-0.675, 3.47, 0.122, 15.99940],
    "Hsurf": [0.40, 1.085, 0.015, 1.00800]
}

# Parameters ofr bond types (harmonic)
# type : [kb(kcal/mol ang^2), ro,ij(ang)]
# "#define gb_1        0.1000  1.5700e+07"
data_bondstype = {
    ("SIbulk", "Obulk"): [285, 1.68],
    ("SIbulk", "Osurf"): [285, 1.68],
    ("Osurf", "Hsurf"): [495, 0.945]
}

# Parameters ofr angle types (harmonic)
# type : [ka(kcal/mol rad^2), thetao,ijk(degree)]
data_anglestype = {
    ("Obulk", "SIbulk", "Obulk"): [100, 109.5],
    ("Obulk", "SIbulk", "Osurf"): [100, 109.5],
    ("Osurf", "SIbulk", "Osurf"): [100, 109.5],
    ("SIbulk", "Obulk", "SIbulk"): [100, 149.0],
    ("SIbulk", "Osurf", "Hsurf"): [50, 115.0]
}
