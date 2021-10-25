
"""
Module that defines the functions and classes for working with a plane.

"""


import nanomaterial as nano
import numpy as np
import pandas as pd
import itertools as it
import os
import time

# dat location
location = os.path.dirname(os.path.realpath(__file__))
cell_unit = os.path.join(location, "cell_unit_crystalobalite.xyz")


def build_flat_surface(dimensions, file=cell_unit):
    """
    Here is the core of the construction of the nanoparticle.

    dimensions -- X, Y, Z dimension's

    """

    # Steps:

    # 0 -- Init read unit cell
    cell = nano.load_xyz(file)

    # 1 -- Expand unit cell to cubic box in (diameter + 1 nm)^3
    flat_init, box_length = _expand_cell(dimensions, cell)
    # nano.save_xyz(box_init, name='flat_init')

    # 2 -- Search connectivity
    #      Complete the surface on sphere initial.
    #      Add hydrogen and oxygen atoms.
    # flat_final, connectivity = _surface_clean(flat_init)
    _surface_clean(flat_init, box_length)


def _expand_cell(dimensions, cell):
    """Generates the plane by expanding the unit cell to the desired size."""
    print('Expand box', end=' -- ')
    t0 = time.time()

    # Returns number of cells used in each dimension.
    dimensions = np.array(dimensions.split('x'), dtype='float64') * 10.0 + 2.5
    # n A
    A = nano.par['A']
    nA = int(round(dimensions[0] / A))

    # n B
    B = nano.par['B']
    nB = int(round(dimensions[1] / B))

    # n C
    C = nano.par['C']
    nC = int(round(dimensions[2] / C))

    # Box length (3x1)
    box = np.array([nA * A, nB * B, nC * C])

    # DataFrame (4xn) [ atom symbol, x, y, z ]
    coord = pd.DataFrame({
        'atsb': [],
        'x': [],
        'y': [],
        'z': []
    })

    # Expand box
    for (a, b, c) in it.product(range(nA), range(nB), range(nC)):
        # copy from cell
        test_coord = pd.DataFrame(cell, copy=True)
        # modify coordinates
        traslation = np.array([a, b, c]) * np.array([A, B, C])
        test_coord.loc[:, ['x', 'y', 'z']] += traslation
        # add to the system
        coord = coord.append(test_coord, ignore_index=True)

    dt = time.time() - t0
    print("Done in %.0f s" % dt)

    return coord, box


def _surface_clean(coord, box):
    """The surface of the nanoparticle is completed with O, H.
    Search connectivity."""

    print("Clean surface and search connectivity", end=" -- ")
    t0 = time.time()

    # search connectivity
    # remove not connected atoms
    connect = nano.connectivity()
    connect.get_connectivity(coord)
    # Uses periodic boundary conditions to search for remaining connectivity
    connect.periodic_boxes(box, pbc='xy')
    connect.add_oxygens(box, pbc=True)
    # connect.add_hydrogen()

    ncoord = connect.get_df()
    nano.save_xyz(ncoord, name='flat_oxygen')


class plane:
    """
    Class to represent a flat surface of SiO2.
    """
    def __init__(self, par, file=cell_unit):
        """
        Initialize the class by reading the dimensions of the plane.

        par -- str, XxYxZ

        """
        build_flat_surface(par, file)
