
"""
Module that defines the functions and classes for working with a plane.

"""


import nanomaterial as nano
import numpy as np
import pandas as pd
import itertools as it
import os
import time
from scipy.spatial import ConvexHull
import networkx as nx

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
    flat_final, connectivity = _surface_clean(flat_init, box_length)

    return flat_final, connectivity, box_length


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
    connect.add_hydrogen(box, pbc=True)
    connect = connect.reset_nodes()

    flat = connect.get_df()
    dt = time.time() - t0
    print("Done in %.0f s" % dt)
    return flat, connect


class plane(nano.NANO):
    """Class to represent a flat surface of SiO2."""

    def __init__(self, par, file=cell_unit):
        """
        Initialize the class by reading the dimensions of the plane.

        par -- str, XxYxZ

        """
        super().__init__(file)

        # Gen dataframe estructure with coordinates and connectivity
        self.flat_final, self.connectivity, self.box_length = build_flat_surface(par, file)

        self.flat_final['surf'] = self.flat_final['z'].apply(self._class_surfaces)

    @property
    def surface(self):
        """Surface area from ConvexHull"""
        hcoord0 = self.flat_final[(self.flat_final.atsb == 'H') & (self.flat_final.surf == 0.0)]
        hcoord1 = self.flat_final[(self.flat_final.atsb == 'H') & (self.flat_final.surf == 1.0)]

        xyz0 = hcoord0.loc[:, ['x', 'y', 'z']].values.astype(np.float64)
        xyz1 = hcoord1.loc[:, ['x', 'y', 'z']].values.astype(np.float64)

        hull0 = ConvexHull(xyz0)
        hull1 = ConvexHull(xyz1)

        indexs0 = hull0.simplices
        indexs1 = hull1.simplices

        self.faces0 = xyz0[indexs0]
        self.faces1 = xyz0[indexs1]

        # return in nanometers^2
        return (hull0.area + hull1.area) / 100

    def _class_surfaces(self, z):
        m_zL = self.box_length[-1] / 2

        if z > m_zL:
            return 1
        elif z < m_zL:
            return 0

    @property
    def H_surface(self):
        """Number of H per nm2"""
        coord = self.flat_final[self.flat_final.atsb == 'H']

        return len(coord) / self.surface

    def get_types_interactions(self):
        """ Searching atoms, bonds angles types."""

        # Call the list of interactions
        bonds_list, angles_list, pairs_list = self._get_interctions_list()

        print("assigning force field parameters", end=" -- ")
        t0 = time.time()

        self.dfatoms, self.dfbonds, self.dfangles = self._set_atoms_types(
                self.flat_final,
                self.connectivity,
                bonds_list,
                angles_list
            )

        self.pairs_list = pairs_list

        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _get_interctions_list(self):
        """Lists of interactions are generated."""
        print("Get interactions List", end=" -- ")
        t0 = time.time()

        connect = self.connectivity

        all_ps = dict(nx.algorithms.all_pairs_shortest_path_length(connect))
        all_paths = []

        for s in all_ps.keys():
            for e in all_ps[s].keys():
                if all_ps[s][e] == 1:
                    all_paths += list(nx.algorithms.all_simple_paths(connect, s, e, cutoff=1))

                elif all_ps[s][e] == 2:
                    all_paths += list(nx.algorithms.all_simple_paths(connect, s, e, cutoff=2))

                elif all_ps[s][e] == 3:
                    all_paths += list(nx.algorithms.all_simple_paths(connect, s, e, cutoff=3))

        bonds_list = [tuple(p) for p in all_paths if len(set(p)) == 2]
        angles_list = [tuple(p) for p in all_paths if len(set(p)) == 3]
        pairs_list = [(p[0], p[3]) for p in all_paths if len(set(p)) == 4]

        dt = time.time() - t0
        print("Done in %.0f s" % dt)

        return bonds_list, angles_list, pairs_list
