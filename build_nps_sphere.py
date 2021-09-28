#!/bin/env python
# -*- conding=utf-8 -*-

"""
This program generates a structure and topology for a spherical SiO2 nanoparticle
to be used in gromacs. The code starts from the unit cell of the crystallobalite.
The force field parameters were taken from: https://pubs.acs.org/doi/10.1021/cm500365c

"""

import os
import argparse
import time
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy import random
import itertools as it
from scipy.spatial.distance import cdist
import nanomaterial as nano
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import mpl_toolkits.mplot3d as a3
import matplotlib as mpl
import networkx as nx


# dat location
location = os.path.dirname(os.path.realpath(__file__))
cell_unit = os.path.join(location, "cell_unit_crystalobalite.xyz")


def build_sphere_nps(diameter, file=cell_unit):
    """
    Here is the core of the construction of the nanoparticle.

    diameter -- diameter for work in angstroms

    """

    # Steps:

    # 0 -- Init read unit cell
    cell = nano.load_xyz(file)

    # 1 -- Expand unit cell to cubic box in (diameter + 1 nm)^3
    box_init, box_length = _expand_cell(diameter, cell)

    # 2 -- cut sphere in a sphere from the center of box.
    sphere_init = _cut_sphere(diameter, box_init, box_length)
    # nano.save_xyz(sphere_init, name="sphere_init")

    # 3 -- Complete the surface on sphere initial.
    #      Add hydrogen and oxygen atoms.
    sphere_final, connectivity = _surface_clean(sphere_init)
    # nano.save_xyz(sphere_final, name="sphere_final")

    return sphere_final, connectivity, box_length


def _expand_cell(diameter, cell):
    """Expand the cell coordinates to cubic box with dimension
    (diameter + 2.5 angs)^3."""

    print('Expand box', end=' -- ')
    t0 = time.time()
    d = diameter + 2.5
    # extract parameter from unit cell
    # n A
    nA = int(round(d / nano.par['A']))
    A = nano.par['A']
    # n B
    nB = int(round(d / nano.par['B']))
    B = nano.par['B']
    # n C
    nC = int(round(d / nano.par['C']))
    C = nano.par['C']
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


def _cut_sphere(diameter, box_init, box_length):
    """Cut a sphere of defined diameter centered in the center of the case."""
    print('Cut sphere', end=' -- ')
    t0 = time.time()
    coord = box_init.copy()
    sphere = pd.DataFrame({
        'atsb': [],
        'x': [],
        'y': [],
        'z': []
    })
    # center of box
    center = box_length / 2
    # sphere radio
    r = (diameter + 1) / 2
    # searching atoms in sphere
    for i in coord.index:
        vec = coord.loc[i, ['x', 'y', 'z']].values.astype(np.float64)
        r_vec = LA.norm(vec - center)
        if r_vec < r:
            sphere = sphere.append(coord.loc[i, :], ignore_index=True)

    dt = time.time() - t0
    print("Done in %.0f s" % dt)

    return sphere


def _surface_clean(coord):
    """The surface of the nanoparticle is completed with O, H."""
    print("Clean surface and search connectivity", end=" -- ")
    t0 = time.time()
    # search connectivity
    # remove not connected atoms
    connect = nano.connectivity()
    connect.get_connectivity(coord)
    connect.add_oxygens()
    connect.add_hydrogen()
    sphere = connect.get_df()
    dt = time.time() - t0
    print("Done in %.0f s" % dt)
    return sphere, connect


class spherical(nano.NANO):
    """
    Class to represent a specifically spherical nanoparticle

    """

    def __init__(self, diameter, file=cell_unit):
        """
        Initialize building a box cubic from a unit cell, then the cell will be
        cut to a sphere. For default is the unit cell of the crystallobalyte.

        diameter -- diameter for nanoparticle in nanometers.

        """

        super().__init__(file)

        # changing diameter to angstroms
        self.diameter = diameter * 10.0

        # Gen dataframe estructure with coordinates and connectivity
        self.sphere_final, self.connectivity, self.box_length = build_sphere_nps(
            self.diameter, file
        )

        # Initialize face from surface
        self.faces = None

    @property
    def center_of_mass(self):
        xyz = self.sphere_final.loc[:, ['x', 'y', 'z']].values
        return center_of_mass(xyz)

    @property
    def r_final(self):
        """Compute radius of nanoparticles."""
        coord = self.sphere_final[self.sphere_final.atsb == 'H']
        c = self.center_of_mass
        xyz = coord.loc[:, ['x', 'y', 'z']].values
        m = cdist(xyz, np.array([c]), 'euclidean')

        return np.mean(m) / 10.0

    @property
    def surface(self):
        """Surface area from ConvexHull"""
        hcoord = self.sphere_final[self.sphere_final.atsb == 'H']
        xyz = hcoord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)

        hull = ConvexHull(xyz)
        indexs = hull.simplices
        self.faces = xyz[indexs]

        # return in nanometers^2
        return hull.area / 100

    @property
    def H_surface(self):
        """Number of H per nm2"""
        coord = self.sphere_final[self.sphere_final.atsb == 'H']

        return len(coord) / self.surface

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

    def get_types_interactions(self):
        """ Searching atoms, bonds angles types."""

        # Call the list of interactions
        bonds_list, angles_list, pairs_list = self._get_interctions_list()

        print("assigning force field parameters", end=" -- ")
        t0 = time.time()

        self.dfatoms, self.dfbonds, self.dfangles = self._set_atoms_types(
                self.sphere_final,
                self.connectivity,
                bonds_list,
                angles_list
            )

        self.pairs_list = pairs_list

        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def fixing_surface(self, Hsurface=5.0):
        """Method used to adjust the surface to a surface hydrogen value."""

        # Check that the particle contains a surface type Q3, 4.7 H per nm.

        if self.H_surface > Hsurface:
            self._reach_surface_Q3(Hsurface)
        nano.save_xyz(self.sphere_final, 'sphere_test')

    def _reach_surface_Q3(self, Hsurface=5.0):
        """Check that the particle contains a surface type Q3, 4.7 H per nm.
        Surface silicon is searched for, which meet the following conditions to
        be bonded to 3 O groups. The search is randomized over the surface si,
        the list is extracted from the sphere_final instance, which is updated
        at each iteration of which
        """
        print("Removing groups Si(OH)3", end=" -- \n")
        t0 = time.time()
        dHO = 0.945
        thHOSi = 115.0
        random.seed(1)
        natoms_init = len(self.sphere_final)
        nit = 0

        while self.H_surface > Hsurface:
            coord = self.sphere_final.copy()
            connect = self.connectivity.copy()
            coord_si = coord[coord.atsb == 'Si']
            surface_si = {3: [], 2: []}
            coord['count_OH'] = 0

            is_added = False

            # searchinf si near to surface
            for i in coord_si.index:
                count_OH = 0
                si_connect = connect.neighbors(i)
                for j in si_connect:
                    o_connect = connect.neighbors(j)
                    if 'H' in list(coord.loc[list(o_connect), 'atsb'].values):
                        count_OH += 1
                if count_OH == 3:
                    surface_si[3].append(i)
                    coord.loc[i, 'count_OH'] = 3

                elif count_OH == 2:
                    surface_si[2].append(i)
                    coord.loc[i, 'count_OH'] = 2

            random.shuffle(surface_si[2])
            # Search si index like group OSi(OH)3
            if len(surface_si[3]) > 0:
                i = surface_si[3][0]
                # Search for silicon atom connectivity
                si_connect = connect.neighbors(i)
                index_drop = []
                for j in si_connect:
                    # oxygen atom connectivity
                    if connect.at_connect(j) == 'Si-Si':
                        o_si = [j, i]
                    else:
                        index_drop.append(j)
                # searching hydrogen
                h_drop = []
                for o in index_drop:
                    for h in connect.neighbors(o):
                        if h != i:
                            h_drop.append(h)
                index_drop += h_drop
                if len(index_drop) != 6:
                    print("Error, six atoms must be skipped")
                    print(f"{len(index_drop)} was selected")
                    print(coord.loc[index_drop, :])
                    continue

                # The OSi(OH)3 group has been selected.
                for at in index_drop:
                    connect.remove_node(at)
                # compute the vector o--si
                vo = coord.loc[o_si[0], ['x', 'y', 'z']].values
                vsi = coord.loc[o_si[1], ['x', 'y', 'z']].values
                u_osi = (vsi - vo) / LA.norm(vsi - vo)
                xyz_h = dHO * u_osi + vo
                # Adding in new coord
                connect.add_new_at(o_si[1], o_si[0], xyz_h, 'H')
                # saving in the class

                # Updating Coordinates and Connectivuty
                self.connectivity = connect.copy()
                self.sphere_final = connect.get_df()

                is_added = True
                if self.H_surface < 5.0:
                    break
                if self.r_final * 2 < self.diameter / 10:
                    print(" Limite alcanzado, aumente el tamano")
                    break
            '''
            elif len(surface_si[3]) == 0 and len(surface_si[2]) > 0:
                n_at_test = len(surface_si[2])
                i = surface_si[2][0]
                # for i in surface_si[2]:
                # Search for silicon atom connectivity
                si_connect = self.connectivity[i]
                index_drop = []
                news_oxygen_free = []
                for j in si_connect:
                    # oxygen atom connectivity
                    o_connect = self.connectivity[j]
                    try:
                        o_links = '-'.join(self.sphere_final.loc[o_connect, 'atsb'].values)
                    except KeyError:
                        print("Error")
                        print(j)
                        print(o_connect)
                        print(self.sphere_final.loc[j, :])
                        exit()
                    if o_links == 'Si-Si':
                        news_oxygen_free.append(j)
                    else:
                        index_drop.append(j)

                # searching hydrogen
                h_drop = []
                for o in index_drop:
                    for h in self.connectivity[o]:
                        if h != i:
                            h_drop.append(h)
                index_drop += h_drop
                # adding i from silice
                news_oxygen_free.append(i)
                # print(index_drop)
                if len(index_drop) != 4:
                    print(f"Error, five atoms must be skipped: {len(index_drop)}")
                    # print(self.sphere_final.loc[index_drop, :])
                    # exit()
                    continue
                # assert len(index_drop) == 4, f"Error, five atoms must be skipped: {len(index_drop)}"
                # The O2Si(OH)2 group has been selected.
                new_coord = self.sphere_final.drop(index=index_drop)
                new_connectivity = self.connectivity.copy()
                for at in index_drop:
                    new_connectivity.pop(at)
                # print(new_connectivity[news_oxygen_free[1]])
                # adding firt oxygen to index i
                # Oxygen coordinate
                ox = self.sphere_final.loc[
                    news_oxygen_free[0],
                    ['x', 'y', 'z']
                ].values.astype(np.float64)
                # Silicon connected (count_OH = 0)
                si = self.sphere_final.loc[
                    new_connectivity[news_oxygen_free[0]]
                ].drop(index=i)
                v_si = si.loc[:, ['x', 'y', 'z']].values.astype(np.float64)[0]
                osi_vec = v_si - ox
                osi_u = osi_vec / np.linalg.norm(osi_vec)
                # ramdom insertion of firt H
                th = 0.0
                while not np.isclose(thHOSi, th, atol=1):
                    phi = random.uniform(0, 2 * np.pi, 1)[0]
                    theta = 0.0
                    # Find the sign of the z-axis
                    if osi_vec[2] > 0:
                        theta += random.uniform(0, np.pi / 2, 1)[0]
                    else:
                        theta += random.uniform(np.pi / 2, np.pi, 1)[0]
                    dx = dHO * np.cos(phi) * np.sin(theta)
                    dy = dHO * np.sin(phi) * np.sin(theta)
                    dz = dHO * np.cos(theta)
                    oh_vec = np.array([dx, dy, dz])
                    oh_u = oh_vec / np.linalg.norm(oh_vec)
                    h = oh_vec + ox
                    th_hosi = np.arccos(np.dot(oh_u, osi_u))
                    th_hosi *= 180 / np.pi
                    th = np.round(th_hosi, decimals=1)
                    # Adding in new coord
                    new_coord.loc[news_oxygen_free[-1], 'atsb'] = 'H'
                    new_coord.loc[news_oxygen_free[-1], 'x'] = h[0]
                    new_coord.loc[news_oxygen_free[-1], 'y'] = h[1]
                    new_coord.loc[news_oxygen_free[-1], 'z'] = h[2]
                    new_coord.loc[news_oxygen_free[-1], 'nb'] = 1
                    new_coord.loc[news_oxygen_free[-1], 'count_OH'] = None
                # adding seconf oxygen to new index
                # Oxygen coordinate
                ox = self.sphere_final.loc[
                    news_oxygen_free[1],
                    ['x', 'y', 'z']
                ].values.astype(np.float64)
                # Silicon connected (count_OH = 0)
                # print(new_connectivity[news_oxygen_free[1]])
                si = self.sphere_final.loc[
                    new_connectivity[news_oxygen_free[1]]
                ].drop(index=i)
                v_si = si.loc[:, ['x', 'y', 'z']].values.astype(np.float64)[0]
                osi_vec = v_si - ox
                osi_u = osi_vec / np.linalg.norm(osi_vec)
                # ramdom insertion of firt H
                th = 0.0
                while not np.isclose(thHOSi, th, atol=1):
                    phi = random.uniform(0, 2 * np.pi, 1)[0]
                    theta = 0.0
                    # Find the sign of the z-axis
                    if osi_vec[2] > 0:
                        theta += random.uniform(0, np.pi / 2, 1)[0]
                    else:
                        theta += random.uniform(np.pi / 2, np.pi, 1)[0]
                    dx = dHO * np.cos(phi) * np.sin(theta)
                    dy = dHO * np.sin(phi) * np.sin(theta)
                    dz = dHO * np.cos(theta)
                    oh_vec = np.array([dx, dy, dz])
                    oh_u = oh_vec / np.linalg.norm(oh_vec)
                    h = oh_vec + ox
                    th_hosi = np.arccos(np.dot(oh_u, osi_u))
                    th_hosi *= 180 / np.pi
                    th = np.round(th_hosi, decimals=1)
                natoms = self.sphere_final.index[-1] + 1
                # add row to system coordinate
                newH = pd.DataFrame({
                    'atsb': ['H'],
                    'x': [h[0]],
                    'y': [h[1]],
                    'z': [h[2]],
                    'nb': [1],
                    'count_OH': [None]}, index=[natoms])
                new_coord = pd.concat([new_coord, newH], ignore_index=False)
                new_connectivity[news_oxygen_free[1]].remove(i)
                new_connectivity[news_oxygen_free[1]].add(natoms)
                new_connectivity[natoms] = set()
                new_connectivity[natoms].add(news_oxygen_free[1])

                """Surface area from ConvexHull"""
                hcoord = new_coord[new_coord.atsb == 'H']
                xyz = hcoord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)
                hull = ConvexHull(xyz)

                h_surface_test = len(hcoord) / hull.area / 100

                if h_surface_test < self.H_surface:
                    # saving in the class
                    print('test', h_surface_test, 'back', self.H_surface)
                    self.sphere_final = new_coord.copy()
                    self.connectivity = new_connectivity.copy()
                    is_added = True

                    if self.H_surface < 5.0:
                        break

                    if self.r_final * 2 < self.diameter / 10:
                        print(" Limite alcanzado, aumente el tamano")
                        break

                else:
                    print(f'{i} dont added')
                    exit()
                    n_at_test -= 1
                    if n_at_test == 0:
                        break
                    else:
                        continue
            '''

            if is_added:
                print("-"*80)
                print(surface_si[3])
                print("New H_surface", self.H_surface)
                # print("N atoms total", len(self.sphere_final))
                # print("percentage atoms removed", (natoms_init - len(self.sphere_final)) * 100 / natoms_init, "%")
                # print("Dimeter actual", self.r_final * 2, "nm, Initial", self.diameter / 10, "nm")
                # print("Iteration number", nit)
                print("-"*80)
                nit += 1

            if nit > 50:
                break

            # if self.r_final * 2 < round(self.diameter, 0) / 10:
            #     print(" La estructura alcanzo el limite para mantener el tamano deseado")
            #     break

            # if len(surface_si[3]) == 0 and len(surface_si[2]) == 0:
            if len(surface_si[3]) == 0:
                # if len(surface_si[3]) == 0 and len(surface_si[2]) == 0:
                break

        self.connectivity = self.connectivity.reset_nodes()
        self.sphere_final = self.connectivity.get_df()
        dt = time.time() - t0
        print("Done in %.0f s" % dt)


def center_of_mass(coords, masses=None):
    r"""Compute the center of mass of the points at coordinates `coords` with
    masses `masses`.
    Args:
        coords (np.ndarray): (N, 3) matrix of the points in :math:`\mathbb{R}^3`
        masses (np.ndarray): vector of length N with the masses
    Returns:
        The center of mass as a vector in :math:`\mathbb{R}^3`
    """
    # check coord array
    try:
        coords = np.array(coords, dtype=np.float64)
        coords = coords.reshape(coords.size // 3, 3)
    except ValueError:
        print("coords = ", coords)
        raise ValueError("Cannot convert coords in a numpy array of floats"
                         " with a shape (N, 3).")

    # check masses
    if masses is None:
        masses = np.ones(coords.shape[0])
    else:
        try:
            masses = np.array(masses, dtype=np.float64)
            masses = masses.reshape(coords.shape[0])
        except ValueError:
            print("masses = ", masses)
            raise ValueError("Cannot convert masses in a numpy array of "
                             "floats with length coords.shape[0].")
    if masses is None:
        masses = np.ones(coords.shape[0])

    return np.sum(coords * masses[:, np.newaxis], axis=0) / masses.sum()


def show_surface_nps(coord, faces, d):
    hcoord = coord[coord.atsb == 'H']
    xyz = hcoord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')

    # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color='b')

    ax.set_xlim3d(0, xyz[:, 0].max())
    ax.set_ylim3d(0, xyz[:, 1].max())
    ax.set_zlim3d(0, xyz[:, 2].max())

    for f in faces:
        face = a3.art3d.Poly3DCollection([f])
        face.set_color(mpl.colors.rgb2hex(random.rand(3)))
        face.set_edgecolor('k')
        face.set_alpha(0.5)
        ax.add_collection3d(face)

    plt.show()


def options():
    """Generate command line interface."""

    parser = argparse.ArgumentParser(
        prog="BUILDER NANO TOPOLOGY",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [-d] diameter",
        epilog="Enjoy the program!",
        description=__doc__
    )

    # Add the arguments
    # File
    parser.add_argument("-d", "--diameter",
                        help="Nanoparticle diameter in nanometers",
                        default=2.0,
                        metavar="angstroms",
                        type=np.float64)

    return vars(parser.parse_args())


def main():
    t0 = time.time()

    # starting
    args = options()

    # in nanometers
    diameter = args['diameter']
    print(f"Diameter initial {diameter} nm")

    # initialize nanoparticle with diameter's
    nps = spherical(diameter)

    # Fitting surface
    nps.fixing_surface(Hsurface=5.0)

    # Gen interaction lists
    nps.get_types_interactions()

    # saving files
    nps.save_forcefield(nps.dfatoms, nps.box_length)

    print(f"Radius final: {nps.r_final:.3f} nm")
    print(f"Diameter final: {nps.r_final * 2:.3f} nm")
    print(f"Surface: {nps.surface:.3f} nm2")
    print(f"H per nm2: {nps.H_surface:.3f}")

    # Show surface H
    show_surface_nps(nps.dfatoms, nps.faces, diameter)

    dt = time.time() - t0
    print("Build done in %.0f s" % dt)


if __name__ == '__main__':
    main()
