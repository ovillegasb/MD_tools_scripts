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
from numpy import random
import itertools as it
from scipy.spatial.distance import cdist
import nanomaterial as nano

# dat location
location = os.path.dirname(os.path.realpath(__file__))
cell_unit = os.path.join(location, "cell_unit_crystalobalite.xyz")


def build_sphere_nps(diameter, file=cell_unit):
    """Here is the core of the construction of the nanoparticle."""

    # Steps:

    # changing diameter to angstroms
    diameter = diameter * 10.0

    # 0 -- Init read unit cell
    cell = nano.load_xyz(file)

    # 1 -- Expand unit cell to cubic box in (diameter + 1 nm)^3
    box_init, box_length = _expand_cell(diameter, cell)
    # nano.save_xyz(box_init, name="box_init")

    # 2 -- cut sphere in a sphere from the center of box.
    sphere_init = _cut_sphere(diameter, box_init, box_length)
    # nano.save_xyz(sphere_init, name="sphere_init")

    # 3 -- Complete the surface on sphere initial.
    sphere_clean, connect = _surface_clean(sphere_init)
    # nano.save_xyz(sphere_clean, name="sphere_clean")

    # 4 -- Adding hydrogen and oxygen atoms.
    print("Adding hydrogen and oxygen atoms", end=" -- ")
    t0 = time.time()
    connect.add_oxygens()
    connect.add_hydrogen()
    sphere_final = connect.save_df()
    # nano.save_xyz(sphere_final, name="sphere_final")
    dt = time.time() - t0
    print("Done in %.0f s" % dt)

    return sphere_final, connect


def _expand_cell(diameter, cell):
    """Expand the cell coordinates to cubic box with dimension
    (diameter + 2.5 angs)^3."""
    print('Expand box', end=' -- ')
    t0 = time.time()
    # exapnd to 2.5 angstroms
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

    box = np.array([nA * A, nB * B, nC * C])

    coord = pd.DataFrame({
        'atsb': [],
        'x': [],
        'y': [],
        'z': []
    })

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
        r_vec = np.linalg.norm(vec - center)
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
    connect = nano.connectivity()
    connect.get_connectivity(coord)
    # Updates number of bonds for dfatoms.
    new_coord = coord.loc[list(connect.nodes()), :]
    new_coord['index0'] = new_coord.index
    new_coord['nb'] = new_coord.apply(lambda x: connect.nbonds(x['index0']), axis=1)
    dt = time.time() - t0
    print("Done in %.0f s" % dt)
    return new_coord, connect


class spherical(nano.NANO):
    """Class to represent a specifically spherical nanoparticle."""

    def __init__(self, diameter, file=cell_unit):
        """
        Initialize building a box cubic from a unit cell, then the cell will be
        cut to a sphere. For default is the unit cell of the crystallobalyte.

        diameter -- diameter for nanoparticle in nanometers.

        """
        super().__init__()

        # save and convert diameter input to angstroms
        self.diameter = diameter * 10.0
        # Gen dataframe estructure with coordinates and connectivity
        self.sphere, self.connectivity = build_sphere_nps(diameter, file)

        # Charge interaction lists
        self._interactions_lists()

    @property
    def center_of_mass(self):
        xyz = self.sphere.loc[:, ['x', 'y', 'z']].values
        return center_of_mass(xyz)

    @property
    def r_final(self):
        """Compute radius of nanoparticles."""
        coord = self.sphere[self.sphere.atsb == 'H']
        c = self.center_of_mass
        xyz = coord.loc[:, ['x', 'y', 'z']].values
        m = cdist(xyz, np.array([c]), 'euclidean')

        return np.mean(m) / 10.0

    @property
    def surface(self):
        """area from surface."""
        return 4 * np.pi * self.r_final**2

    @property
    def H_surface(self):
        """Number of H per nm2"""
        coord = self.sphere_final[self.sphere_final.atsb == 'H']
        return len(coord) / self.surface

    def _interactions_lists(self):
        """Lists of interactions are generated."""
        print("Bonds list", end=" -- ")
        connect = self.connectivity
        t0 = time.time()
        self.bonds_list = self._get_bonds_list(connect)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

        print("Angles list", end=" -- ")
        t0 = time.time()
        # angles and pairs 1-4
        self.angles_list = self.get_angles_list(connect)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)
        exit()

        print("Pairs list", end=" -- ")
        t0 = time.time()
        self.pairs_list = self.get_pairs_list(
            self.bonds_list, self.angles_list)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _get_types_interactions(self):
        """ Searching atoms, bonds angles types."""

        print("assigning force field parameters", end=" -- ")
        t0 = time.time()

        self.dfatoms, self.dfbonds, self.dfangles = self._set_atoms_types(
            self.sphere_final, self.connectivity, self.bonds_list, self.angles_list)

        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _reach_surface_Q3(self):
        """Check that the particle contains a surface type Q3, 4.7 H per nm.
        Surface silicon is searched for, which meet the following conditions to
        be bonded to 3 O groups. The search is randomized over the surface si,
        the list is extracted from the sphere_final instance, which is updated
        at each iteration of which
        """
        print("Removing groups Si(OH)3", end=" -- ")
        t0 = time.time()
        dHO = 0.945
        thHOSi = 115.0
        random.seed(1)
        natoms_init = len(self.sphere_final)

        while self.H_surface > 5.0:
            # coord = self.sphere_final.copy()
            # connectivity = self.connectivity.copy()
            coord_si = self.sphere_final[self.sphere_final.atsb == 'Si']
            surface_si = {3: [], 2: []}
            self.sphere_final['count_OH'] = 0

            # searchinf si near to surface
            for i in coord_si.index:
                count_OH = 0
                si_connect = self.connectivity[i]
                for j in si_connect:
                    o_connect = self.connectivity[j]
                    if 'H' in list(self.sphere_final.loc[o_connect, 'atsb'].values):
                        count_OH += 1
                if count_OH == 3:
                    surface_si[3].append(i)
                    self.sphere_final.loc[i, 'count_OH'] = 3

                elif count_OH == 2:
                    surface_si[2].append(i)
                    self.sphere_final.loc[i, 'count_OH'] = 2

            random.shuffle(surface_si[3])
            random.shuffle(surface_si[2])
            print(surface_si)
            # print(coord.loc[surface_si[3], :])
            # print(coord.loc[surface_si[2], :])

            # Search si index like group OSi(OH)3
            if len(surface_si[3]) > 0:
                for i in surface_si[3]:
                    # Search for silicon atom connectivity
                    si_connect = self.connectivity[i]
                    index_drop = []
                    for j in si_connect:
                        # oxygen atom connectivity
                        o_connect = self.connectivity[j]
                        o_links = '-'.join(self.sphere_final.loc[o_connect, 'atsb'].values)
                        if o_links == 'Si-Si':
                            o_si = [j, i]
                        else:
                            index_drop.append(j)
                    # searching hydrogen
                    h_drop = []
                    for o in index_drop:
                        for h in self.connectivity[o]:
                            if h != i:
                                h_drop.append(h)
                    index_drop += h_drop
                    assert len(index_drop) == 6, "Error, six atoms must be skipped"
                    # The OSi(OH)3 group has been selected.
                    new_coord = self.sphere_final.drop(index=index_drop)
                    new_connectivity = self.connectivity.copy()
                    for at in index_drop:
                        new_connectivity.pop(at)
                    # compute the vector o--si
                    vo = self.sphere_final.loc[o_si[0], ['x', 'y', 'z']].values
                    vsi = self.sphere_final.loc[o_si[1], ['x', 'y', 'z']].values
                    u_osi = (vsi - vo) / np.linalg.norm(vsi - vo)
                    xyz_h = dHO * u_osi + vo
                    # Adding in new coord
                    new_coord.loc[o_si[1], 'atsb'] = 'H'
                    new_coord.loc[o_si[1], 'x'] = xyz_h[0]
                    new_coord.loc[o_si[1], 'y'] = xyz_h[1]
                    new_coord.loc[o_si[1], 'z'] = xyz_h[2]
                    new_coord.loc[o_si[1], 'nb'] = 1
                    # saving in the class
                    self.sphere_final = new_coord.copy()
                    self.connectivity = new_connectivity.copy()
                    print("New H_surface", self.H_surface)
                    print("N atoms total", len(self.sphere_final))
                    print("percentage atoms removed", (natoms_init - len(self.sphere_final)) * 100 / natoms_init, "%")
                    print("Dimeter actual", self.r_final * 2, "nm, Initial", self.diameter / 10, "nm")
                    if self.H_surface < 5.0:
                        break
                    if self.r_final * 2 < self.diameter / 10:
                        print(" Limite alcanzado, aumente el tamano")
                        break
                    # print(i)
                    continue
            """
            if len(surface_si[3]) == 0 and len(surface_si[2]) > 0:
                for i in surface_si[2]:
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
                        # print(o_links)
                    # exit()
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
                    # saving in the class
                    self.sphere_final = new_coord.copy()
                    self.connectivity = new_connectivity.copy()
                    print("New H_surface", self.H_surface)
                    print("N atoms total", len(self.sphere_final))
                    print("percentage atoms removed", (natoms_init - len(self.sphere_final)) * 100 / natoms_init, "%")
                    print("Dimeter actual", self.r_final * 2, "nm, Initial", self.diameter / 10, "nm")
                    if self.H_surface < 5.0:
                        break
                    if self.r_final * 2 < self.diameter / 10:
                        print(" Limite alcanzado, aumente el tamano")
                        break
                    # print(i)
                    break
            """

            if self.r_final * 2 < self.diameter / 10:
                print(" La estructura alcanzo el limite para mantener el tamano deseado")
                break

            #if len(surface_si[3]) == 0 and len(surface_si[2]) == 0:
            if len(surface_si[3]) == 0:
                break
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

    # Build a sphere for diameter desire

    # The spherical class is initialized from the structure of the generated nanoparticle.
    nps = spherical(diameter)

    # Construct a sphere from the indicated diameter.
    # nps.build_sphere_nps()

    # save xyz
    # nps.save_xyz(nps.sphere_init, name="sphere_init")

    # saving files
    # nps.save_forcefield(nps.dfatoms, nps.box_length)

    # print(f"Radius final: {nps.r_final:.3f} nm")
    # print(f"Diameter final: {nps.r_final * 2:.3f} nm")
    # print(f"Surface: {nps.surface:.3f} nm2")
    # print(f"H per nm2: {nps.H_surface:.3f}")

    dt = time.time() - t0
    print("Build done in %.0f s" % dt)


if __name__ == '__main__':
    main()
