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
from silica_forcefield import data_atomstype, data_bondstype, data_anglestype
from scipy.spatial.distance import cdist

# dat location
location = os.path.dirname(os.path.realpath(__file__))
cell_unit = os.path.join(location, "cell_unit_crystalobalite.xyz")


class NANO:
    """
    Object nanoparticle

    """

    """
    Cell Parameter

    crystalobalite:
        A: 4.97170 angs   alpha: 90.0 degree
        B: 4.97170 angs   beta: 90.0 degree
        C: 6.92230 angs   gamma: 90.0 degree
    """

    par = {'A': 4.97170, 'B': 4.97170, 'C': 6.92230}

    def __init__(self, file=cell_unit):

        # Initialize instance

        # El objeto NANO se inicializa cargado una estructura de referencia
        # por defecto es la celda unitaria de la cristalobalita

        self.cell = self.load_cell(file)

    def load_cell(self, file=cell_unit):
        """
        Read a file xyz.

        """
        # Reading cell
        coord = pd.read_csv(
            file,
            sep=r'\s+',
            skiprows=2,
            header=None,
            names=['atsb', 'x', 'y', 'z'],
            dtype={'x': np.float64, 'y': np.float64, 'z': np.float64}
        )

        return coord

    def _neighboring_pairs(self, coord):
        """Return neighboring pairs"""
        xyz = coord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)
        # compute distance
        m = cdist(xyz, xyz, 'euclidean')
        m = np.triu(m)

        indexs = np.where((m > 0.) & (m <= 2.0))

        return map(lambda in0, in1: (in0, in1), indexs[0], indexs[1])

    def get_connectivity(self, coord):
        """ Return connectivity of the system no periodic"""
        connect = dict()
        pairs = self._neighboring_pairs(coord)

        for i, j in pairs:
            if i in connect:
                connect[i].add(j)
            else:
                connect[i] = set()
                connect[i].add(j)

            if j in connect:
                connect[j].add(i)
            else:
                connect[j] = set()
                connect[j].add(i)

        return connect

    def _atoms_not_connected(self, coord, connect):
        # search atoms not connected
        not_connected = []
        for i in coord.index:
            if i not in connect:
                not_connected.append(i)

        if len(not_connected) > 0:
            new = coord.drop(not_connected)
            new = new.reset_index()
            return new
        else:
            return coord

    def _update_nbonds(self, coord, connect):
        """Updates number of bonds for dfatoms."""
        coord['nb'] = 0

        for i in coord.index:
            coord.loc[i, 'nb'] = len(connect[i])

        return coord

    def save_xyz(self, coord, name='nps'):
        """Saves an xyz file of coordinates."""

        nat = len(coord)
        xyz = "%s.xyz" % name

        lines = ''
        lines += '%d\n' % nat
        lines += '%s\n' % name
        for i in coord.index:
            line = (coord.atsb[i], coord.x[i], coord.y[i], coord.z[i])
            lines += '%3s%8.3f%8.3f%8.3f\n' % line

        # writing all
        with open(xyz, "w") as f:
            f.write(lines)

        print(f'Name of xyz file: {xyz}')

    def _add_oxygens(self, coord, connect):
        """Adding news oxygens to silice with nb < 4"""
        si_coord = coord[(coord['atsb'] == 'Si') & (coord['nb'] < 4)]
        si_coord = si_coord.sort_values('nb', ascending=False)

        ncoord = []

        natoms = coord.index[-1] + 1

        for iSi in si_coord.index:
            # Silicon coordinates
            si = coord.loc[iSi, ['x', 'y', 'z']].values.astype(np.float64)

            if si_coord.loc[iSi, 'nb'] == 3:
                # One bond is required.
                oxs = coord.loc[
                    connect[iSi], ['x', 'y', 'z']].values.astype(np.float64)

                # print(np.sum(si - oxs, axis=0), end='\n\n')

                n_ox = np.sum(si - oxs, axis=0) + si
                # print(n_ox, end='\n\n')

                # add row to system coordinate
                newO = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [n_ox[0]],
                    'y': [n_ox[1]],
                    'z': [n_ox[2]],
                    'nb': [1]}, index=[natoms])

                ncoord.append(newO)

                connect[iSi].add(natoms)
                connect[natoms] = set()

                connect[natoms].add(iSi)
                coord.loc[iSi, 'nb'] += 1
                natoms += 1

            elif si_coord.loc[iSi, 'nb'] == 2:
                # Two bonds are required.
                oxs = coord.loc[
                    connect[iSi], ['x', 'y', 'z']].values.astype(np.float64)

                M = np.sum(oxs, axis=0) / 2
                N = 2 * si - M

                MA = oxs[0] - M
                MP = si - M

                # MA x MP
                vnew = np.cross(MA, MP)

                ox1 = N + vnew / np.linalg.norm(MP)
                ox2 = N - vnew / np.linalg.norm(MP)

                newO1 = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [ox1[0]],
                    'y': [ox1[1]],
                    'z': [ox1[2]],
                    'nb': [1]}, index=[natoms])

                newO2 = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [ox2[0]],
                    'y': [ox2[1]],
                    'z': [ox2[2]],
                    'nb': [1]}, index=[natoms + 1])

                ncoord.append(newO1)
                ncoord.append(newO2)

                connect[iSi].add(natoms)
                connect[iSi].add(natoms + 1)
                connect[natoms] = set()
                connect[natoms + 1] = set()

                connect[natoms].add(iSi)
                connect[natoms + 1].add(iSi)
                coord.loc[iSi, 'nb'] += 2
                natoms += 2

            elif si_coord.loc[iSi, 'nb'] == 1:
                # Three bonds are required
                # some parameters

                dSiO = 1.68  # angs
                thOSiO = 109.5  # degree

                oxs = coord.loc[
                    connect[iSi], ['x', 'y', 'z']].values.astype(np.float64)[0]

                sio_vec_i = oxs - si
                sio_u_i = sio_vec_i / np.linalg.norm(sio_vec_i)

                # The first oxygen is inserted randomly.
                th = 0.0
                while not np.isclose(thOSiO, th, atol=1):
                    phi = random.uniform(0, 2 * np.pi, 1)[0]
                    theta = 0.0

                    # Find the sign of the z-axis
                    if sio_vec_i[2] > 0:
                        theta += random.uniform(0, np.pi / 2, 1)[0]
                    else:
                        theta += random.uniform(np.pi / 2, np.pi, 1)[0]

                    dx = dSiO * np.cos(phi) * np.sin(theta)
                    dy = dSiO * np.sin(phi) * np.sin(theta)
                    dz = dSiO * np.cos(theta)

                    sio_vec = np.array([dx, dy, dz])
                    sio_u = sio_vec / np.linalg.norm(sio_vec)

                    ox = sio_vec + si

                    th_osio = np.arccos(np.dot(sio_u, sio_u_i))
                    th_osio *= 180 / np.pi
                    th = np.round(th_osio, decimals=1)

                # first oxygen
                ox0 = ox

                oxs = np.array([oxs, ox0])
                # seconds
                M = np.sum(oxs, axis=0) / 2
                N = 2 * si - M

                MA = oxs[0] - M
                MP = si - M

                # MA x MP
                vnew = np.cross(MA, MP)

                ox1 = N + vnew / np.linalg.norm(MP)
                ox2 = N - vnew / np.linalg.norm(MP)

                newO0 = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [ox0[0]],
                    'y': [ox0[1]],
                    'z': [ox0[2]],
                    'nb': [1]}, index=[natoms])

                newO1 = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [ox1[0]],
                    'y': [ox1[1]],
                    'z': [ox1[2]],
                    'nb': [1]}, index=[natoms + 1])

                newO2 = pd.DataFrame({
                    'atsb': ['O'],
                    'x': [ox2[0]],
                    'y': [ox2[1]],
                    'z': [ox2[2]],
                    'nb': [1]}, index=[natoms + 2])

                ncoord.append(newO0)
                ncoord.append(newO1)
                ncoord.append(newO2)

                connect[iSi].add(natoms)
                connect[iSi].add(natoms + 1)
                connect[iSi].add(natoms + 2)
                connect[natoms] = set()
                connect[natoms + 1] = set()
                connect[natoms + 2] = set()

                connect[natoms].add(iSi)
                connect[natoms + 1].add(iSi)
                connect[natoms + 2].add(iSi)
                coord.loc[iSi, 'nb'] += 3
                natoms += 3

        ncoord = pd.concat(ncoord, ignore_index=False)
        # print(ncoord)

        # coord = coord.append(ncoord, ignore_index=False)
        sphere = pd.concat([coord, ncoord], ignore_index=False)

        return sphere, connect

    def _add_hydrogen(self, coord, connect):

        o_coord = coord[(coord['atsb'] == 'O') & (coord['nb'] < 2)]

        ncoord = []

        natoms = coord.index[-1] + 1

        for iO in o_coord.index:
            dHO = 0.945  # angs
            thHOSi = 115.0  # degree

            # Oxygen coordinate
            ox = coord.loc[
                iO, ['x', 'y', 'z']
            ].values.astype(np.float64)

            # Silicon connected
            si = coord.loc[
                connect[iO], ['x', 'y', 'z']
            ].values.astype(np.float64)[0]

            # print(ox)

            # print(si)

            osi_vec = si - ox
            osi_u = osi_vec / np.linalg.norm(osi_vec)

            # print(osi_vec)
            # print(osi_u)

            # ramdom insertion of H
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

            # add row to system coordinate
            newH = pd.DataFrame({
                'atsb': ['H'],
                'x': [h[0]],
                'y': [h[1]],
                'z': [h[2]],
                'nb': [1]}, index=[natoms])

            ncoord.append(newH)

            connect[iO].add(natoms)
            connect[natoms] = set()

            connect[natoms].add(iO)
            coord.loc[iO, 'nb'] += 1
            natoms += 1

        ncoord = pd.concat(ncoord, ignore_index=False)
        # print(ncoord)

        # coord = coord.append(ncoord, ignore_index=False)
        sphere = pd.concat([coord, ncoord], ignore_index=False)

        return sphere, connect

    def _get_bonds_list(self, connect):
        """
        Returns a list of bonds.

        """
        bonds_list = set()

        for at in connect:
            for i, j in it.product([at], connect[at]):
                if (i, j) not in bonds_list and (j, i) not in bonds_list:
                    bonds_list.add((i, j))

        return bonds_list

    def get_angles_list(self, connect, bonds):
        """
        Returns a list oof angles.

        """
        angles_list = set()

        at_bonds = dict()

        for at in connect:
            at_bonds[at] = set()
            for bds in bonds:
                at_bonds[at].add(bds)

        for at in at_bonds:
            ats = []
            if len(at_bonds[at]) > 1:
                for ai, aj in at_bonds[at]:
                    if at == ai:
                        ats.append(aj)
                    elif at == aj:
                        ats.append(ai)

                for ai, aj in it.combinations(ats, 2):
                    if (ai, at, aj) not in angles_list and (aj, at, ai) not in angles_list:
                        angles_list.add((ai, at, aj))

        return angles_list

    def get_pairs_list(self, bonds, angles):
        """
        Return a list of pair (1-4)

        """
        pairs_list = set()

        for ia, ja, ka in angles:
            for ib, jb in bonds:
                if ia == ib and ja != jb:
                    if (jb, ka) in pairs_list or (ka, jb) in pairs_list:
                        pass
                    else:
                        pairs_list.add((jb, ka))
                        break

                elif ia == jb and ja != ib:
                    if (ib, ka) in pairs_list or (ka, ib) in pairs_list:
                        pass
                    else:
                        pairs_list.add((ib, ka))
                        break

                elif ka == ib and ja != jb:
                    if (jb, ia) in pairs_list or (ia, jb) in pairs_list:
                        pass
                    else:
                        pairs_list.add((jb, ia))
                        break

                elif ka == jb and ja != ib:
                    if (ib, ia) in pairs_list or (ia, ib) in pairs_list:
                        pass
                    else:
                        pairs_list.add((ib, ia))
                        break

        return pairs_list

    def _set_atoms_types(self, coord, connect, bonds, angles):
        """
        Assing atoms type to atoms

        """
        coord['type'] = 'no found'

        for i in coord.index:
            # For H
            if coord.loc[i, 'atsb'] == 'H':
                coord.loc[i, 'type'] = 'Hsurf'

            # For Si
            if coord.loc[i, 'atsb'] == 'Si':
                coord.loc[i, 'type'] = 'SIbulk'

            # For O bulk and surf
            if coord.loc[i, 'atsb'] == 'O':
                n_Si = 0
                n_H = 0
                for j in connect[i]:
                    if coord.loc[j, 'atsb'] == 'Si':
                        n_Si += 1
                    if coord.loc[j, 'atsb'] == 'H':
                        n_H += 1

                if n_Si == 2:
                    coord.loc[i, 'type'] = 'Obulk'
                elif n_Si == 1 and n_H == 1:
                    coord.loc[i, 'type'] = 'Osurf'

        if 'no found' in coord['type']:
            print(coord[coord.type == 'no found'])
            exit()

        # bonds types
        bonds_list = list(bonds)
        bonds_types = list()
        for ai, aj in bonds_list:
            bonds_types.append((coord.loc[ai, "type"], coord.loc[aj, "type"]))

        dfbonds = pd.DataFrame({
            "list": bonds_list, "type": bonds_types})

        # angles types
        angles_list = list(angles)
        angles_types = list()
        for ai, aj, ak in angles_list:
            angles_types.append((
                coord.loc[ai, "type"],
                coord.loc[aj, "type"],
                coord.loc[ak, "type"]))

        dfangles = pd.DataFrame({
            "list": angles_list, "type": angles_types})

        return coord, dfbonds, dfangles

    def save_forcefield(self, coord, box, res='NPS'):
        # Testing if Si and H its ok
        sicoord = coord[(coord['atsb'] == 'Si') & (coord['nb'] < 4)]
        ocoord = coord[(coord['atsb'] == 'O') & (coord['nb'] < 2)]

        if len(sicoord) != 0 or len(ocoord) != 0:
            print('ERROR atoms')
            print(f'silice {sicoord}')
            print(f'oxygens {ocoord}')
            print('gro and itp not saved')

        else:
            self.write_gro(coord, box, res)
            self.write_itp(res)

    def write_gro(self, coord, box, res='NPS'):
        """
        Write coordinates to file .gro.

        """
        nat = len(coord)

        GRO = open('%s.gro' % res.lower(), 'w', encoding='utf-8')
        GRO.write('GRO FILE nanoparticle\n')
        GRO.write('%5d\n' % nat)
        for i in coord.index:
            GRO.write('{:>8}{:>7}{:5d}{:8.3f}{:8.3f}{:8.3f}\n'.format(
                '1' + res,
                coord.loc[i, 'atsb'].upper(),
                i + 1,
                coord.loc[i, 'x'] * 0.1,
                coord.loc[i, 'y'] * 0.1,
                coord.loc[i, 'z'] * 0.1)
            )
        GRO.write('   {:.5f}   {:.5f}   {:.5f}\n'.format(
            box[0] * 0.1,
            box[1] * 0.1,
            box[2] * 0.1))

        GRO.close()
        print("file %s.gro writed" % res.lower())

    def write_itp(self, res='NPS'):
        """
        Writing .itp file

        """

        coord = self.dfatoms
        dfbonds = self.dfbonds
        dfangles = self.dfangles

        now = time.ctime()
        nSi = len(coord[coord['atsb'] == 'Si'])
        nO = len(coord[coord['atsb'] == 'O'])
        nH = len(coord[coord['atsb'] == 'H'])
        formule = 'SI{}O{}H{}'.format(nSi, nO, nH)

        # Head
        lines = ""
        lines += "; Topology file created on {}\n".format(now)
        lines += "; Created by Orlando Villegas\n"
        lines += "; mail: orlando.villegas@univ-pau.fr\n"
        lines += ";" + "-" * 60 + "\n"
        lines += "; RES: {}\n".format(res)
        lines += "; Formule: {}\n".format(formule)
        lines += "; Total Charge %.1f\n" % sum([data_atomstype[coord.loc[ch, 'type']][0] for ch in coord.index])

        # atoms types
        lines += "\n[ atomtypes ]\n"
        lines += "; name   at.num      mass      charge   ptype         c6        c12\n"
        lines += "SIbulk       14     28.08000       0.000       A     0.0079544       4.063484e-05\n"
        lines += "Obulk         8     15.99940       0.000       A     0.0015783       2.755408e-06\n"
        lines += "Osurf         8     15.99940       0.000       A     0.0035659       6.225181e-06\n"
        lines += "Hsurf         1      1.00800       0.000       A     4.0973980e-07       6.684771e-13\n"

        # pairs types
        lines += "\n[ pairtypes ]\n"
        lines += "; i    j    func         cs6          cs12 ; THESE ARE 1-4 INTERACTIONS\n"
        lines += "Obulk    SIbulk    1    0.0035432       1.058137e-05\n"
        lines += "Hsurf    Osurf     1    3.8224222e-05       2.039948e-09\n"
        lines += "Hsurf    Obulk     1    2.5430146e-05       1.357176e-09\n"
        lines += "Osurf    SIbulk    1    0.0053258       1.590469e-05\n"

        # molecule type
        lines += '\n[ moleculetype ]\n'
        lines += '; name  nrexcl\n'
        lines += '{}     3\n'.format(res)

        # atoms definition
        lines += '\n[ atoms ]\n'
        lines += ';   nr     type  resnr  residu    atom    cgnr  charge     mass\n'

        for n in coord.index:
            lines += '{:8d}{:>9}{:5d}{:>8}{:>8}{:5d}{:8.3f}{:10.4f}\n'.format(
                n + 1,
                coord.loc[n, 'type'],
                1,  # n,
                res,
                coord.loc[n, 'atsb'].upper(),
                n + 1,
                data_atomstype[coord.loc[n, 'type']][0],
                data_atomstype[coord.loc[n, 'type']][3])

        # bonds parameters
        lines += '\n[ bonds ]\n'
        lines += ';  ai    aj    ak funct\n'

        for i in dfbonds.index:
            # funct:
            # 1: Harmonic potential
            # 2: GROMOS bonds
            iat = dfbonds.loc[i, 'list'][0] + 1
            jat = dfbonds.loc[i, 'list'][1] + 1
            typ = dfbonds.loc[i, 'type']
            try:
                typ = data_bondstype[typ]
            except KeyError:
                typ = data_bondstype[typ[::-1]]
            lines += '{:8d}{:8d}{:5d}{:10.4f}{:14.4f}\n'.format(
                iat, jat,
                1,
                typ[1] / 10,
                typ[0] * 100 * 4.1858)

        # angles parameters
        lines += '\n[ angles ]\n'
        lines += ';  ai    aj    ak funct\n'

        for i in dfangles.index:
            # funct:
            # 1: Harmonic potential
            # 2: GROMOS bonds
            iat = dfangles.loc[i, 'list'][0] + 1
            jat = dfangles.loc[i, 'list'][1] + 1
            kat = dfangles.loc[i, 'list'][2] + 1
            typ = dfangles.loc[i, 'type']
            try:
                typ = data_anglestype[typ]
            except KeyError:
                typ = data_anglestype[typ[::-1]]
            lines += '{:8d}{:8d}{:8d}{:5d}{:10.2f}{:10.2f}\n'.format(
                iat, jat, kat,
                1,
                typ[1],
                typ[0] * 4.1858)

        # pairs parameters
        lines += '\n[ pairs ]\n'
        lines += ';  ai    aj    funct\n'
        for iat, jat in self.pairs_list:
            lines += '{:8d}{:8d}{:5d}\n'.format(
                iat + 1,
                jat + 1,
                1)

        # writing all
        with open("%s.itp" % res.lower(), "w") as f:
            f.write(lines)

        print("file %s.itp writed" % res.lower())


class spherical(NANO):
    """
    Class to represent a specifically spherical nanoparticle

    """

    def __init__(self, diameter, file=cell_unit):
        """
        Initialize building a box cubic from a unit cell, then the cell will be
        cut to a sphere.

        diameter -- diameter for nanoparticle in nanometers.

        """

        super().__init__(file)

        # save and convert diameter input to angstroms
        self.diameter = diameter * 10.0

    def build_sphere_nps(self):
        # Steps:

        # 0 -- Init read unit cell
        # 1 -- Expand unit cell to cubic box in (diameter + 1 nm)^3
        self._expand_cell()
        # print(self.box_init)

        # 2 -- cut sphere in a sphere from the center of box.
        self._cut_sphere()
        # print(self.sphere_init)

        # 3 -- Complete the surface on sphere initial.
        self._surface_clean()

        # 4 -- Adding hydrogen and oxygen atoms.
        self._surface_fill()

        # 4.1 -- Check that the particle contains a surface type Q3, 4.7 H per nm.
        self.save_xyz(self.sphere_final, name="test_initial")
        exit()
        if self.H_surface > 5.0:
            self._reach_surface_Q3()

        self.save_xyz(self.sphere_final, name="test_final")
        exit()


        # 5 -- Lists of interactions are generated
        self._interactions_lists()

        # 6 - Assing force field parameters
        self._get_types_interactions()

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
        """area from surface."""
        return 4 * np.pi * self.r_final**2

    @property
    def H_surface(self):
        """Number of H per nm2"""
        coord = self.sphere_final[self.sphere_final.atsb == 'H']

        return len(coord) / self.surface

    def _expand_cell(self):
        """Expand the cell coordinates to cubic box with dimension
        (diameter + 2.5 angs)^3."""

        d = self.diameter + 2.5
        cell = self.cell
        # extract parameter from unit cell
        # n A
        nA = int(round(d / self.par['A']))
        A = self.par['A']
        # n B
        nB = int(round(d / self.par['B']))
        B = self.par['B']

        # n C
        nC = int(round(d / self.par['C']))
        C = self.par['C']

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

        self.box_init = coord.copy()
        self.box_length = box
        # self.save_xyz(self.box_init, 'box_init')

    def _cut_sphere(self):
        """Cut a sphere of defined diameter centered in the center of the case."""
        coord = self.box_init.copy()

        sphere = pd.DataFrame({
            'atsb': [],
            'x': [],
            'y': [],
            'z': []
        })

        # center of box
        center = self.box_length / 2

        # sphere radio
        r = (self.diameter + 1) / 2

        # searching atoms in sphere
        for i in coord.index:
            vec = coord.loc[i, ['x', 'y', 'z']].values.astype(np.float64)
            r_vec = np.linalg.norm(vec - center)

            if r_vec < r:
                sphere = sphere.append(coord.loc[i, :], ignore_index=True)

        self.sphere_init = sphere.copy()
        # self.save_xyz(self.sphere_init, name="sphere_init")

    def _surface_clean(self):
        """The surface of the nanoparticle is completed with O, H."""

        print("Searching connectivity", end=" - ")
        t0 = time.time()

        # search connectivity
        connect = self.get_connectivity(self.sphere_init)

        # removing no connected atoms
        sphere = self._atoms_not_connected(self.sphere_init, connect)
        # search connectivity again
        connect = self.get_connectivity(sphere)

        # update number of bonds for atom
        sphere = self._update_nbonds(sphere, connect)
        # self.save_xyz(sphere, 'sphere')

        self.sphere_clean = sphere.copy()
        self.connectivity = connect
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _surface_fill(self):
        """It adds hydrogen and oxygen to the surface."""
        print("Adding hydrogen and oxygen atoms", end=" - ")
        t0 = time.time()
        sphere, connect = self._add_oxygens(self.sphere_clean, self.connectivity)
        sphere, connect = self._add_hydrogen(sphere, connect)
        # self.save_xyz(sphere, 'sphere_H')

        # search connectivity
        self.sphere_final = sphere.copy()
        self.connectivity.update(connect)

        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _interactions_lists(self):
        """Lists of interactions are generated."""

        connect = self.connectivity

        print("Bonds list", end=" - ")
        t0 = time.time()
        self.bonds_list = self._get_bonds_list(connect)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

        print("Angles list", end=" - ")
        t0 = time.time()
        # angles and pairs 1-4
        self.angles_list = self.get_angles_list(
            self.connectivity, self.bonds_list)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

        print("Pairs list", end=" - ")
        t0 = time.time()
        self.pairs_list = self.get_pairs_list(
            self.bonds_list, self.angles_list)
        dt = time.time() - t0
        print("Done in %.0f s" % dt)

    def _get_types_interactions(self):
        """ Searching atoms, bonds angles types."""

        print("assigning force field parameters", end=" - ")
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

            if self.r_final * 2 < self.diameter / 10:
                print(" La estructura alcanzo el limite para mantener el tamano deseado")
                break

            if len(surface_si[3]) == 0 and len(surface_si[2]) == 0:
                break


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

    # initialize nanoparticle with diameter's
    nps = spherical(diameter)

    # build sphere
    nps.build_sphere_nps()

    # save xyz
    # nps.save_xyz(nps.sphere_init, name="sphere_init")

    # saving files
    nps.save_forcefield(nps.dfatoms, nps.box_length)

    print(f"Radius final: {nps.r_final:.3f} nm")
    print(f"Diameter final: {nps.r_final * 2:.3f} nm")
    print(f"Surface: {nps.surface:.3f} nm2")
    print(f"H per nm2: {nps.H_surface:.3f}")

    dt = time.time() - t0
    print("Build done in %.0f s" % dt)


if __name__ == '__main__':
    main()
