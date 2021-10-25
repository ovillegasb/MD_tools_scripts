"""
Module used to store general classes and methods when dealing with nanoparticles.

"""

import time
import pandas as pd
import numpy as np
from numpy import linalg as LA
from numpy import random
import networkx as nx
import itertools as it
from scipy.spatial.distance import cdist
from silica_forcefield import data_atomstype, data_bondstype, data_anglestype


"""
Cell Parameter

crystalobalite:
    A: 4.97170 angs   alpha: 90.0 degree
    B: 4.97170 angs   beta: 90.0 degree
    C: 6.92230 angs   gamma: 90.0 degree
"""

par = {'A': 4.97170, 'B': 4.97170, 'C': 6.92230}


def load_xyz(file):
    """Read a file xyz."""
    coord = pd.read_csv(
        file,
        sep=r'\s+',
        skiprows=2,
        header=None,
        names=['atsb', 'x', 'y', 'z'],
        dtype={'x': np.float64, 'y': np.float64, 'z': np.float64}
    )

    return coord


def save_xyz(coord, name='nps'):
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


class connectivity(nx.DiGraph):
    """Building a class connectivity from directed graphs."""
    def __init__(self):
        super().__init__()
        self._open_si = []
        self._open_o = []

        # index checker
        self._id_removed = []

    def get_connectivity(self, coord):
        """Build connectivity from coordinates using nodes like atoms."""

        # Add nodes using atoms andsymbols and coordinates
        for i in coord.index:
            self.add_node(
                i,
                xyz=coord.loc[i, ['x', 'y', 'z']].values,
                atsb=coord.loc[i, 'atsb']
            )

        # Add edges like bonds
        # get pairs atoms bonded
        pairs = _neighboring_pairs(coord)
        for i, j in pairs:
            self.add_edge(i, j)
            self.add_edge(j, i)

        # remove atoms not conected
        for i in coord.index:
            # remove any atom not bonded
            # and list atoms si, o in surface
            if self.nbonds(i) == 0:
                self.remove_node(i)
                self._id_removed.append(i)

            elif self.nodes[i]['atsb'] == 'Si' and 1 < self.nbonds(i) < 4:
                self._open_si.append(i)

            elif self.nodes[i]['atsb'] == 'O' and self.nbonds(i) == 1:
                self._open_o.append(i)

    def periodic_boxes(self, box, pbc='xy'):
        """Generate the box periodics from pbc."""
        components = []
        components[:0] = pbc

        coord = self.get_df()

        # Separate coordinates from Si and O
        dfSI = coord[coord['atsb'] == 'Si']
        dfO = coord[coord['atsb'] == 'O']

        AB = box[:-1]
        # print(f'Plane dimensions AxB: {AB}')
        """
        boxs :

               | _x+y|
        | -x_y | _x_y| +x_y |
               | _x-y|

        4 box news
        """
        boxs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for (i, j) in boxs:
            # translating coordinates to the quadrant (i, j)
            dfOn = pd.DataFrame(dfO, copy=True)
            dfOn.loc[:, components] += np.array([i, j]) * AB

            # compute distance matric SI vs O (for differents periodics boxs)
            xyzSI = dfSI.loc[:, ['x', 'y', 'z']].values.astype(np.float)
            xyzO = dfOn.loc[:, ['x', 'y', 'z']].values.astype(np.float)

            m = cdist(xyzSI, xyzO, 'euclidean')

            # search connectivity
            for si in range(len(m)):
                # Index where is O between 0.0 to 1.8 ang
                ndxO = dfOn.iloc[np.where((m[si, :] > 0.0) & (m[si, :] <= 1.8))[0], :].index
                ndxSI = dfSI.iloc[[si], :].index

                if len(ndxO) > 0:
                    self.add_edge(ndxSI[0], ndxO[0])
                    self.add_edge(ndxO[0], ndxSI[0])
                else:
                    continue

        # Update the silicon connected
        for i in dfSI.loc[self._open_si, :].index:
            if self.nodes[i]['atsb'] == 'Si' and self.nbonds(i) == 4:
                if i in self._open_si:
                    self._open_si.remove(i)

    def nbonds(self, inode):
        """Return number of atoms in connected to iat."""
        return int(self.degree[inode] / 2)

    def at_connect(self, inode):
        """Return the atoms symbols conectec to i node"""

        return '-'.join([self.nodes[a]['atsb'] for a in list(self.neighbors(inode))])

    def reset_nodes(self):
        """Reset le count of node from 0 to sizes nodes -1"""
        mapping = {value: count for count, value in enumerate(self.nodes, start=1)}

        return nx.relabel_nodes(self, mapping, copy=True)

    def _pbc(self, box, vec):
        """Transfers the coordinates to the main box."""
        nvec = []
        for q, qL in zip(vec, box):
            if q < 0:
                nvec.append(q + qL)
            elif q > qL:
                nvec.append(q - qL)
            else:
                nvec.append(q)

        return np.array(nvec, dtype=np.float64)

    def add_oxygens(self, box, pbc=None):
        """Adding news oxygens to silice with 1 < nb < 4."""
        natoms = len(self.nodes)
        for ai in self._open_si:
            # Silicon coordinates
            si = np.array(self.nodes[ai]['xyz'], dtype=np.float64)
            if self.nbonds(ai) == 3:
                # One bond is required.
                oxygens = list(self.neighbors(ai))
                ox1 = np.array(self.nodes[oxygens[0]]['xyz'], dtype=np.float64)
                ox2 = np.array(self.nodes[oxygens[1]]['xyz'], dtype=np.float64)
                ox3 = np.array(self.nodes[oxygens[2]]['xyz'], dtype=np.float64)
                new_ox = (si - ox1) + (si - ox2) + (si - ox3) + si
                # Using pbc
                if pbc:
                    new_ox = self._pbc(box, new_ox)
                # adding new atom O
                if len(self._id_removed) > 0:
                    self.add_new_at(self._id_removed[0], ai, new_ox, 'O')
                    self._open_o.append(self._id_removed[0])
                    self._id_removed.pop(0)
                    natoms += 1
                else:
                    self.add_new_at(natoms, ai, new_ox, 'O')
                    self._open_o.append(natoms)
                    natoms += 1

            if self.nbonds(ai) == 2:
                # Two bonds are required.
                oxygens = list(self.neighbors(ai))
                ox1 = np.array(self.nodes[oxygens[0]]['xyz'], dtype=np.float64)
                ox2 = np.array(self.nodes[oxygens[1]]['xyz'], dtype=np.float64)
                # Calculing new coordinates for two O
                M = (ox1 + ox2) / 2
                N = 2 * si - M
                MA = ox1 - M
                MP = si - M
                # MA x MP
                vnew = np.cross(MA, MP)
                new_ox1 = N + vnew / LA.norm(MP)
                new_ox2 = N - vnew / LA.norm(MP)
                # Using pbc
                if pbc:
                    new_ox1 = self._pbc(box, new_ox1)
                    new_ox2 = self._pbc(box, new_ox2)

                # adding new atom O1
                if len(self._id_removed) > 0:
                    self.add_new_at(self._id_removed[0], ai, new_ox1, 'O')
                    self._open_o.append(self._id_removed[0])
                    self._id_removed.pop(0)
                    natoms += 1
                else:
                    self.add_new_at(natoms, ai, new_ox1, 'O')
                    self._open_o.append(natoms)
                    natoms += 1

                # adding new atom O2
                if len(self._id_removed) > 0:
                    self.add_new_at(self._id_removed[0], ai, new_ox2, 'O')
                    self._open_o.append(self._id_removed[0])
                    self._id_removed.pop(0)
                    natoms += 1
                else:
                    self.add_new_at(natoms, ai, new_ox2, 'O')
                    self._open_o.append(natoms)
                    natoms += 1

    def add_hydrogen(self):
        """Adding hydrogen to terminal oxygens."""
        natoms = max(list(self.nodes)) + 1
        dHO = 0.945  # angs
        thHOSi = 115.0  # degree
        for ai in self._open_o:
            # Silicon coordinates
            ox = np.array(self.nodes[ai]['xyz'], dtype=np.float64)
            # Silicon connected
            si = np.array(self.nodes[list(self.neighbors(ai))[0]]['xyz'], dtype=np.float64)
            # compute the coordinate hydrogen randomly
            osi_vec = si - ox
            osi_u = osi_vec / LA.norm(osi_vec)
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
                oh_u = oh_vec / LA.norm(oh_vec)
                h = oh_vec + ox
                th_hosi = np.arccos(np.dot(oh_u, osi_u))
                th_hosi *= 180 / np.pi
                th = np.round(th_hosi, decimals=1)

            self.add_new_at(natoms, ai, h, 'H')
            natoms += 1

    def add_new_at(self, n, m, vector, symbol):
        """Add news atoms in the structure."""
        self.add_node(
            n,
            xyz=vector,
            atsb=symbol
        )
        # adding connectivity
        self.add_edge(n, m)
        self.add_edge(m, n)

    def get_df(self):
        """Return the connectivity as a Pandas DataFrame."""
        indexs = list(self.nodes)
        rows = list()

        for i in self.nodes:
            rows.append({
                'atsb': self.nodes[i]['atsb'],
                'x': self.nodes[i]['xyz'][0],
                'y': self.nodes[i]['xyz'][1],
                'z': self.nodes[i]['xyz'][2]
            })

        df = pd.DataFrame(
            rows, index=indexs
        )
        return df


def _neighboring_pairs(coord):
    """Return neighboring pairs"""
    xyz = coord.loc[:, ['x', 'y', 'z']].values.astype(np.float64)
    # compute distance
    m = cdist(xyz, xyz, 'euclidean')
    m = np.triu(m)
    indexs = np.where((m > 0.) & (m <= 2.0))
    return map(lambda in0, in1: (in0, in1), indexs[0], indexs[1])


class NANO:
    """
    Object nanoparticle

    """

    def __init__(self, file):
        """The NANO object is initialized by loading a reference structure."""
        pass

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

                ox1 = N + vnew / LA.norm(MP)
                ox2 = N - vnew / LA.norm(MP)

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
                sio_u_i = sio_vec_i / LA.norm(sio_vec_i)

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
                    sio_u = sio_vec / LA.norm(sio_vec)

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

                ox1 = N + vnew / LA.norm(MP)
                ox2 = N - vnew / LA.norm(MP)

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
            osi_u = osi_vec / LA.norm(osi_vec)

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
                oh_u = oh_vec / LA.norm(oh_vec)

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
                for j in connect.neighbors(i):
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
        lines += f"Radius final: {self.r_final:.3f} nm"
        lines += f"Diameter final: {self.r_final * 2:.3f} nm"
        lines += f"Surface: {self.surface:.3f} nm2"
        lines += f"H per nm2: {self.H_surface:.3f}"

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
