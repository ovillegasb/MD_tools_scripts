"""
Module used to store general classes and methods when dealing with nanoparticles.

"""

import time
import pandas as pd
import numpy as np
from numpy import random
import itertools as it
import networkx as nx
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
    print(f'Saved xyz file: {xyz}')


class connectivity(nx.DiGraph):
    def __init__(self):
        super().__init__()
        self._open_si = []
        self._open_o = []

    def get_connectivity(self, coord):
        """Build connectivity from Graph."""
        # Add nodes using atom and coordinates
        for i in coord.index:
            self.add_node(
                i,
                xyz=coord.loc[i, ['x', 'y', 'z']].values,
                atsb=coord.loc[i, 'atsb']
            )

        # get pairs atoms bonded
        pairs = _neighboring_pairs(coord)
        for i, j in pairs:
            self.add_edge(i, j)
            self.add_edge(j, i)

        # remove atoms not conected
        for i in coord.index:
            # remove any atom not bonded
            if self.nbonds(i) == 0:
                self.remove_node(i)

            elif self.nodes[i]['atsb'] == 'Si' and self.nbonds(i) == 1:
                self._open_o.append(list(self.neighbors(i))[0])
                self.remove_node(i)

            elif self.nodes[i]['atsb'] == 'Si' and 1 < self.nbonds(i) < 4:
                self._open_si.append(i)

            elif self.nodes[i]['atsb'] == 'O' and self.nbonds(i) == 1:
                self._open_o.append(i)

    def nbonds(self, iat):
        """Return number of atoms in connected to iat."""
        return int(self.degree[iat] / 2)

    def add_oxygens(self):
        """Adding news oxygens to silice with 1 < nb < 4."""
        natoms = max(list(self.nodes)) + 1
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
                # adding new molecule
                self._add_new_at(natoms, ai, new_ox, 'O')
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
                new_ox1 = N + vnew / np.linalg.norm(MP)
                new_ox2 = N - vnew / np.linalg.norm(MP)
                # adding new molecule
                self._add_new_at(natoms, ai, new_ox1, 'O')
                self._open_o.append(natoms)
                natoms += 1
                self._add_new_at(natoms, ai, new_ox2, 'O')
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
            osi_u = osi_vec / np.linalg.norm(osi_vec)
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

            self._add_new_at(natoms, ai, h, 'H')
            natoms += 1

    def _add_new_at(self, n, m, vector, symbol):
        """Add news atoms in the structure."""
        self.add_node(
            n,
            xyz=vector,
            atsb=symbol
        )
        # adding connectivity
        self.add_edge(n, m)
        self.add_edge(m, n)

    def save_df(self):
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
    """ Object nanoparticle """

    def __init__(self):
        """The NANO object is initialized."""
        pass

    def _get_bonds_list(self, connect):
        """Returns a list of bonds."""
        bonds_list = set()
        for i, j in connect.edges:
            if (i, j) not in bonds_list and (j, i) not in bonds_list:
                bonds_list.add((i, j))

        return bonds_list

    def _get_angles_list(self, connect):
        """Returns a list oof angles."""
        angles_list = set()
        for at in connect.nodes:
            at_links = list(connect.neighbors(at))
            if len(at_links) == 2:
                angles_list.add((at_links[0], at, at_links[1]))
            else:
                for i, k in it.combinations(at_links, 2):
                    angles_list.add((i, at, k))

        return angles_list

    def _get_pairs_list(self, bonds, angles):
        """Return a list of pair (1-4)."""
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
        """Assing atoms type to atoms."""
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

        if 'no found' in coord['type'].values:
            print(coord[coord.type == 'no found'])
            print(coord)
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
        # sicoord = coord[(coord['atsb'] == 'Si') & (coord['nb'] < 4)]
        # ocoord = coord[(coord['atsb'] == 'O') & (coord['nb'] < 2)]

        # if len(sicoord) != 0 or len(ocoord) != 0:
        #     print('ERROR atoms')
        #     print(f'silice {sicoord}')
        #     print(f'oxygens {ocoord}')
        #     print('gro and itp not saved')

        # else:
        #     self.write_gro(coord, box, res)
        #     self.write_itp(res)
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
