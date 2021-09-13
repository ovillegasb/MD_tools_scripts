#!/usr/bin/env python
# -*- conding=utf-8 -*-

import argparse
import numpy as np
from scipy.constants import e

"""
  _____ _____ _____   ____  _               _____
 |  __ \_   _|  __ \ / __ \| |        /\   |  __ \
 | |  | || | | |__) | |  | | |       /  \  | |__) |
 | |  | || | |  ___/| |  | | |      / /\ \ |  _  /
 | |__| || |_| |    | |__| | |____ / ____ \| | \ \
 |_____/_____|_|     \____/|______/_/    \_\_|  \_\


Dipolar is a program to determine the total charge and dipole moment from a GROMACS
.gro and .itp file.

"""


def options():
    """ Generate command line interface."""

    parser = argparse.ArgumentParser(
        prog="DIOLAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s [options] file",
        epilog="Enjoy the program!",
        description=__doc__
    )

    # Add the arguments
    # File
    parser.add_argument("-r", "--resname",
                        help="resname from itp and gro file",
                        default="",
                        metavar="res",
                        type=str)

    return vars(parser.parse_args())


class MDipolar:

    def __init__(self, res):

        gro = '%s.gro' % res
        itp = '%s.itp' % res
        file_gro = self._load_gro(gro)
        file_itp = self._load_itp(itp)

        self.name = ' '.join(next(file_gro))
        self.nat = int(next(file_gro)[0])

        coord = list()
        for xyz in file_gro:
            if f'1{res.upper()}' in xyz:
                coord.append(xyz[3:6])

        self.coord = np.array(coord, dtype=np.float64)

        charges = list()
        for line in file_itp:
            if ' '.join(line) == '[ atoms ]':
                next(file_itp)
                break

        for line in file_itp:
            if res.upper() in line:
                charges.append(line[6])
            else:
                break

        self.charges = np.array(charges, dtype=np.float64)

    def _load_gro(self, gro):
        with open(gro, 'r') as GRO:
            for line in GRO:
                line = line.split()

                yield line

    def _load_itp(self, itp):
        with open(itp, 'r') as ITP:
            for line in ITP:
                line = line.split()

                yield line

    @property
    def mu(self):
        q_xyz = (self.coord.T * self.charges).T * (4.8 / 1.6e-29) * e * 1e-9
        q_xyz = np.sum(q_xyz, axis=0)
        return np.linalg.norm(q_xyz)


def main():
    # starting
    args = options()

    # setting files
    res = args['resname']
    res = res.lower()

    mu = MDipolar(res)

    print('Name:', mu.name)
    print('N atoms:', mu.nat)
    print('Coordinates: [ x , y , z ]\n', mu.coord)
    print('Charges: %.4f\n' % np.sum(mu.charges), mu.charges)
    print('Moment Dipolar: %.4f D' % mu.mu)


if __name__ == '__main__':
    main()
