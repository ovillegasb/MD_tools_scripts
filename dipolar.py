#!/usr/bin/env python
# -*- conding=utf-8 -*-

import argparse

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
    parser.add_argument("-f", "--file",
                        help="resname from itp and gro file",
                        default="",
                        metavar="RES",
                        type=str)

    return vars(parser.parse_args())


def main():
    pass


if __name__ == '__main__':
    main()
