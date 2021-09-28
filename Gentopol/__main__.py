"""
Module created from the LigParGen program used to generate and assign the force
field for OPLSAA.

"""
import argparse
from .converter import convert


def options():

    parser = argparse.ArgumentParser(
        prog='Gentopol',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
    FF formats provided :
    --------------------
    OpenMM       .xml
    CHARMM/NAMD  .prm & .rtf
    GROMACS      .itp & .gro
    CNS/X-PLOR   .param & .top
    Q            .Q.prm & .Q.lib
    DESMOND      .cms
    BOSS/MCPRO   .z
    PDB2PQR      .pqr

    Input Files supported :
    --------------------
    SMILES code
    PDB
    MDL MOL Format

    ################################################
    if using MOL file
    Usage: LigParGen -m phenol.mol    -r PHN -c 0 -o 0

    if using PDB file
    Usage: LigParGen -p phenol.pdb    -r PHN -c 0 -o 0

    if using BOSS SMILES CODE 
    Usage: LigParGen -s 'c1ccc(cc1)O' -r PHN -c 0 -o 0

    REQUIREMENTS:
    BOSS (need to set BOSSdir in bashrc and cshrc)
    Preferably Anaconda python with following modules
    pandas
    argparse
    numpy
    openbabel

    Please cite following references: 
    1. LigParGen web server: an automatic OPLS-AA parameter generator for organic ligands  
       Leela S. Dodda  Israel Cabeza de Vaca  Julian Tirado-Rives William L. Jorgensen 
       Nucleic Acids Research, Volume 45, Issue W1, 3 July 2017, Pages W331–W336
    2. 1.14*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for Condensed-Phase Simulations
       Leela S. Dodda, Jonah Z. Vilseck, Julian Tirado-Rives , and William L. Jorgensen 
       Department of Chemistry, Yale University, New Haven, Connecticut 06520-8107, United States
       J. Phys. Chem. B, 2017, 121 (15), pp 3864–3870
    3. Accuracy of free energies of hydration using CM1 and CM3 atomic charges.
       Udier–Blagović, M., Morales De Tirado, P., Pearlman, S. A. and Jorgensen, W. L. 
       J. Comput. Chem., 2004, 25,1322–1332. doi:10.1002/jcc.20059
    """
    )
    parser.add_argument(
        "-r", "--resname",
        help="Residue name from PDB FILE",
        type=str)

    parser.add_argument(
        "-s", "--smiles",
        help="Paste SMILES code from CHEMSPIDER or PubChem",
        type=str)

    parser.add_argument(
        "-m", "--mol",
        help="Submit MOL file from CHEMSPIDER or PubChem",
        type=str)

    parser.add_argument(
        "-p", "--pdb",
        help="Submit PDB file from CHEMSPIDER or PubChem",
        type=str)

    parser.add_argument(
        "-o", "--opt",
        help="Optimization or Single Point Calculation",
        type=int,
        choices=[0, 1, 2, 3])

    parser.add_argument(
        "-c", "--charge",
        type=int,
        choices=[0, -1, 1, -2, 2],
        help="0: Neutral <0: Anion >0: Cation ")

    parser.add_argument(
        "-l", "--lbcc",
        help="Use 1.14*CM1A-LBCC charges instead of 1.14*CM1A",
        action="store_true")

    return vars(parser.parse_args())


def main():

    args = options()

    convert(**args)


# RUN
main()
