"""
Submodule dedicated to deal with the inputs and configure the outputs.

"""

from .creatzmat import GenMolRep
from shutil import which
from .BOSSReader import BOSSReader, CheckForHs
from .BOSS2GMX import mainBOSS2GMX
import os
import pickle


def convert(**kwargs):

    # set the default values
    options = {
        'opt': 0,
        'smiles': None,
        'charge': 0,
        'lbcc': False,
        'mol': None,
        'resname': 'UNK',
        'pdb': None}

    # update the default values based on the arguments
    options.update(kwargs)

    # set the arguments that you would used to get from argparse
    opt = options['opt']
    smiles = options['smiles']
    charge = options['charge']
    lbcc = options['lbcc']
    resname = options['resname']
    mol = options['mol']
    pdb = options['pdb']

    if opt:
        optim = opt
    else:
        optim = 0

    clu = False

    assert which('babel'), "OpenBabel is Not installed or the executable location is not accessable"

    # cleaning tmp files
    if os.path.exists('/tmp/' + resname + '.xml'):
        os.system('/bin/rm /tmp/' + resname + '.*')

    # charges
    if lbcc:
        if charge == 0:
            lbcc = True
            print('LBCC converter is activated')
        else:
            lbcc = False
            print('1.14*CM1A-LBCC is only available for neutral molecules\n Assigning unscaled CM1A charges')

    # Types of entry
    if smiles:
        # is smiles
        # Write file smi in tmp
        os.chdir('/tmp/')
        smifile = open('%s.smi' % resname, 'w+')
        smifile.write('%s' % smiles)
        smifile.close()
        GenMolRep('%s.smi' % resname, optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, optim, charge, lbcc)

    elif mol:
        os.system('cp %s /tmp/' % mol)
        os.chdir('/tmp/')
        GenMolRep(mol.split('/')[-1], optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, optim, charge, lbcc)

    elif pdb is not None:
        os.system('cp %s /tmp/' % pdb)
        os.chdir('/tmp/')
        GenMolRep(pdb, optim, resname, charge)
        mol = BOSSReader('%s.z' % resname, optim, charge, lbcc)
        clu = True

    assert (mol.MolData['TotalQ']['Reference-Solute'] ==
            charge), "PROPOSED CHARGE IS NOT POSSIBLE: SOLUTE MAY BE AN OPEN SHELL"
    assert(CheckForHs(mol.MolData['ATOMS'])
           ), "Hydrogens are not added. Please add Hydrogens"

    pickle.dump(mol, open(resname + ".p", "wb"))
    # mainBOSS2OPM(resname, clu)
    # print('DONE WITH OPENMM')
    # mainBOSS2Q(resname, clu)
    # print('DONE WITH Q')
    # mainBOSS2XPLOR(resname, clu)
    # print('DONE WITH XPLOR')
    # mainBOSS2CHARMM(resname, clu)
    # print('DONE WITH CHARMM/NAMD')
    mainBOSS2GMX(resname, clu)
    print('DONE WITH GROMACS')
    # mainBOSS2LAMMPS(resname, clu)
    # print('DONE WITH LAMMPS')
    # mainBOSS2DESMOND(resname, clu)
    print('DONE WITH DESMOND')
    os.remove(resname + ".p")
    mol.cleanup()
