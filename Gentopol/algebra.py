import numpy as np
from numpy import linalg


def bossElement2Num(elem):
    symb2mass = {
        'H':  1,
        'B':  5,
        'C':  6,
        'N':  7,
        'O':  8,
        'F':  9,
        'Si': 14,
        'P': 15,
        'S': 16,
        'Cl': 17,
        'Br': 35,
        'I': 53,
    }
    try:
        res = symb2mass[elem]
    except NameError:
        print("Mass for atom %s is not available \n add it to symb2mass dictionary")
    return res


def Distance(u, v):
    """Return length of a vector."""
    u = np.array(u)
    v = np.array(v)

    return linalg.norm(u - v)


def angle(p0, p1, p2):
    "Return angle [0..pi] between two vectors."
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)

    v0 = p0 - p1
    v1 = p2 - p1

    cos_a = np.dot(v0, v1) / linalg.norm(v0) / linalg.norm(v1)

    return 180.0 * np.arccos(round(cos_a, 3)) * 7.0 / 22.0


def Mol_angle(v0, v1):
    "Return angle [0..pi] between two vectors."
    cos_a = round(np.dot(v0, v1) / linalg.norm(v0) / linalg.norm(v1), 3)
    return np.arccos(cos_a)


def dihedral(p0, p1, p2, p3):
    """Return angle [0..2*pi] formed by vertices p0-p1-p2-p3."""
    p0 = np.array(p0)
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    v01 = p0 - p1
    v32 = p3 - p2
    v12 = p1 - p2

    v0 = np.cross(v12, v01)
    v3 = np.cross(v12, v32)
    # The cross product vectors are both normal to the axis
    # vector v12, so the angle between them is the dihedral
    # angle that we are looking for.  However, since "angle"
    # only returns values between 0 and pi, we need to make
    # sure we get the right sign relative to the rotation axis
    a = Mol_angle(v0, v3)
    if np.dot(np.cross(v0, v3), v12) > 0:
        a = -a
    return a * 180.0 * 7.0 / 22.0


def pairing_func(a, b):
    ans = (a + b) * (a + b + 1) * 0.5
    if a > b:
        ans = ans + a
    else:
        ans = ans + b
    return int(ans)


def tor_id(a):
    bond = pairing_func(a[1], a[2])
    ends = pairing_func(a[0], a[3])
    return '%d-%d' % (bond, ends)


def ang_id(a):
    bond_a = pairing_func(a[0], a[1])
    bond_b = pairing_func(a[1], a[2])
    return pairing_func(bond_a, bond_b)
