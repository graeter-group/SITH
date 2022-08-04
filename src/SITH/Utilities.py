from array import array
from pathlib import Path
import pathlib
from typing import Tuple

from ase.units import Bohr
import numpy as np
from openbabel import openbabel as ob
from ase.visualize import view
from ase import Atoms
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import HBox, VBox, Output

# TODO: Add a logger


class LTMatrix(list):
    """LTMatrix class and code comes from https://github.com/ruixingw/rxcclib/blob/dev/utils/my/LTMatrix.py"""

    def __init__(self, L):
        """
        Accept a list of elements in a lower triangular matrix.
        """
        list.__init__(self, L)
        self.list = L
        i, j = LTMatrix.getRowColumn(len(L) - 1)
        assert i == j, "Not a LTMatrix"
        self.dimension = i + 1

    def __getitem__(self, key):
        """
        Accept one or two integers.
        ONE: get item at the given position (count from zero)
        TWO: get item at the given (row, column) (both counted from zero)
        """
        if type(key) is tuple:
            return self.list[LTMatrix.getPosition(*key)]
        else:
            return self.list[key]

    @staticmethod
    def getRowColumn(N):
        """
        Return the row and column number of the Nth entry  of a lower triangular matrix.
        N, ROW, COLUMN are counted from ZERO!
        Example:
           C0 C1 C2 C3 C4 C5
        R0 0
        R1 1  2
        R2 3  4  5
        R3 6  7  8  9
        R4 10 11 12 13 14
        R5 15 16 17 18 19 20
        >>> LTMatrix.getRowColumn(18)
        (5, 3)
        18th element is at row 5 and column 3. (count from zero)
        """
        N += 1
        y = int((np.sqrt(1 + 8 * N) - 1) / 2)
        b = int(N - (y**2 + y) / 2)
        if b == 0:
            return (y - 1, y - 1)
        else:
            return (y, b - 1)

    @staticmethod
    def getPosition(i, j):
        """
        Return the number of entry in the i-th row and j-th column of a symmetric matrix.
        All numbers are counted from ZERO.
        >>> LTMatrix.getPosition(3, 4)
        13
        """
        i += 1
        j += 1
        if i < j:
            i, j = j, i
        num = (i * (i - 1) / 2) + j
        num = int(num)
        return num - 1

    def buildFullMat(self):
        """
        build full matrix (np.ndarray).
        """
        L = []
        for i in range(self.dimension):
            L.append([])
            for j in range(self.dimension):
                L[-1].append(self[i, j])
        self._fullmat = np.array(L)
        return self._fullmat

    @property
    def fullmat(self):
        """
        Return the full matrix (np.ndarray)
        """
        return getattr(self, '_fullmat', self.buildFullMat())

    def inverse(self):
        return np.linalg.inv(self.fullmat)

    def uppermat(self):
        up = []
        for i, row in enumerate(self.fullmat):
            up.extend(row[i:])
        return up

    @staticmethod
    def newFromUpperMat(uppermat):
        """
        Example:
           C0 C1 C2 C3 C4 C5
        R0  0  1  2  3  4  5
        R1     6  7  8  9 10
        R2       11 12 13 14
        R3          15 16 17
        R4             18 19
        R5                20
        """
        dimension = LTMatrix.getRowColumn(len(uppermat)-1)[0] + 1

        def getUpperPosition(i, j, dimension):
            firstnum = dimension
            if i == 0:
                return j
            lastnum = firstnum - i + 1
            countnum = i
            abovenums = (firstnum + lastnum)*countnum / 2
            position = abovenums + j - i
            return int(position)
        L = []
        for i in range(dimension):
            for j in range(i+1):
                L.append(uppermat[getUpperPosition(j, i, dimension)])
        return LTMatrix(L)


class Geometry:
    """Houses data associated with a molecular structure, all public variables are intended for access not modification."""

    def __init__(self, name: str, path: pathlib.Path, nAtoms: int) -> None:
        self.name = name
        """Name of geometry, based off of stem of .fchk file path unless otherwise modified."""
        self._path = path
        """Path of geometry .fchk file."""
        self.ric = array('f')
        """Redundant Internal Coordinates of geometry in atomic units (Bohr radius)"""
        self.energy = None
        """Energy associated with geometry based on the DFT or higher level calculations used to generate the .fchk file input"""
        self.atoms = list()
        """List of <SITH_Utilities.Atom> objects associated with geometry."""
        self.nAtoms = nAtoms
        """Number of atoms"""
        self.dims = array('i')
        """Array of number of dimensions of DOF type
        [0]: total dimensions/DOFs
        [1]: bond lengths
        [2]: bond angles
        [3]: dihedral angles
        """
        self.dimIndices = list()
        """List of Tuples referring to the indices of the atoms involved in each dimension/DOF in order of DOF index in ric"""

        self.hessian = None
        """Hessian matrix associated with the geometry"""

    def buildCartesian(self, lines: list):
        """Takes in a list of str containing the lines of a .xyz file, populates self.atoms

        -----
        Data is assumed to be Cartesian and in Angstroms as is standard for .xyz files"""
        first = lines[0]
        if len(first.split()) == 1:
            nAtoms = int(first)
            if nAtoms != self.nAtoms:
                raise Exception("Mismatch in number of atoms.")
        for line in lines:
            sLine = line.split()
            if len(sLine) > 1 and len(sLine) < 5:
                a = Atom(sLine[0], sLine[1:4])
                self.atoms.append(a)
            elif len(sLine) >= 5:
                pass

    def buildRIC(self, dims: list, dimILines: list, coordLines: list):
        """
        Takes in lists of RIC-related data, Populates 

            dims: quantities of each RIC dimension type
            dimILines: list of strings of each line of RIC Indices 
            coordLines: list of strings of each line of RICs
        """
        try:
            self.dims = array('i', [int(d) for d in dims])
        except ValueError:
            raise Exception("Invalid input given for Redundant internal dimensions.")
        assert self.dims[0] == self.dims[1] + self.dims[2] + self.dims[3] and len(
            dims) == 4, "Invalid quantities of dimension types (bond lengths, angles, dihedrals) given in .fchk."

        # region Indices
        # Parses through the 'dimILines' input which indicates which atoms (by index)
        # are involved in each RIC degree of freedom

        rawIndices = list()
        for iLine in dimILines:
            iSplit = iLine.split()
            try:
                rawIndices.extend([int(i) for i in iSplit])
            except ValueError as ve:
                print(ve)
                raise Exception("Invalid atom index given as input.")

        # Check that # indices is divisible by 4
        assert len(rawIndices) % 4 == 0 and len(
            rawIndices) == self.dims[0] * 4, "One or more redundant internal coordinate indices are missing or do not have the expected format. Please refer to documentation"

        # Parse into sets of 4, then into tuples of the relevant number of values
        lengthsCount = 0
        anglesCount = 0
        dihedsCount = 0
        for i in range(0, len(rawIndices), 4):
            a1 = rawIndices[i]
            a2 = rawIndices[i+1]
            a3 = rawIndices[i+2]
            a4 = rawIndices[i+3]
            # Check that the number of values in each tuple matches the dimension type (length, angle, dihedral) for that dim index
            # These should line up with self.dims correctly
            assert all([(x <= self.nAtoms and x >= 0)
                       for x in rawIndices[i:i+4]]), "Invalid atom index given as input."
            assert a1 != a2 and a1 != a3 and a1 != a4 and a2 != a3 and a2 != a4 and (
                a3 != a4 or a3 == 0), "Invalid RIC dimension given, atomic indices cannot repeat within a degree of freedom."
            assert a1 != 0 and a2 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
            # bond lengths check
            if i < self.dims[1]*4:
                assert a3 == 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2))
                lengthsCount += 1
            # bond angles check
            elif i < (self.dims[1] + self.dims[2])*4:
                assert a3 != 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2, a3))
                anglesCount += 1
            # dihedral angles check
            elif i < (self.dims[1] + self.dims[2] + self.dims[3])*4:
                assert a3 != 0 and a4 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2, a3, a4))
                dihedsCount += 1

        assert lengthsCount == self.dims[1] and anglesCount == self.dims[2] and dihedsCount == self.dims[
            3], "Redundant internal coordinate indices given inconsistent with Redundant internal dimensions given."

        # endregion

        try:
            for line in coordLines:
                self.ric.extend([float(ric) for ric in line.split()])
        except ValueError:
            raise(Exception("Redundant internal coordinates contains invalid values, such as strings."))

        self.ric = np.asarray(self.ric, dtype=np.float32)
        assert len(self.ric) == self.dims[0], "Mismatch between the number of degrees of freedom expected ("+str(
            dims[0])+") and number of coordinates given ("+str(len(self.ric))+")."

        # Angles are moved from (-pi, pi) --> (0, 2pi) because abs(angles) are often around pi (phi, psi angles especially)
        # so when calculating deltaQ this is more convenient
        for i in range(self.dims[1], self.dims[0]):
            self.ric[i] = self.ric[i] + np.pi

    def _killDOFs(self, dofis: list[int]):
        """Takes in list of indices of degrees of freedom to remove, Removes DOFs from ric, dimIndices, and hessian, updates dims"""
        self.ric = np.delete(self.ric, dofis)
        self.dimIndices = np.delete(self.dimIndices, dofis)
        lengthsDeleted = sum(x < self.dims[1] and x >= 0 for x in dofis)
        anglesDeleted = sum(
            x < self.dims[2] + self.dims[1] and x >= self.dims[1] for x in dofis)
        dihedralsDeleted = sum(
            x < self.dims[0] and x >= self.dims[2] + self.dims[1] for x in dofis)
        self.dims[0] -= len(dofis)
        self.dims[1] -= lengthsDeleted
        self.dims[2] -= anglesDeleted
        self.dims[3] -= dihedralsDeleted
        if(self.hessian is not None):
            self.hessian = np.delete(self.hessian, dofis, axis=0)
            self.hessian = np.delete(self.hessian, dofis, axis=1)

    def __eq__(self, __o: object) -> bool:
        b = True
        b = b and self.name == __o.name
        b = b and all(self.ric == __o.ric)
        b = b and self.energy == __o.energy
        b = b and self.atoms == __o.atoms
        b = b and self.nAtoms == __o.nAtoms
        b = b and self.dims == __o.dims
        b = b and self.dimIndices == __o.dimIndices
        return b


class Atom:
    """Holds Cartesian coordinate data as well as element data"""

    def __init__(self, element: str, coords: list) -> None:
        self.element = element
        self.coords = coords

    def __eq__(self, __o: object) -> bool:
        return self.element == __o.element and self.coords == __o.coords


class UnitConverter:
    """
    Class to convert units utilizing Atomic Simulation Environment (ase) constants
    xyz standard input is in Angstrom
    RIC are in atomic units
    internal Hessian is in atomic units of length: Ha/Bohr^2 angle: Hartree/radian^2
    Note: values in all Gaussian version 3 formatted checkpoint files are in atomic units
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def angstromToBohr(a: float) -> float:
        return a * Bohr

    @staticmethod
    def bohrToAngstrom(b: float) -> float:
        return b / Bohr

    @staticmethod
    def radianToDegree(r: float) -> float:
        return


class Extractor:
    """Used on a per .fchk file basis to organize lines from the fchk file into a Geometry

    -----
    The user really shouldn't be using this class 0.0 unless they perhaps want a custom one for
    non-fchk files, in which case they should make a new class inheriting from this one."""

    def __init__(self, path: Path, linesList: list) -> None:
        self.__energyHeader = "Total Energy"
        self.__hessianHeader = "Internal Force Constants"
        self.hEnder = "Mulliken Charges"
        self.__cartesianCoordsHeader = "Current cartesian coordinates"
        self.xcEnder = "Force Field"
        self.__ricHeader = "Redundant internal coordinates"
        self.xrEnder = "ZRed-IntVec"
        self.__ricDimHeader = "Redundant internal dimensions"
        self.__ricIndicesHeader = "Redundant internal coordinate indices"
        self.__numAtomsHeader = "Number of atoms"
        self.__atomicNumsHeader = "Atomic numbers"

        self._path = path
        self._name = path.stem

        self.__lines = linesList

    # This is kept because using SithWriter.writeXYZ(self.geometry) would cause circular dependency
    def _writeXYZ(self):
        """
        Writes a .xyz file of the geometry using OpenBabel and the initial SITH input .fchk file
        """
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("fchk", "xyz")

        mol = ob.OBMol()
        assert self._path.exists(), "Path to fchk file does not exist"
        assert obConversion.ReadFile(mol, self._path.as_posix(
        )), "Reading fchk file with openbabel failed."
        assert obConversion.WriteFile(mol, str(
            self._path.parent.as_posix()+self._path.root+self._path.stem+".xyz")), "Could not write XYZ file."

    def _extract(self) -> bool:
        """Extracts and populates Geometry information from self.__lines, Returns false is unsuccessful"""

        try:
            i = 0
            while i < len(self.__lines):
                line = self.__lines[i]

                # This all must be in order because of the way things show up in the fchk file, it makes no sense to build
                # these out separately to reduce dependencies because it will just increase the number of times to traverse the data.
                if self.__numAtomsHeader in line:
                    splitLine = line.split()
                    numAtoms = int(splitLine[len(splitLine)-1])
                    self.geometry = Geometry(self._name, self._path, numAtoms)

                elif self.__energyHeader in line:
                    splitLine = line.split()
                    self.geometry.energy = float(splitLine[len(splitLine)-1])

                elif self.__ricDimHeader in line:
                    i = i+1
                    lin = self.__lines[i]
                    rDims = lin.split()

                elif self.__ricIndicesHeader in line:
                    rdiStart = i+1
                    stop = int(line.split()[-1])
                    count = 0
                    while count < stop:
                        count += len(self.__lines[i].split())
                        i += 1
                    xrDims = self.__lines[rdiStart:i+1]
                    assert len(
                        xrDims) > 0, "Missing Redundant internal coordinate indices."

                if self.__ricHeader in line:
                    i = i+1
                    xrStart = i
                    stop = int(line.split()[-1])
                    count = 0
                    while count < stop:
                        count += len(self.__lines[i].split())
                        i += 1
                    xrRaw = self.__lines[xrStart:i]
                    self.geometry.buildRIC(rDims, xrDims, xrRaw)

                elif self.__hessianHeader in line:
                    i = i+1
                    stop = int(line.split()[-1])
                    self.hRaw = list()
                    while len(self.hRaw) < stop:
                        row = self.__lines[i]
                        rowsplit = row.split()
                        self.hRaw.extend([float(i) for i in rowsplit])
                        i = i+1

                i = i + 1
            print("Building full Hessian matrix.")
            self.buildHessian()

            print("Translating .fchk file to new .xyz file with OpenBabel...")
            # Using the SithWriter method here would cause a circular dependency
            # TODO: remedy this by building it out separately if possible, but not a big deal as long as only write method
            self._writeXYZ()

            print("Opening .xyz file...")
            with open(self._path.parent.as_posix()+self._path.root + self._path.stem + ".xyz", "r") as xyz:
                xyzLines = xyz.readlines()
                self.geometry.buildCartesian(xyzLines)
            print("Cartesian data extracted successfully.")
            return True
        except Exception as e:
            print(e)
            print("Data extraction failed.")
            return False

    def buildHessian(self):
        """Properly formats the Hessian matrix from the lower triangular matrix given by the .fchk data"""
        ltMat = LTMatrix(self.hRaw)
        self.hessian = ltMat.fullmat
        self.geometry.hessian = self.hessian

    def getGeometry(self) -> Geometry:
        """Returns the geometry populated by the Extractor based on the input lines from a .fchk file"""
        if self.geometry:
            return self.geometry
        else:
            raise Exception("There is no geometry.")


#region Visualization

class SithViewer:
    def __init__(self, atoms):
        ''' Set of graphic tools to see the distribution
        of energies in the different degrees of freedom
        (lengths, angles, dihedrals)'''
        self.atoms = atoms
        self.viewer = view(atoms, viewer='ngl')
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.shape = self.viewer.view.shape
        self.box = self.viewer

    def add_bond(self, atom1index, atom2index,
                 color=[0.5, 0.5, 0.5], radius=0.1):
        ''' Add a bond between two atoms:
        atom1 and atom2

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms to be connected.

        color: list. Default gray([0.5, 0.5, 0.5])
            RGB triplet.

        radius: float. Default 0.1
            Radius of the bond.

        Output
        ======

        Return the bonds int the system
        '''

        indexes = [atom1index, atom2index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        self.remove_bond(atom1index, atom2index)
        b = self.shape.add_cylinder(self.atoms[atom1index].position,
                                    self.atoms[atom2index].position,
                                    color,
                                    radius)

        self.bonds[name] = b

        return self.bonds[name]

    def add_bonds(self, atoms1indexes, atoms2indexes, colors=None, radii=None):
        ''' Add a bond between each pair of atoms corresponding to
        two lists of atoms:
        atoms1 and atoms.

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms to be connected
        color: list of color lists. Default all gray([0.5, 0.5, 0.5])
            RGB triplets for each of the bonds. It can be one a triplet
            in case of just one color in all bonds.
        radii: float or list of floats. Default 0.1
            radius of each bond.

        Output
        ======

        Return the bonds int the system
        '''

        if colors is None:
            colors = [0.5, 0.5, 0.5]

        if type(colors[0]) is not list:
            colors = [colors for i in range(len(atoms1indexes))]

        if radii is None:
            radii = 0.07

        if type(radii) is not list:
            radii = [radii for i in range(len(atoms1indexes))]

        assert len(atoms1indexes) == len(atoms2indexes), \
            "The number of atoms in both lists must be the same"
        assert len(atoms1indexes) == len(colors), \
            "The number of colors in must be the same as the number of atoms"
        assert len(atoms1indexes) == len(radii), \
            "The number of radii must be the same as the number of atoms"

        for i in range(len(atoms1indexes)):
            self.add_bond(atoms1indexes[i],
                          atoms2indexes[i],
                          colors[i],
                          radii[i])
        return self.bonds

    def remove_bond(self, atom1index, atom2index):
        ''' Remove a bond between two atoms:
        atoms1 and atoms2.

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms that are connected. This bond
            will be removed.

        Output
        ======

        Return the bonds int the system
        '''
        indexes = [atom1index, atom2index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.bonds.keys():
            self.viewer.view.remove_component(self.bonds[name])
            del self.bonds[name]

    def remove_bonds(self, atoms1indexes=None, atoms2indexes=None):
        ''' remove several bonds in the plot between two list of atoms:
        atoms1 and atoms2.

        Parameters
        ==========

        atom1index (and atom2index): list[int]
            Indexes of the atoms that are connected.

        Note: if atoms2 is None, all bonds with atoms1 will me removed.
        If atoms1 and atoms2 are None, all bonds in the structure are
        removed.
        '''

        if (atoms1indexes is None) and (atoms2indexes is None):
            for name in self.bonds.keys():
                self.viewer.view.remove_component(self.bonds[name])
            self.bonds.clear()
            return self.bonds

        elif (atoms1indexes is not None) and (atoms2indexes is None):
            to_remove = []
            for name in self.bonds.keys():
                for index in atoms1indexes:
                    if str(index) in name:
                        self.viewer.view.remove_component(self.bonds[name])
                        to_remove.append(name)
            for name in to_remove:
                del self.bonds[name]
            return self.bonds

        else:
            assert len(atoms1indexes) == len(atoms2indexes), \
                "The number of atoms in both lists must be the same"
            [self.remove_bond(index1, index2)
             for index1, index2 in
             zip(atoms1indexes, atoms2indexes)]
            return self.bonds

    def plot_arc(self, vertex, arcdots, color):
        ''' Add an arc using triangles.

        Parameters
        ==========

        vertex: array
            center of the arc
        arcdots: list of arrays
            vectors that define the points of the arc. These
            vectors must be defined respect the vertex.

        Output
        ======

        Return the triangles in the angle.
        '''

        triangles = []
        for i in range(len(arcdots)-1):
            vertexes = np.hstack((vertex,
                                  vertex + arcdots[i],
                                  vertex + arcdots[i+1]))
            t = self.shape.add_mesh(vertexes, color)
            triangles.append(t)

        return triangles

    def add_angle(self, atom1index, atom2index, atom3index,
                  color=[0.5, 0.5, 0.5], n=0):
        ''' Add an angle to between three atoms:
        atom1, atom2 and atom3
        - with the vertex in the atom2

        Parameters
        ==========

        atom1index, atom2index and atom3index: int
            Indexes of the three atoms that defines the angle.
        color: color list. Default all gray([0.5, 0.5, 0.5])
            RGB triplet.
        n: int. Default 10
            number of intermedia points to add in the arc of
            the angle.

        Output
        ======
        Return the angles in the system
        '''

        indexes = [atom1index, atom2index, atom3index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)
        self.remove_angle(atom1index, atom2index, atom3index)
        self.angles[name] = []

        vertex = self.atoms[atom2index].position
        side1 = self.atoms[atom1index].position - vertex
        side2 = self.atoms[atom3index].position - vertex
        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1/lenside1
        side2 = 0.7 * lensides * side2/lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.angles[name] = self.plot_arc(vertex, arcdots, color)

        return self.angles[name]

    def intermedia_vectors(self, a, b, n):
        ''' Define the intermedia arc dots between two vectors

        Parameters
        ==========

        a, b: arrays
             side vectors of the angles.
        n: int
             number of intermedia dots.

        Output
        ======
        Return the intermedia vectors between two side vectors.
        '''

        if n == 0:
            return []
        n += 1
        c = b - a
        lena = np.linalg.norm(a)
        lenb = np.linalg.norm(b)
        lenc = np.linalg.norm(c)
        lend = min(lena, lenb)

        theta_total = np.arccos(np.dot(a, b)/(lena * lenb))
        beta = np.arccos(np.dot(a, c)/(lena * lenc))
        intermedia = []

        for i in range(1, n):
            theta = i * theta_total/n
            gamma = beta - theta
            factor = (lena * np.sin(theta))/(lenc * np.sin(gamma))
            dird = a + factor * c
            d = lend * dird/np.linalg.norm(dird)
            intermedia.append(d)
        return intermedia

    def remove_angle(self, atom1index, atom2index, atom3index):
        '''
        Remove an angle if it exists

        Parameters
        ==========

        atom1index (and atom2/3index): int
            Indexes of the three atoms that defines the angle
            to remove.

        Output
        ======
        Return the angles
        '''
        indexes = [atom1index, atom2index, atom3index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.angles.keys():
            for triangle in self.angles[name]:
                self.viewer.view.remove_component(triangle)
            del self.angles[name]

        return self.angles

    def remove_all_angles(self):
        ''' remove all angles'''
        names = self.angles.keys()

        for name in names:
            for triangle in self.angles[name]:
                self.viewer.view.remove_component(triangle)
        self.angles.clear()

    def add_dihedral(self, atom1index, atom2index, atom3index,
                     atom4index, color=[0.5, 0.5, 0.5], n=0):
        ''' Add an dihedral angle between four atoms:
        atom1, atom2, atom3 and atom4
        - with the vertex in the midle of the atom 2 and 3

        Parameters
        ==========

        atom1index, atom2index, atom3index and atom4index: int
            Indexes of the three atoms that defines the angle.
        color: color list. Default all gray([0.5, 0.5, 0.5])
            RGB triplet.
        n: int. Default 10
            number of intermedia points to add in the arc of
            the angle.

        Output
        ======
        Return the dihedral angles
        '''
        indexes = [atom1index, atom2index, atom3index, atom4index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        axis = (self.atoms[atom3index].position -
                self.atoms[atom2index].position)
        vertex = 0.5 * (self.atoms[atom3index].position +
                        self.atoms[atom2index].position)
        axis1 = (self.atoms[atom1index].position -
                 self.atoms[atom2index].position)
        axis2 = (self.atoms[atom4index].position -
                 self.atoms[atom3index].position)

        side1 = axis1 - axis * (np.dot(axis, axis1)/np.dot(axis, axis))
        side2 = axis2 - axis * (np.dot(axis, axis2)/np.dot(axis, axis))

        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1/lenside1
        side2 = 0.7 * lensides * side2/lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.dihedrals[name] = self.plot_arc(vertex, arcdots, color)

        return self.dihedrals[name]

    def remove_dihedral(self, atom1index, atom2index, atom3index, atom4index):
        ''' Remove the dihedral angle between 4 atoms:

        atom1, atom2, atom3 and atom4

        Parameters
        ==========

        atom1index, atom2index, atom3index and atom4index: int
            Indexes of the three atoms that defines the angle.

        Output
        ======

        Return the dihedral angles
        '''
        indexes = [atom1index, atom2index, atom3index, atom4index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.dihedrals.keys():
            for triangle in self.dihedrals[name]:
                self.viewer.view.remove_component(triangle)
            del self.dihedrals[name]
        return self.dihedrals

    def remove_all_dihedrals(self):
        ''' remove all dihedral angles'''
        names = self.dihedrals.keys()

        for name in names:
            for triangle in self.dihedrals[name]:
                self.viewer.view.remove_component(triangle)
        self.dihedrals.clear()

    def download_image(self):
        self.viewer.view.download_image()

    def picked(self):
        return self.viewer.view.picked

class VisualizeEnergies(SithViewer):
    def __init__(self, sith_object, idef=0, **kwargs):
        """
        Set of tools to show a molecule and the 
        distribution of energies in the different DOF.

        Params
        ======

        sith_object : 
            sith object
        idef: int
            number of the deformation to be analyzed. Default=0 
        """
        self.idef = idef
        self.sith = sith_object
        if self.sith.energies is None:
            self._analyze_energies(**kwargs)
            
        # CHANGE: this could be imported directly from
        # sith as an ase.Atoms object or, at least, coordinates
        # must be float from sith.
        molecule = ''.join([atom.element for atom in 
                            self.sith._relaxed.atoms])
        positions = [[float(component) for component in atom.coords]
                     for atom in self.sith._relaxed.atoms]
        atoms = Atoms(molecule, positions)
        
        SithViewer.__init__(self, atoms)
        
        dims = self.sith._relaxed.dims
        self.nbonds = dims[1]
        self.nangles = dims[2]
        self.ndihedral = dims[3]

    def _analyze_energies(self, dofs=[]):
        """
        Execute JEDI method to obtain the energies of the
        DOFs

        see: 
        Parameters
        ==========

        dofs : list of tuples
            Degrees of freedom to be removed from the analysis.
        """
        if self.sith.energies is None:
            self.sith.setKillDOFs(dofs)
            self.sith.extractData()
            self.sith.energyAnalysis()

    def add_dof(self, dof, color=[0.5, 0.5, 0.5], n=5, radius=0.07):
        """
        Add the degree of freedom to the molecule image

        Parameters
        ==========

        dof: tuple
            label of the degree of freedom according with g09 convention.

        Example
        =======
            i=(1, 2) means a bond between atoms 1 and 2
            i=(1, 2, 3) means an angle between atoms 1, 2 and 3
            i=(1, 2, 3, 4) means a dihedral angle between atoms 1, 2 and 3
        """
        
        types = ["bond", "angle", "dihedral"]
        type_dof = types[len(dof)-2]
        
        if type_dof == "bond":
            index1, index2 = dof
            index1 -= 1
            index2 -= 1
            return self.add_bond(index1, index2, color, radius=radius)
            

        elif type_dof == "angle":
            index1, index2, index3 = dof
            index1 -= 1
            index2 -= 1
            index3 -= 1
            return self.add_angle(index1, index2, index3, color, n=n)

        elif type_dof == "dihedral":
            index1, index2, index3, index4 = dof
            index1 -= 1
            index2 -= 1
            index3 -= 1
            index4 -= 1
            return self.add_dihedral(index1, index2, index3,
                                  index4, color, n=n)
        else:
            raise TypeError(f"{dof} does not seem an accepted degree of freedom.")
            
    def energies_bonds(self, **kwargs):
        """
        Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.
        
        Parameters
        ==========
        
        optional kwargs for energies_some_dof
        """
        dofs = self.sith._relaxed.dimIndices[:self.nbonds]
        self.energies_some_dof(dofs, **kwargs)

    def energies_angles(self, **kwargs):
        """
        Add the angles with a color scale that represents the
        distribution of energy according to the JEDI method.
        
        Parameters
        ==========
        
        optional kwargs for energies_some_dof
        """
        dofs = self.sith._relaxed.dimIndices[self.nbonds:self.nbonds+self.nangles]
        self.energies_some_dof(dofs, **kwargs)
    
    def energies_dihedrals(self, **kwargs):
        """
        Add the dihedral angles with a color scale that represents the
        distribution of energy according to the JEDI method.
        
        Parameters
        ==========
        
        optional kwargs for energies_some_dof
        """
        dofs = self.sith._relaxed.dimIndices[self.nbonds+self.nangles:]
        self.energies_some_dof(dofs, **kwargs)

    def energies_all_dof(self, **kwargs):
        """
        Add all DOF with a color scale that represents the
        distribution of energy according to the JEDI method.
        
        Parameters
        ==========
        
        optional kwargs for energies_some_dof
        """
        dofs = self.sith._relaxed.dimIndices
        self.energies_some_dof(dofs, **kwargs)
        
        
    def energies_some_dof(self, dofs, cmap = mpl.cm.get_cmap("Blues"), label="Energy [a.u]", labelsize=20, 
                     orientation="vertical", div=5, deci=2, width="700px", height="500px", **kwargs):
        """
        Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.
        
        Parameters
        ==========
        
        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.
            
        cmap: cmap. Default: mpl.cm.get_cmap("Blues")
            cmap used in the color bar.
        
        label: str. Default: "Energy [a.u]"
            label of the color bar.
        
        labelsize: float. 
            size of the label in the 
        
        orientation: "vertical" or "horizontal". Default: "vertical"
            orientation of the color bar.
            
        div: int. Default: 5
            number of colors in the colorbar.
        
        """
        energies = []
        for dof in dofs:
            for index, sithdof in enumerate(self.sith._relaxed.dimIndices):
                if dof  == sithdof:
                    energies.append(self.sith.energies[index][self.idef])

        assert len(dofs) == len(energies), f"The number of DOFs ({len(dofs)}) does not correspond with the number of energies ({len(energies)})"
        
        minval = min(energies)
        maxval = max(energies)
        
        if orientation == 'v' or orientation == 'vertical':
            rotation=0
        else:
            rotation=90


        boundaries = np.linspace(minval, maxval, div+1)
        normalize = mpl.colors.BoundaryNorm(boundaries, cmap.N)


        self.fig, self.ax = plt.subplots(figsize=(0.5, 8))
        
        # Costumize cbar
        cbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=normalize, cmap=cmap),
                                 cax=self.ax, orientation='vertical',
                                 format='%1.{}f'.format(deci), )
        cbar.set_label(label=label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=0.8*labelsize, rotation=rotation)
        
        # Insert colorbar in view
        self.viewer.view._remote_call("setSize", targe="Widget",
                                      args=[width, height])
        for i, dof in enumerate(dofs):
            color = cmap(normalize(energies[i]))[:3]
            self.add_dof(dof, color=color, **kwargs)
            
        self.viewer.view._remote_call("setSize", targe="Widget", args=[width, height])
        out = Output()
        with out:
            plt.show()
        self.box = HBox(children=[self.viewer.view, out])

    def show_dof(self, dofs, **kwargs):
        """
        Show specific degrees of freedom. 

        Params
        ======

        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        for dof in dofs:
            self.add_dof(dof, **kwargs)

    def show_bonds(self, **kwargs):
        """
        Show the bonds in the molecule of freedom. 

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        dofs = self.sith._relaxed.dimIndices[:self.nbonds]
        self.show_dof(dofs, **kwargs)

    def show(self):
        """
        Show the molecule.
        """
        return self.box
