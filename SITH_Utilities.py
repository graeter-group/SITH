from multiprocessing.sharedctypes import Value
import sys
from importlib.resources import path
from pathlib import Path
from typing import Tuple

import numpy as np
from openbabel import openbabel as ob

# import Geometry #No longer needed because I thought it could fit into utilities just fine

# LTMatrix class comes from https://github.com/ruixingw/rxcclib/blob/dev/utils/my/LTMatrix.py


class LTMatrix(list):
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

    def __init__(self, name: str, nAtoms: int) -> None:
        self.name = name
        self.rawRIC = list()
        self.lengths = list()
        self.angles = list()
        self.diheds = list()
        self.energy = np.inf
        self.atoms = list()
        self.nAtoms = nAtoms
        self.dims = list()
        self.dimIndices = list()
        #self.hessian = np.ndarray()

    #! Need to adapt to new format, make sure to specify and/or convert units
    def buildCartesian(self, lines: list):
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

    def numAtoms(self):
        return self.nAtoms

    def buildRIC(self, dims: list, dimILines: list, coordLines: list):
        self.dims = [int(d) for d in dims]

        # region Indices
        # Parse through the 'dimILines' input which indicates which atoms (by index)
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
            rawIndices) == self.dims[0] * 4, "Redundant internal coordinate indices input has invalid dimensions."

        # Parse into sets of 4, then into tuples of the relevant number of values
        for i in range(0, len(rawIndices), 4):
            a1 = rawIndices[i]
            a2 = rawIndices[i+1]
            a3 = rawIndices[i+2]
            a4 = rawIndices[i+3]
            assert a1 <= self.nAtoms and a2 <= self.nAtoms and a3 <= self.nAtoms and a4 <= self.nAtoms, "Invalid atom index given as input."
            assert a1 != 0 and a2 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
            # bond lengths check
            if i < self.dims[1]*4:
                assert a3 == 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2))
            # bond angles check
            elif i < (self.dims[1] + self.dims[2])*4:
                assert a3 != 0 and a4 == 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2, a3))
            # dihedral angles check
            elif i < (self.dims[1] + self.dims[2] + self.dims[3])*4:
                assert a3 != 0 and a4 != 0, "Mismatch between given 'RIC dimensions' and given RIC indices."
                self.dimIndices.append((a1, a2, a3, a4))
            # self.dimIndices.append((int(),))

        # Check that the number of values in each tuple matches the dimension type (length, angle, dihedral) for that dim index
        # These should line up with self.dims correctly

        # endregion

        for line in coordLines:
            self.rawRIC.extend([float(ric) for ric in line.split()])
        assert len(self.rawRIC) == self.dims[0], "Mismatch between the number of degrees of freedom expected ("+str(
            dims[0])+") and number of coordinates given ("+str(len(self.rawRIC))+")."
        for i in range(len(self.rawRIC)):
            if i < int(dims[1]):
                self.lengths.append(self.rawRIC[i])
            elif i < int(dims[1]) + int(dims[2]):
                self.angles.append(self.rawRIC[i])
            elif i < int(dims[1]) + int(dims[2]) + int(dims[3]):
                self.diheds.append(self.rawRIC[i])
            else:
                raise Exception(
                    "Mismatch between RIC dimensions specified aside from the total.")

    def getAtoms(self) -> list:
        if self.atoms:
            return self.atoms
        else:
            pass

    def getEnergy(self) -> float:
        if self.energy != np.inf:
            return self.energy
        else:
            raise Exception(
                "Energy has not been set for this geometry.  You done goofed real bad.")

    def __eq__(self, __o: object) -> bool:
        #if __o is type(self):
            b = True
            b = b and self.name == __o.name
            b = b and self.rawRIC == __o.rawRIC
            b = b and self.lengths == __o.lengths
            b = b and self.angles == __o.angles
            b = b and self.diheds == __o.diheds
            b = b and self.energy == __o.energy
            b = b and self.atoms == __o.atoms
            b = b and self.nAtoms == __o.nAtoms
            b = b and self.dims == __o.dims
            b = b and self.dimIndices == __o.dimIndices
            return b
        #else:
            #return False


class Atom:
    def __init__(self, element: str, coords: list) -> None:
        self.element = element
        self.coords = coords

    def __eq__(self, __o: object) -> bool:
        return self.element == __o.element and self.coords ==__o.coords


class UnitConverter:

    def __init__(self) -> None:
        pass


class Extractor:

    def __init__(self, path: Path, linesList: list) -> None:
        self.eHeader = "Total Energy"
        self.hHeader = "Internal Force Constants"
        self.hEnder = "Mulliken Charges"
        self.xcHeader = "Current cartesian coordinates"
        self.xcEnder = "Force Field"
        self.xrHeader = "Redundant internal coordinates"
        self.xrEnder = "ZRed-IntVec"
        self.rdHeader = "Redundant internal dimensions"
        self.rdEnder = "Redundant internal coordinate indices"
        self.nHeader = "Number of atoms"
        self.aHeader = "Atomic numbers"

        self.path = path
        self.name = path.stem

        self.lines = linesList

    def writeXYZ(self):
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("fchk", "xyz")

        mol = ob.OBMol()
        assert self.path.exists(), "Path to fchk file does not exist"
        assert obConversion.ReadFile(mol, self.path.as_posix(
        )), "Reading fchk file with openbabel failed."   # Open Babel will uncompress automatically
        assert obConversion.WriteFile(mol, str(
            self.path.parent.as_posix()+self.path.root+self.path.stem+".xyz")), "Could not write XYZ file."

    def extract(self):
        self.writeXYZ()

        #! Change to a while < len loop?
        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # This all must be in order
            #! Sandbox this ASAP you dummy
            if self.nHeader in line:
                splitLine = line.split()
                numAtoms = int(splitLine[len(splitLine)-1])
                self.geometry = Geometry(self.name, numAtoms)

            elif self.aHeader in line:
                i = i+1
                lin = self.lines[i]
                atomicNums = lin.split()

            elif self.eHeader in line:
                splitLine = line.split()
                self.geometry.energy = float(splitLine[len(splitLine)-1])

            # elif self.xcHeader in line:
            #     i=i+1
            #     xcStart = i
            #     while self.xcEnder not in self.lines[i]:
            #         i=i+1
            #     xcRaw = self.lines[xcStart:i]
            #     assert (numAtoms * 3) == len(xcRaw), "Number of coordinates does not match number of atoms (3 * atoms)."
            #     self.geometry.buildCartesian(xcRaw)

            elif self.rdHeader in line:
                i = i+1
                lin = self.lines[i]
                rDims = lin.split()

            elif self.rdEnder in line:
                rdiStart = i+1
                while self.xrHeader not in self.lines[i+1]:
                    i = i+1
                xrDims = self.lines[rdiStart:i+1]
                #! assert validation of number of degrees of freedom?
                if not xrDims:
                    raise Exception(
                        "Missing 'Redundant internal coordinate indices'.")

            if self.xrHeader in line:
                i = i+1
                xrStart = i
                while self.xrEnder not in self.lines[i]:
                    i = i+1
                xrRaw = self.lines[xrStart:i]
                #! assert validation of number of degrees of freedom?
                #if not xrDims: raise Exception("Missing 'Redundant internal coordinate indices'.")
                self.geometry.buildRIC(rDims, xrDims, xrRaw)

            elif self.hHeader in line:
                hFirst = line.split()
                i = i+1
                self.hRaw = list()
                while self.hEnder not in self.lines[i]:
                    row = self.lines[i]
                    rowsplit = row.split()
                    self.hRaw.extend([float(i) for i in rowsplit])
                    i = i+1

            i = i + 1
        with open(self.path.parent.as_posix()+self.path.root + self.path.stem + ".xyz", "r") as xyz:
            xyzLines = xyz.readlines()
            self.geometry.buildCartesian(xyzLines)
        self.buildHessian()

    def buildHessian(self):
        ltMat = LTMatrix(self.hRaw)
        self.hessian = ltMat.fullmat
        self.geometry.hessian = self.hessian

    def getGeometry(self) -> Geometry:
        if self.geometry:
            return self.geometry
        else:
            raise Exception("There is no geometry.")

# if __name__ == '__main__':
#     import doctest
#     doctest.testmod()
