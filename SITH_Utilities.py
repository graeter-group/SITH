from pathlib import Path
import sys
from typing import Tuple
import LTMatrix
from openbabel import openbabel as ob
import numpy as np
import Geometry

class Extractor:

    def __init__(self, name:str, linesList:list) -> None:
        self.eHeader = "Total Energy"
        self.hHeader = "Internal Force Constants"
        self.hEnder = "Mulliken Charges"
        self.xcHeader = "Current cartesian coordinates"
        self.xcEnder = "Force Field"
        self.xrHeader = "Redundant internal coordinates"
        self.xrEnder = "ZRed-IntVec"
        self.nHeader = "Number of atoms"
        self.aHeader = "Atomic numbers"
        self.geometry = Geometry(name)

        self.lines = linesList

        self.obConversion = ob.OBConversion()
        self.obConversion.SetInAndOutFormats("fchk", "xyz")

        mol = ob.OBMol()
        self.obConversion.ReadFile(mol, sys.argv[1])   # Open Babel will uncompress automatically
        #! find a way to just get it out as list of string or something
        self.extract()

    def extract(self):
        
        for i in range(0, len(self.lines)):
            line = self.lines[i]

            if self.nHeader in line:
                splitLine = line.split()
                numAtoms = int(splitLine[len(splitLine)-1])

            elif self.aHeader in line:
                i=i+1
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

            elif "Redundant internal dimensions" in line:
                i=i+1
                lin = self.lines[i]
                rDims = lin.split()


            elif self.xrHeader in line:
                i=i+1
                xrStart = i
                while self.xrEnder not in self.lines[i]:
                    i=i+1
                xrRaw = self.lines[xrStart:i]
                #! assert validation of number of degrees of freedom?
                self.geometry.buildRIC(rDims, xrRaw)

            elif self.hHeader in line:
                i=i+1
                hRaw = list()
                while self.hEnder not in self.lines[i]:
                    row = self.lines[i]
                    rowsplit = row.split()
                    hRaw.extend([float(i) for i in rowsplit])
                    i=i+1
                break

    def getGeometry(self) -> Geometry:
        if self.geometry:
            return self.geometry
        else:
            sys.exit("There is no geometry.")

    def getHessian(self):
        pass


    

class UnitConverter:

    def __init__(self) -> None:
        pass