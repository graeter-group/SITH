from importlib.resources import path
from pathlib import Path
import sys
from typing import Tuple
import LTMatrix #Move into here soon, be sure to credit og github though
from openbabel import openbabel as ob
import numpy as np
#import Geometry #No longer needed because I thought it could fit into utilities just fine

class Extractor:

    def __init__(self, path:Path, linesList:list) -> None:
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

        self.name = path.name

        self.lines = linesList

        self.obConversion = ob.OBConversion()
        self.obConversion.SetInAndOutFormats("fchk", "xyz")

        mol = ob.OBMol()
        self.obConversion.ReadFile(mol, path.as_posix())   # Open Babel will uncompress automatically
        self.obConversion.WriteFile(mol, str(path.name+".xyz"))
        #! find a way to just get it out as list of string or something
        self.extract()

    def extract(self):
        
        for i in range(0, len(self.lines)):
            line = self.lines[i]

            #This all must be in order
            #! Sandbox this ASAP you dummy
            if self.nHeader in line:
                splitLine = line.split()
                numAtoms = int(splitLine[len(splitLine)-1])
                self.geometry = Geometry(self.name, numAtoms)

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

            elif self.rdHeader in line:
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
                hFirst = line.split()
                i=i+1
                hRaw = list()
                while self.hEnder not in self.lines[i]:
                    row = self.lines[i]
                    rowsplit = row.split()
                    hRaw.extend([float(i) for i in rowsplit])
                    i=i+1
                break
        with open(self.geometry.name + ".xyz", "r") as xyz:
            xyzLines = xyz.readlines()
            self.geometry.buildCartesian(xyzLines)


    def getGeometry(self) -> Geometry:
        if self.geometry:
            return self.geometry
        else:
            sys.exit("There is no geometry.")

    def getHessian(self):
        pass

from nis import match
import sys
import numpy as np

class Geometry:

    def __init__(self, name:str, nAtoms:int) -> None:
        self.name = name
        self.rawRIC = list()
        self.lengths = list()
        self.angles = list()
        self.diheds = list()
        self.energy = np.inf
        self.atoms = list()
        self.nAtoms = nAtoms
        self.dims = list()


    #! Need to adapt to new format, make sure to specify and/or convert units
    def buildCartesian(self, lines:list):
        first = lines[0]
        if len(first.split()) == 1:
            nAtoms = int(first)
            if nAtoms != self.nAtoms:
                sys.exit("Mismatch in number of atoms.")
        for line in lines:
            sLine = line.split()
            if len(sLine) > 1 and len(sLine) < 5:
                a = Atom(sLine[0], sLine[1:4])
                self.atoms.append(a)
            elif len(sLine) >= 5:
                pass
        

    def numAtoms(self):
        return len(self.atoms)


    def buildRIC(self, dims:list, lines:list):
        self.dims = dims
        for line in lines:
            self.rawRIC.extend(line.split())
        if len(self.rawRIC) != dims[0]:
            sys.exit("Mismatch between the number of degrees of freedom expected ("+str(dims[0])+") and number of coordinates given ("+str(len(self.rawRIC))+").")
        for i in range(len(self.rawRIC)):
            if i < int(dims[1]):
                self.lengths.append(self.rawRIC[i])
            elif i < int(dims[1]) + int(dims[2]):
                self.angles.append(self.rawRIC[i])
            elif i < int(dims[1]) + int(dims[2]) + int(dims[3]):
                self.diheds.append(self.rawRIC[i])
            else:
                sys.exit("Mismatch between RIC dimensions specified aside fromt the total.")
            


    def getAtoms(self)->list:
        if self.atoms:
            return self.atoms
        else:
            pass
    
    
    def getEnergy(self)->float:
        if self.energy:
            return self.energy
        else:
            pass




class Atom:
    def __init__(self, element:str, coords:list) -> None:
        self.element = element
        self.coords = coords

    

class UnitConverter:

    def __init__(self) -> None:
        pass