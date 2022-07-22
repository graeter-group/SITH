from nis import match
import sys
import numpy as np

class Geometry:

    def __init__(self, name:str, nAtoms:int) -> None:
        self.name = name
        self.rawRIC = list()
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


#TODO: get rid of once verified as vestigial code
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
