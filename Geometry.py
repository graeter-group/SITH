import numpy as np

class Geometry:

    def __init__(self, name:str) -> None:
        self.name = name
        self.lengths = list()
        self.angles = list()
        self.diheds = list()
        self.energy = np.inf
        self.atoms = list()
        self.dims['total'] = self.dims['lengths'] = self.dims['angles'] = self.dims['diheds'] = 0


    #! Need to adapt to new format, make sure to specify and/or convert units
    # def buildCartesian(self, lines:list):
    #     first = lines[0]
    #     if len(first.split()) == 1:
    #         self.nAtoms = int(first)
    #     for line in lines:
    #         sLine = line.split()
    #         if len(sLine) > 1 and len(sLine) < 5:
    #             a = Atom(sLine[0], sLine[1:4])
    #             self.atoms.append(a)
    #         elif len(sLine) >= 5:
    #             pass

    def nAtoms(self):
        return len(self.atoms)


    def buildRIC(self, dims:list, lines:list):
        pass

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
