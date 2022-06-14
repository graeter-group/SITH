import numpy as np

class Geometry:

    def __init__(self, name:str, lines:list) -> None:
        self.name = name

        self.buildFromLines(lines)

    def buildFromLines(self, lines:list):
        self.atoms = list()
        first = lines[0]
        if len(first.split()) == 1:
            self.nAtoms = int(first)
        for line in lines:
            sLine = line.split()
            if len(sLine) > 1 and len(sLine) < 5:
                a = Atom(sLine[0], sLine[1:4])
                self.atoms.append(a)
            elif len(sLine) >= 5:
                pass

    def nAtoms(self):
        return len(self.atoms)








class Atom:
    def __init__(self, element:str, coords:list) -> None:
        self.element = element
        self.coords = coords
