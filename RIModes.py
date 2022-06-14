import os
from collections import Counter
import numpy as np
import pandas as pd

class RIModes():

    # Used to convert the Cartesian input coordinates into RIModes and then
    # contains the RIM values

    def __init__(self) -> None:
        self.lengths = pd.DataFrame(columns=('BL_1', 'BL_2')) # dataframe containing bondlengths
        self.angles = pd.DataFrame(columns=('BA_1', 'BA_2', 'BA_3')) # bondangles
        self.torsionables = pd.DataFrame(columns=('TA_Atom_1', 'TA_Atom_2')) # all bonds with torsion angles
        self.dihedrals = pd.DataFrame(columns=('DA_1', 'DA_2', 'DA_3', 'DA_4')) # torsion angles

        self.elementsDict = pd.DataFrame(columns=('radius', 'element'))
        self.pullElements()

        #extract cartesian coordinates into list -> dataframe


    def pullElements(self):
        with open("jedi_elements-library.txt", "r") as lib_elements: 
            row = 0
            for element_line in lib_elements: 
                element_line = element_line.split()
                
                self.elementsDict.loc[row] = [element_line[0], element_line[2]]
                row += 1
            
    def convert(self):
        pass

    def vector_length(x_atom1, y_atom1, z_atom1, x_atom2, y_atom2, z_atom2): # function to calculate bondlength
        vec = [0, 0, 0]
        vec[0] = float(x_atom2) - float(x_atom1)
        vec[1] = float(y_atom2) - float(y_atom1)
        vec[2] = float(z_atom2) - float(z_atom1)
    
        len_vec = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    
        return len_vec

