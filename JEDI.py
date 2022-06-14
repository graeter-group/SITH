import os
from pathlib import Path
import sys
from Geometry import Atom
from Geometry import Geometry
from SITH_Utilities import Extractor

class JEDI:

    #Set 'pathIO' if you would like to give it a specific working directory for I/O

    #@property


    #! Decide if just use one constructor and always pass explicit values, or make overloaded constructor
    #! Change this so that there is a relaxed Energy ePath can be either a singular file 
    def __init__(self, rePath='x0.fchk', dePath='xF.fchk'):
        
        self.rPath = Path(rePath)
        self.defPath = Path(dePath)

        #Check that all files exist and that, if given, the I/O path is a directory
        assert self.rPath.exists(), "Path to relaxed geometry data does not exist."
        assert self.defPath.exists(), "Path to deformed geometry data does not exist."

        if (not self.rPath.exists()) or (not self.defPath.exists()):
            sys.exit("Path given for one or more input files does not exist.")

        self.dDir = self.defPath.is_dir()

        self.extractData()

        # next it runs jedi_directory.py only if --d specifies a directory with the RICS,
        # this seems unnecessary though so I won't fill it out at least for now and will 
        # simply assume no --d cus I got rid of the option anyway, all input is in Cartesian

        # jedi_rims
        #Converts cartesian coordinates into RIModes
        #Completely unnecessary, Gaussian does all of this already

    def extractData(self):
        try:
            with self.rPath.open() as rFile:
                rData = rFile.readlines()
        #except:
        #    print(sys.exc_info()[0])

            self.dData = list()
            if self.dDir:
                dPaths = list(self.defPath.glob('*.fchk'))
            else:
                dPaths = [self.defPath]

            for dp in dPaths:
                with dp.open() as dFile:
                    self.dData.append((dp.name, dFile.readlines()))

            # with self.enePath.open() as eFile:
            #     eData = eFile.readlines()
            #     eSplit = list()
            #     if len(eData) > 1:
            #         for item in eData:
            #             iSplit = item.split()
            #             eSplit.extend(iSplit)
            #         if len(eSplit) == 3:
            #             raise ValueError("Energy file input must be of the form: {Energy Difference} {E Deformed} {E Relaxed}")
            #     else:
            #         eSplit = eData[0].split()


            # with self.hesPath.open() as hFile:
            #     hData = hFile.readlines()
        except:
            print("An exception occurred during the extraction of data from input files.")
            sys.exit(sys.exc_info()[0])

        #! Add in checks to make sure they aren't just empty files
        self.validateFiles()

        rExtractor = Extractor(rData)
        #Create Geometry objects from relaxed and deformed data
        self.rRIC, self.rXYZ, self.hRIC, self.rEnergy = extractor.extract(rData)
        self.rRIC = Geometry(self.rPath.name, rData)
        self.deformed = list()
        for dd in self.dData:
            self.deformed.append(Geometry(dd[0], dd[1]))
        
        if any([d.nAtoms() != self.rRIC.nAtoms() for d in self.deformed]):
            sys.exit("Inconsistency in number of atoms of input geometries.")
        self.nCarts = 3 * self.rRIC.nAtoms()

        #Populate energy fields from eData
        #! Double check this is correct
        #! Sanna says the energy file is optional?  Ask about this because it's not clear in documentation
        self.eDiff = eSplit[0]
        self.dEnergy = eSplit[1]
        self.rEnergy = eSplit[2]

        #Populate Hessian, Hessian data must already be in the correct form
        #! Need to do properly so it is a good np.ndarray
        self.hMat = hData
        
            

    def validateFiles(self):
        pass

