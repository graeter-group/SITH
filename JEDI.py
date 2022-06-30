import os
import sys
import numpy as np
from pathlib import Path

from SITH_Utilities import Extractor, Geometry


class JEDI:

    # Set 'pathIO' if you would like to give it a specific working directory for I/O

    # @property

    #! Decide if just use one constructor and always pass explicit values, or make overloaded constructor
    #! Change this so that there is a relaxed Energy ePath can be either a singular file
    def __init__(self, rePath='x0.fchk', dePath='xF.fchk'):

        self.rPath = Path(rePath)
        self.defPath = Path(dePath)

        # Check that all files exist and that, if given, the I/O path is a directory
        assert self.rPath.exists(), "Path to relaxed geometry data does not exist."
        assert self.defPath.exists(), "Path to deformed geometry data does not exist."

        if (not self.rPath.exists()) or (not self.defPath.exists()):
            sys.exit("Path given for one or more input files does not exist.")

        self.dDir = self.defPath.is_dir()

        #region global variable initialization

        self.q0 = np.ndarray()
        self.qF = np.ndarray()
        self.delta_q = np.ndarray()
        self.energies = np.ndarray()
        self.pEnergies = np.ndarray()
        self.deformedTotalEnergies = np.ndarray() #vertical vector bc each row is a different deformed geometry

        #endregion

        self.extractData()

        # next it runs jedi_directory.py only if --d specifies a directory with the RICS,
        # this seems unnecessary though so I won't fill it out at least for now and will
        # simply assume no --d cus I got rid of the option anyway, all input is in Cartesian

        # jedi_rims
        # Converts cartesian coordinates into RIModes
        # Completely unnecessary, Gaussian does all of this already

    def extractData(self):
        try:
            with self.rPath.open() as rFile:
                rData = rFile.readlines()
        # except:
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

        rExtractor = Extractor(self.rPath, rData)
        rExtractor.extract()
        # Create Geometry objects from relaxed and deformed data
        #self.rRIC, self.rXYZ, self.hRIC, self.rEnergy = rExtractor.extract(rData)
        self.relaxed = rExtractor.getGeometry()
        self.deformed = list()
        for dd in self.dData:
            self.deformed.append(Geometry(dd[0], dd[1]))

        if any([d.nAtoms() != self.rRIC.nAtoms() for d in self.deformed]):
            sys.exit("Inconsistency in number of atoms of input geometries.")
        self.nCarts = 3 * self.rRIC.nAtoms()
        # might get rid of if decide for the geometries to hold their respective hessians only, but this seems best for now
        self.hMat = self.relaxed.hessian

        self.validateGeometries()

        self.killAtoms()

        # normally the b matrix calculation
        # delta q calculation, so ran jedi_delta_q and pulled delta_q


        self.populateQ()



    def killAtoms(self):
        """
        Implement this later first get methanol test working
        """
        pass

    def validateFiles(self):
        pass

    def validateGeometries(self):
        """
        Ensure that the relaxed and deformed geometries are compatible(# atoms, # dofs, etc.)
        """
        pass

    def populateQ(self):
        self.q0 = np.hstack(self.relaxed.lengths, self.relaxed.angles, self.relaxed.diheds)
        
        for i in range(len(self.deformed)):
            deformation = self.deformed[i]
            #qF rows correspond to each deformed geometry, the columns correspond to the degrees of freedom 
            # qF[row or deformed geometry index][d.o.f. index] -> value of d.o.f. for deformed geometry at index of self.deformed
            self.qF[i] = np.hstack(deformation.lengths, deformation.angles, deformation.diheds)
            #delta_q is organized in the same shape as qF
            self.delta_q = self.qF - self.q0

    def energyAnalysis(self):
        for i in range(len(self.deformed)):
            self.deformedTotalEnergies[i] = 0.5 * np.transpose(self.delta_q[i]).dot(self.hMat).dot(self.delta_q[i]) #scalar 1x1 total Energy
            for j in range(len(self.delta_q[i])):
                isolatedDOF = np.hstack(np.zeros(j), self.delta_q[i, j], np.zeros(len(self.delta_q[i])-j-1)) #tricky, need to troubleshoot
                self.energies[i, j] = 0.5 * np.transpose(isolatedDOF).dot(self.hMat).dot(isolatedDOF)

            self.pEnergies[i] = float(100) * self.energies[i] / self.deformedTotalEnergies[i]
            

