import os
import pathlib
import sys
import numpy as np
from pathlib import Path

from SITH_Utilities import Extractor, Geometry


class JEDI:

    # Set 'pathIO' if you would like to give it a specific working directory for I/O

    # @property

    #! Decide if just use one constructor and always pass explicit values, or make overloaded constructor
    #! Change this so that there is a relaxed Energy ePath can be either a singular file
    def __init__(self, rePath='/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', dePath='/hits/fast/mbm/farrugma/sw/SITH/tests/xF.fchk'):

        self.rPath = Path(rePath)
        self.defPath = Path(dePath)

        self.validateFiles()

        # region global variable initialization

        #self.q0 = np.ndarray()
        #self.qF = np.ndarray()
        #self.delta_q = np.ndarray()
        #self.energies = np.ndarray()
        #self.pEnergies = np.ndarray()
        # vertical vector bc each row is a different deformed geometry
        #self.deformationEnergy = np.ndarray()

        # endregion

        self.extractData()


    def extractData(self):
        
        self.getContents()

        rExtractor = Extractor(self.rPath, self.rData)
        rExtractor.extract()
        # Create Geometry objects from relaxed and deformed data
        self.relaxed = rExtractor.getGeometry()
        self.deformed = list()
        for dd in self.dData:
            dExtractor = Extractor(dd[0], dd[1])
            dExtractor.extract()
            self.deformed.append(dExtractor.getGeometry())

        # might get rid of if decide for the geometries to hold their respective hessians only, but this seems best for now
        self.hMat = self.relaxed.hessian

        self.validateGeometries()

        self.killAtoms()

        # normally the b matrix calculation
        # delta q calculation, so ran jedi_delta_q and pulled delta_q

        self.populateQ()

    def killAtoms(self):
        """
        Implement this later first get methanol test working. Should be as simple as just deleting the correct rows
        """
        pass

    def validateFiles(self):
        """
        Check that all files exist, are not empty, and whether the deformed path is a directory
        """
        assert self.rPath.exists(), "Path to relaxed geometry data does not exist."
        assert self.defPath.exists(), "Path to deformed geometry data does not exist."

        if (not self.rPath.exists()) or (not self.defPath.exists()):
            sys.exit("Path given for one or more input files does not exist.")

        self.dDir = self.defPath.is_dir()


    def getContents(self):
        """
        Gets the contents of the input files, including those in a deformed directory and verifies they are not empty.
        """
        try:
            with self.rPath.open() as rFile:
                self.rData = rFile.readlines()
                assert len(self.rData) > 0, "Relaxed data file is empty."

            self.dData = list()
            if self.dDir:
                dPaths = list(self.defPath.glob('*.fchk'))
                dPaths = [pathlib.Path(dp) for dp in dPaths]
            else:
                dPaths = [self.defPath]

            assert len(dPaths) > 0, "Deformed file directory is empty."
            for dp in dPaths:
                with dp.open() as dFile:
                    dLines = dFile.readlines()
                    assert len(dLines) > 0, "One or more deformed files are empty."
                    self.dData.append((dp, dLines))

        except:
            print("An exception occurred during the extraction of the input files' contents.")
            sys.exit(sys.exc_info()[0])

    def validateGeometries(self):
        """
        Ensure that the relaxed and deformed geometries are compatible(# atoms, # dofs, etc.)
        """
        #if any([d.nAtoms() != self.relaxed.nAtoms() for d in self.deformed]):
        #    sys.exit("Inconsistency in number of atoms of input geometries.")
        #self.nCarts = 3 * self.relaxed.nAtoms()
        pass

    def populateQ(self):
        self.q0 = np.hstack((self.relaxed.lengths,
                            self.relaxed.angles, self.relaxed.diheds))
        # qF rows correspond to each deformed geometry, the columns correspond to the degrees of freedom
        # qF[row or deformed geometry index][d.o.f. index] -> value of d.o.f. for deformed geometry at index of self.deformed
        self.qF = np.hstack((self.deformed[0].lengths,
                                       self.deformed[0].angles, self.deformed[0].diheds))
        if len(self.deformed) > 1:
            for i in range(1, len(self.deformed)):
                deformation = self.deformed[i]
                temp = np.hstack((deformation.lengths,
                                       deformation.angles, deformation.diheds))
                self.qF = np.vstack((self.qF, temp))
            # delta_q is organized in the same shape as qF
        self.delta_q = self.qF - self.q0

    def energyAnalysis(self):
        self.energies = np.ndarray((len(self.deformed), self.relaxed.dims[0]))
        if len(self.deformed) == 1:
            self.deformationEnergy = 0.5 * np.transpose(self.delta_q).dot(
                    self.hMat).dot(self.delta_q)  # scalar 1x1 total Energy
            for j in range(len(self.delta_q)):
                isolatedDOF = np.hstack((np.zeros(j), self.delta_q[j], np.zeros(
                    len(self.delta_q)-j-1)))  # tricky, need to troubleshoot
                self.energies[0,j] = 0.5 * \
                    np.transpose(isolatedDOF).dot(self.hMat).dot(isolatedDOF)

            self.pEnergies = float(
                100) * self.energies / self.deformationEnergy
        else:
            for i in range(len(self.deformed)):
                self.deformationEnergy[i] = 0.5 * np.transpose(self.delta_q[i,:]).dot(
                    self.hMat).dot(self.delta_q[i,:])  # scalar 1x1 total Energy
                for j in range(len(self.delta_q[i])):
                    isolatedDOF = np.hstack((np.zeros(j), self.delta_q[i, j], np.zeros(
                        len(self.delta_q[i])-j-1)))  # tricky, need to troubleshoot
                    self.energies[i, j] = 0.5 * \
                        np.transpose(isolatedDOF).dot(self.hMat).dot(isolatedDOF)
    
                self.pEnergies[i] = float(
                    100) * self.energies[i] / self.deformationEnergy[i]
