from operator import contains, indexOf
import pathlib
import sys
from typing import Tuple
import numpy as np
from pathlib import Path

from SITH_Utilities import Extractor


class SITH:

    #! Decide if just use one constructor and always pass explicit values, or make overloaded constructor
    #! Change this so that there is a relaxed Energy ePath can be either a singular file
    def __init__(self, rePath='/hits/fast/mbm/farrugma/sw/SITH/tests/x0.fchk', dePath='/hits/fast/mbm/farrugma/sw/SITH/tests/xF.fchk'):

        self.rPath = Path(rePath)
        self.defPath = Path(dePath)

        self._kill = False
        self._killAtoms = list()
        self._killDOFs = list()

        self.validateFiles()

        # self.extractData()

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

        if self._kill:
            self.kill()

        self.validateGeometries()

        self.populateQ()

    def setKillAtoms(self, atoms: list):
        self._killAtoms = atoms
        self._kill = True

    def setKillDOFs(self, dofs: list):
        self._killDOFs = dofs
        self._kill = True

    def kill(self):
        self.killDOFs(self._killDOFs)
        self.killAtoms(self._killAtoms)

    def killAtoms(self, atoms: list):
        """
        Removes the indicated atoms from the JEDI analysis, as such it removes any associated degrees of freedom from
        the geometries' RICs as well as from the Hessian matrix.
        """
        for atomIndex in atoms:
            self.killAtom(atomIndex)

    def killAtom(self, atom: int):
        """
        Removes the indicated atoms from the JEDI analysis, as such it removes any associated degrees of freedom from
        the geometries' RICs as well as from the Hessian matrix.
        """
        dimsToKill = [dim for dim in self.relaxed.dimIndices if atom in dim]
        dimIndicesToKill = [index for index in range(
            self.relaxed.dims[0]) if atom in self.relaxed.dimIndices[index]]
        self.killDOFs(dimsToKill)


    def killDOFs(self, dofs: list[Tuple]):
        """
        Removes the indicated degrees of freedom from the JEDI analysis, as such it removes them from the geometries' RICs
        as well as from the Hessian matrix.
        """
        # blah = list()
        # for dof in dofs:
        #     for i in range(self.relaxed.dims[0]):
        #         if self.relaxed.dimIndices[i] == dof:
        #             blah.append(i)
        rIndices = list()
        dIndices = list()
        for dof in dofs:
            rIndices.extend([i for i in range(self.relaxed.dims[0]) if self.relaxed.dimIndices[i] == dof])
            dIndices.extend([i for i in range(self.deformed[0].dims[0]) if self.deformed[0].dimIndices[i] == dof])
        self.relaxed.killDOFs(rIndices)
        for deformation in self.deformed:
            deformation.killDOFs(dIndices)

    # def killDOFs(self, dofs: list[Tuple]):
    #     """
    #     Removes the indicated degrees of freedom from the JEDI analysis, as such it removes them from the geometries' RICs
    #     as well as from the Hessian matrix.
    #     """

    #     indices = [[indexOf(self.relaxed.dimIndices[i], dof) for i in range(self.relaxed.dims[0])] for dof in dofs]
    #     self.killDOFs(indices)

    def validateFiles(self):
        """
        Check that all files exist, are not empty, and whether the deformed path is a directory
        """
        assert self.rPath.exists(), "Path to relaxed geometry data does not exist."
        assert self.defPath.exists(), "Path to deformed geometry data does not exist."

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
                dPaths = list(sorted(self.defPath.glob('*.fchk')))
                dPaths = [pathlib.Path(dp) for dp in dPaths]
            else:
                dPaths = [self.defPath]

            assert len(dPaths) > 0, "Deformed directory is empty."
            for dp in dPaths:
                with dp.open() as dFile:
                    dLines = dFile.readlines()
                    assert len(
                        dLines) > 0, "One or more deformed files are empty."
                    self.dData.append((dp, dLines))

        except:
            # This exception catch can be made more specific if necessary, but it really shouldn't be needed
            print(
                "An exception occurred during the extraction of the input files' contents.")
            sys.exit(sys.exc_info()[0])

    def validateGeometries(self):
        """
        Ensure that the relaxed and deformed geometries are compatible(# atoms, # dofs, etc.)
        """
        assert all([deformn.numAtoms() == self.relaxed.numAtoms() and all([deformn.dims[i] == self.relaxed.dims[i] for i in range(
            4)]) for deformn in self.deformed]), "Incompatible number of atoms or dimensions amongst input files."

    def populateQ(self):
        self.q0 = np.zeros((self.relaxed.dims[0], 1))
        self.qF = np.zeros((self.relaxed.dims[0], 1))
        self.q0[:, 0] = np.transpose(np.asarray(self.relaxed.ric))
        # qF columns correspond to each deformed geometry, the rows correspond to the degrees of freedom
        # qF[d.o.f. index, row or deformed geometry index] -> value of d.o.f. for deformed geometry at index of self.deformed
        self.qF[:, 0] = np.transpose(np.asarray(self.deformed[0].ric))
        if len(self.deformed) > 1:
            for i in range(1, len(self.deformed)):
                deformation = self.deformed[i]
                temp = np.transpose(np.asarray(deformation.ric))
                self.qF = np.column_stack((self.qF, temp))
            # delta_q is organized in the same shape as qF
        self.delta_q = np.subtract(self.qF, self.q0)

        """This adjustment is to account for cases where dihedral angles oscillate about 180 degrees or pi and, since the 
        coordinate system in Gaussian for example is from pi to -pi, it shows up as -(pi-k) - (pi - l) = -2pi + k + l
        instead of what it should be: k + l"""
        # TODO: make this ore definitive because collagen use case phi psi angles often around pi regime, perhaps just convert domain of radians from (-pi, pi) -(+pi)-> (0, 2pi) when taking in coordinates initially?
        with np.nditer(self.delta_q, op_flags=['readwrite']) as dqit:
            for dq in dqit:
                dq[...] = np.abs(dq - 2*np.pi) if dq > (2*np.pi -
                                                        0.005) else (dq + 2*np.pi if dq < -(2*np.pi - 0.005) else dq)

# TODO: reove the necessity of separately handling single deformation vs deformation vector
    def energyAnalysis(self):
        self.energies = np.zeros((self.relaxed.dims[0], len(self.deformed)))
        self.deformationEnergy = np.zeros((1, len(self.deformed)))
        self.pEnergies = np.zeros((self.relaxed.dims[0], len(self.deformed)))
        if len(self.deformed) == 1:
            self.deformationEnergy[0, 0] = 0.5 * np.transpose(self.delta_q).dot(
                self.hMat).dot(self.delta_q)  # scalar 1x1 total Energy
            for j in range(self.relaxed.dims[0]):
                isolatedDOF = np.hstack((np.zeros(j), self.delta_q[j], np.zeros(
                    self.relaxed.dims[0]-j-1)))
                self.energies[j, 0] = 0.5 * \
                    (isolatedDOF).dot(self.hMat).dot(self.delta_q)
            self.pEnergies[:, 0] = float(
                100) * self.energies[:, 0] / self.deformationEnergy[0, 0]
        else:
            for i in range(len(self.deformed)):
                self.deformationEnergy[0, i] = 0.5 * np.transpose(self.delta_q[:, i]).dot(
                    self.hMat).dot(self.delta_q[:, i])  # scalar 1x1 total Energy
                for j in range(self.relaxed.dims[0]):
                    isolatedDOF = np.hstack((np.zeros(j), self.delta_q[j, i], np.zeros(
                        self.relaxed.dims[0]-j-1)))
                    self.energies[j, i] = 0.5 * \
                        (isolatedDOF).dot(self.hMat).dot(isolatedDOF)

                self.pEnergies[:, i] = float(
                    100) * self.energies[:, i] / self.deformationEnergy[0, i]
