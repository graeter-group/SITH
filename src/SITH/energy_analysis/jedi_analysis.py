import numpy as np


class JediAnalysis:
    def __init__(self, structures_info):
        self.structures_info = structures_info
        self.structures_info.delta_q = self.get_jedi_dq()

    def jedi_analysis(self):
        """Performs the SITH energy analysis, populates energies, and
        energies_percentage.

        Notes
        -----
        Consists of the dot multiplication of the structure vectors and the
        Hessian matrix (analytical gradient of the harmonic potential energy
        surface) to produce both the total calculated change in energy between
        the reference structure and each structure (SITH.structure_energy) as
        well as the subdivision of that energy into each DOF
        (SITH.dof_energies).
        """
        # TODO: replace print by logging
        # print("Performing energy analysis...")
        if self.structures_info.all_hessians[0] is None:
            raise Exception(
                "The Hessian matrix of the reference structure was not " +
                "properly read from the input files")
        self.structures_info.dofs_energy = 0.5 * \
            np.matmul(self.structures_info.delta_q,
                      self.structures_info.all_hessians[self.structures_info.reference]) * \
            self.structures_info.delta_q
        self.structures_info.structure_energy = np.sum(self.structures_info.dofs_energy,
                                                  axis=1)
        # TODO: replace print by logging 
        # print("Execute Order 67. Successful energy analysis completed.")

        return self.structures_info.structure_energy, \
            self.structures_info.dofs_energy
            
    
    def get_jedi_dq(self) -> np.ndarray:
        """Populates delta_q taking the changes respect to the reference
        structure"""
        # TODO: replace print by logging
        #print("Populating DOF vectors and calculating \u0394q...")

        delta_dofs = self.structures_info.all_dofs - self.structures_info.all_dofs[0]

        # This adjustment is to account for cases where dihedral angles
        # oscillate around pi and -pi.
        condition = delta_dofs[:, self.structures_info.dims[1]:] > np.pi
        delta_dofs[:, self.structures_info.dims[1]:][condition] -= 2 * np.pi
        condition = delta_dofs[:, self.structures_info.dims[1]:] < -np.pi
        delta_dofs[:, self.structures_info.dims[1]:][condition] += 2 * np.pi

        self.structures_info.delta_q = delta_dofs

        return delta_dofs
