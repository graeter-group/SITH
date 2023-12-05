import numpy as np


class SithAnalysis:
    def __init__(self, structures_info):
        self.structures_info = structures_info
        self.structures_info.delta_q = self.get_sith_dq()

    def get_sith_dq(self):
        delta_dofs = self.structures_info.all_dofs - \
            np.insert(self.structures_info.all_dofs[:-1], 0,
                      self.structures_info.all_dofs[0], axis=0)

        condition = delta_dofs[:, self.structures_info.dims[1]:] > np.pi
        delta_dofs[:, self.structures_info.dims[1]:][condition] -= 2 * np.pi
        condition = delta_dofs[:, self.structures_info.dims[1]:] < -np.pi
        delta_dofs[:, self.structures_info.dims[1]:][condition] += 2 * np.pi

        return delta_dofs

    def rectangle_integration(self):
        """
        Numerical integration using rectangle rule algorithm. Method 0 in this
        class (see Sith parameters).

        Return
        ======
        (tuple) [energies, total_ener] energies computed by SITH
        method.
        """
        all_values = - self.structures_info.all_forces * self.structures_info.delta_q
        energies = np.cumsum(all_values, axis=0)
        total_ener = np.sum(energies, axis=1)

        return energies, total_ener

    def trapezoid_integration(self):
        """
        Numerical integration using trapezoid rule algorithm. Method 1 in this
        class (see Sith parameters).

        Return
        ======
        (tuple) [energies, total_ener] energies computed by SITH
        method.
        """
        # energy for the optimized config must be the reference

        added_forces = (self.structures_info.all_forces[1:] +
                        self.structures_info.all_forces[:-1]) / 2
        all_values = added_forces * self.structures_info.delta_q[1:]
        all_values = np.insert(all_values, 0, np.zeros(self.structures_info.dims[0]),
                               axis=0)
        energies = -np.cumsum(all_values, axis=0)
        total_ener = np.sum(energies, axis=1)

        return energies, total_ener

    def simpson_integration(self):
        """
        Numerical integration using simpson algorithm. Method 2 in this class
        (see Sith parameters).

        Return
        ======
        (tuple) [energies, total_ener] energies computed by SITH
        method.
        """
        try:
            from scipy.integrate import simpson
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Install scipy to use simpson" +
                                      "integration")
        dofs = self.structures_info.all_dofs.copy().T

        for angle in dofs[self.structures_info.dims[1]:]:
            for i in range(self.structures_info.n_structures - 1):
                while angle[i + 1] - angle[i] > np.pi:
                    angle[i + 1:] -= 2 * np.pi
                while angle[i + 1] - angle[i] < -np.pi:
                    angle[i + 1:] += 2 * np.pi
        dofs = dofs.T

        # first array counts the  energy in the dofs for the optimized
        # configuration. that's why it is zero
        all_ener = np.array([[0] * self.structures_info.dims[0]])
        # next loop is a 'nasty' cummulative integration. Maybe it could
        # be improved
        for i in range(1, self.structures_info.n_structures):
            ener_def = -simpson(self.structures_info.all_forces[: i + 1],
                                x=dofs[: i + 1],
                                axis=0)
            all_ener = np.append(all_ener, np.array([ener_def]), axis=0)
        total_ener = np.sum(all_ener, axis=1)
        return all_ener, total_ener
