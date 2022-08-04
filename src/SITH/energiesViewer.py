from ase import Atoms
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import HBox, Output
from src.SITH.visualizator import MoleculeViewer
import numpy as np


class VisualizeEnergies(MoleculeViewer):
    def __init__(self, sith_object, idef=0, **kwargs):
        """
        Set of tools to show a molecule and the
        distribution of energies in the different DOF.

        Params
        ======

        sith_object :
            sith object
        idef: int
            number of the deformation to be analized. Default=0
        """
        self.idef = idef
        self.sith = sith_object
        if self.sith.energies is None:
            self._analize_energies(**kwargs)

        # CHANGE: this could be imported directly from
        # sith as an ase.Atoms object or, at least, coordinates
        # must be float from sith.
        molecule = ''.join([atom.element for atom in
                            self.sith._reference.atoms])
        positions = [[float(component) for component in atom.coords]
                     for atom in self.sith._reference.atoms]
        atoms = Atoms(molecule, positions)

        MoleculeViewer.__init__(self, atoms)

        dims = self.sith._reference.dims
        self.nbonds = dims[1]
        self.nangles = dims[2]
        self.ndihedral = dims[3]

    def _analize_energies(self, dofs=[]):
        """
        Execute JEDI method to obtain the energies of the
        DOFs

        see:
        Parameters
        ==========

        dofs : list of tuples
            Degrees of freedom to be removed from the analysis.
        """
        if self.sith.energies is None:
            self.sith.setKillDOFs([])
            self.sith.extractData()
            self.sith.energyAnalysis()

    def add_dof(self, dof, color=[0.5, 0.5, 0.5], n=5, radius=0.07):
        """
        Add the degree of freedom to the molecule image

        Parameters
        ==========

        dof: tuple
            label of the degree of freedom according with g09 convention.

        Example
        =======
            i=(1, 2) means a bond between atoms 1 and 2
            i=(1, 2, 3) means an angle between atoms 1, 2 and 3
            i=(1, 2, 3, 4) means a dihedral angle between atoms 1, 2 and 3
        """

        types = ["bond", "angle", "dihedral"]
        type_dof = types[len(dof)-2]

        if type_dof == "bond":
            index1, index2 = dof
            return self.add_bond(index1, index2, color, radius=radius)

        elif type_dof == "angle":
            index1, index2, index3 = dof
            return self.add_angle(index1, index2, index3, color, n=n)

        elif type_dof == "dihedral":
            index1, index2, index3, index4 = dof
            return self.add_dihedral(index1, index2, index3,
                                     index4, color, n=n)
        else:
            raise TypeError(f"{dof} is not an accepted degree of freedom.")

    def energies_bonds(self, **kwargs):
        """
        Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith._reference.dimIndices[:self.nbonds]
        self.energies_some_dof(dofs, **kwargs)

    def energies_angles(self, **kwargs):
        """
        Add the angles with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith._reference.dimIndices[self.nbonds:self.nbonds +
                                               self.nangles]
        self.energies_some_dof(dofs, **kwargs)

    def energies_dihedrals(self, **kwargs):
        """
        Add the dihedral angles with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith._reference.dimIndices[self.nbonds+self.nangles:]
        self.energies_some_dof(dofs, **kwargs)

    def energies_all_dof(self, **kwargs):
        """
        Add all DOF with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith._reference.dimIndices
        self.energies_some_dof(dofs, **kwargs)

    def energies_some_dof(self, dofs, cmap=mpl.cm.get_cmap("Blues"),
                          label="Energy [a.u]", labelsize=20,
                          orientation="vertical", div=5, deci=2,
                          width="700px", height="500px", **kwargs):
        """
        Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.

        cmap: cmap. Default: mpl.cm.get_cmap("Blues")
            cmap used in the color bar.

        label: str. Default: "Energy [a.u]"
            label of the color bar.

        labelsize: float.
            size of the label in the

        orientation: "vertical" or "horizontal". Default: "vertical"
            orientation of the color bar.

        div: int. Default: 5
            number of colors in the colorbar.
        """
        energies = []
        for dof in dofs:
            for index, sithdof in enumerate(self.sith._reference.dimIndices):
                if dof == sithdof:
                    energies.append(self.sith.energies[index][self.idef])

        assert len(dofs) == len(energies), "The number of DOFs " + \
            f"({len(dofs)}) does not correspond with the number of " + \
            f"energies ({len(energies)})"

        minval = min(energies)
        maxval = max(energies)

        if orientation == 'v' or orientation == 'vertical':
            rotation = 0
        else:
            rotation = 90

        boundaries = np.linspace(minval, maxval, div+1)
        normalize = mpl.colors.BoundaryNorm(boundaries, cmap.N)

        self.fig, self.ax = plt.subplots(figsize=(0.5, 8))

        # Costumize cbar
        cbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=normalize,
                                                       cmap=cmap),
                                 cax=self.ax, orientation='vertical',
                                 format='%1.{}f'.format(deci), )
        cbar.set_label(label=label, fontsize=labelsize)
        cbar.ax.tick_params(labelsize=0.8*labelsize, rotation=rotation)

        # Insert colorbar in view
        self.viewer.view._remote_call("setSize", targe="Widget",
                                      args=[width, height])
        for i, dof in enumerate(dofs):
            color = cmap(normalize(energies[i]))[:3]
            self.add_dof(dof, color=color, **kwargs)

        self.viewer.view._remote_call("setSize",
                                      targe="Widget",
                                      args=[width, height])
        out = Output()
        with out:
            plt.show()
        self.box = HBox(children=[self.viewer.view, out])

    def show_dof(self, dofs, **kwargs):
        """
        Show specific degrees of freedom.

        Params
        ======

        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        for dof in dofs:
            self.add_dof(dof, **kwargs)

    def show_bonds(self, **kwargs):
        """
        Show the bonds in the molecule of freedom.

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        dofs = self.sith._reference.dimIndices[:self.nbonds]
        self.show_dof(dofs, **kwargs)

    def show(self):
        """
        Show the molecule.
        """
        return self.box
