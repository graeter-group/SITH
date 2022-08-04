from ase import Atoms
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import HBox, Output
from src.SITH.visualizator import MoleculeViewer
import numpy as np
from ase.visualize import view


class MoleculeViewer:
    def __init__(self, atoms):
        """ Set of graphic tools to see the distribution
        of energies in the different degrees of freedom
        (lengths, angles, dihedrals)"""
        self.atoms = atoms
        self.viewer = view(atoms, viewer='ngl')
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.shape = self.viewer.view.shape
        self.box = self.viewer

    def add_bond(self, atom1index, atom2index,
                 color=[0.5, 0.5, 0.5], radius=0.1):
        """ Add a bond between two atoms:
        atom1 and atom2

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms to be connected according with g09
            convention.

        color: list. Default gray([0.5, 0.5, 0.5])
            RGB triplet.

        radius: float. Default 0.1
            Radius of the bond.

        Output
        ======

        Return the bonds int the system
        """

        indexes = [atom1index, atom2index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        self.remove_bond(atom1index, atom2index)
        b = self.shape.add_cylinder(self.atoms[atom1index-1].position,
                                    self.atoms[atom2index-1].position,
                                    color,
                                    radius)

        self.bonds[name] = b

        return self.bonds[name]

    def add_bonds(self, atoms1indexes, atoms2indexes, colors=None, radii=None):
        """ Add a bond between each pair of atoms corresponding to
        two lists of atoms:
        atoms1 and atoms.

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms to be connected according with g09
            convention.
        color: list of color lists. Default all gray([0.5, 0.5, 0.5])
            RGB triplets for each of the bonds. It can be one a triplet
            in case of just one color in all bonds.
        radii: float or list of floats. Default 0.1
            radius of each bond.

        Output
        ======

        Return the bonds int the system
        """

        if colors is None:
            colors = [0.5, 0.5, 0.5]

        if type(colors[0]) is not list:
            colors = [colors for i in range(len(atoms1indexes))]

        if radii is None:
            radii = 0.07

        if type(radii) is not list:
            radii = [radii for i in range(len(atoms1indexes))]

        assert len(atoms1indexes) == len(atoms2indexes), \
            "The number of atoms in both lists must be the same"
        assert len(atoms1indexes) == len(colors), \
            "The number of colors in must be the same as the number of atoms"
        assert len(atoms1indexes) == len(radii), \
            "The number of radii must be the same as the number of atoms"

        for i in range(len(atoms1indexes)):
            self.add_bond(atoms1indexes[i],
                          atoms2indexes[i],
                          colors[i],
                          radii[i])
        return self.bonds

    def remove_bond(self, atom1index, atom2index):
        """ Remove a bond between two atoms:
        atoms1 and atoms2.

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms that are connected. This bond
            will be removed.

        Output
        ======

        Return the bonds int the system
        """
        indexes = [atom1index, atom2index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.bonds.keys():
            self.viewer.view.remove_component(self.bonds[name])
            del self.bonds[name]
            print(f"angle {name} removed")
        return self.bonds

    def remove_bonds(self, atoms1indexes=None, atoms2indexes=None):
        """ remove several bonds in the plot between two list of atoms:
        atoms1 and atoms2.

        Parameters
        ==========

        atom1index (and atom2index): list[int]
            Indexes of the atoms that are connected.

        Note: if atoms2 is None, all bonds with atoms1 will me removed.
        If atoms1 and atoms2 are None, all bonds in the structure are
        removed.
        """

        if (atoms1indexes is None) and (atoms2indexes is None):
            for name in self.bonds.keys():
                self.viewer.view.remove_component(self.bonds[name])
            self.bonds.clear()
            return self.bonds

        elif (atoms1indexes is not None) and (atoms2indexes is None):
            to_remove = []
            for name in self.bonds.keys():
                for index in atoms1indexes:
                    if str(index) in name:
                        self.viewer.view.remove_component(self.bonds[name])
                        to_remove.append(name)
            for name in to_remove:
                del self.bonds[name]
            return self.bonds

        else:
            assert len(atoms1indexes) == len(atoms2indexes), \
                "The number of atoms in both lists must be the same"
            [self.remove_bond(index1, index2)
             for index1, index2 in
             zip(atoms1indexes, atoms2indexes)]
            return self.bonds

    def remove_all_bonds(self):
        """ remove all bonds"""
        return self.remove_bonds()

    def plot_arc(self, vertex, arcdots, color):
        """ Add an arc using triangles.

        Parameters
        ==========

        vertex: array
            center of the arc
        arcdots: list of arrays
            vectors that define the points of the arc. These
            vectors must be defined respect the vertex.

        Output
        ======

        Return the triangles in the angle.
        """

        triangles = []
        for i in range(len(arcdots)-1):
            vertexes = np.hstack((vertex,
                                  vertex + arcdots[i],
                                  vertex + arcdots[i+1]))
            t = self.shape.add_mesh(vertexes, color)
            triangles.append(t)

        return triangles

    def add_angle(self, atom1index, atom2index, atom3index,
                  color=[0.5, 0.5, 0.5], n=0):
        """ Add an angle to between three atoms:
        atom1, atom2 and atom3
        - with the vertex in the atom2

        Parameters
        ==========

        atom1index, atom2index and atom3index: int
            Indexes of the three atoms that defines the angle.
        color: color list. Default all gray([0.5, 0.5, 0.5])
            RGB triplet.
        n: int. Default 10
            number of intermedia points to add in the arc of
            the angle.

        Output
        ======
        Return the angles in the system
        """

        indexes = [atom1index, atom2index, atom3index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)
        self.remove_angle(atom1index, atom2index, atom3index)
        self.angles[name] = []

        vertex = self.atoms[atom2index-1].position
        side1 = self.atoms[atom1index-1].position - vertex
        side2 = self.atoms[atom3index-1].position - vertex
        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1/lenside1
        side2 = 0.7 * lensides * side2/lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.angles[name] = self.plot_arc(vertex, arcdots, color)

        return self.angles[name]

    def intermedia_vectors(self, a, b, n):
        """ Define the intermedia arc dots between two vectors

        Parameters
        ==========

        a, b: arrays
             side vectors of the angles.
        n: int
             number of intermedia dots.

        Output
        ======
        Return the intermedia vectors between two side vectors.
        """

        if n == 0:
            return []
        n += 1
        c = b - a
        lena = np.linalg.norm(a)
        lenb = np.linalg.norm(b)
        lenc = np.linalg.norm(c)
        lend = min(lena, lenb)

        theta_total = np.arccos(np.dot(a, b)/(lena * lenb))
        beta = np.arccos(np.dot(a, c)/(lena * lenc))
        intermedia = []

        for i in range(1, n):
            theta = i * theta_total/n
            gamma = beta - theta
            factor = (lena * np.sin(theta))/(lenc * np.sin(gamma))
            dird = a + factor * c
            d = lend * dird/np.linalg.norm(dird)
            intermedia.append(d)
        return intermedia

    def remove_angle(self, atom1index, atom2index, atom3index):
        """
        Remove an angle if it exists

        Parameters
        ==========

        atom1index (and atom2/3index): int
            Indexes of the three atoms that defines the angle
            to remove.

        Output
        ======
        Return the angles
        """
        indexes = [atom1index, atom2index, atom3index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.angles.keys():
            for triangle in self.angles[name]:
                self.viewer.view.remove_component(triangle)
            del self.angles[name]

        return self.angles

    def remove_all_angles(self):
        """ remove all angles"""
        names = self.angles.keys()

        for name in names:
            for triangle in self.angles[name]:
                self.viewer.view.remove_component(triangle)
        self.angles.clear()

    def add_dihedral(self, atom1index, atom2index, atom3index,
                     atom4index, color=[0.5, 0.5, 0.5], n=0):
        """ Add an dihedral angle between four atoms:
        atom1, atom2, atom3 and atom4
        - with the vertex in the midle of the atom 2 and 3

        Parameters
        ==========

        atom1index, atom2index, atom3index and atom4index: int
            Indexes of the three atoms that defines the angle.
        color: color list. Default all gray([0.5, 0.5, 0.5])
            RGB triplet.
        n: int. Default 10
            number of intermedia points to add in the arc of
            the angle.

        Output
        ======
        Return the dihedral angles
        """
        indexes = [atom1index, atom2index, atom3index, atom4index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        axis = (self.atoms[atom3index-1].position -
                self.atoms[atom2index-1].position)
        vertex = 0.5 * (self.atoms[atom3index-1].position +
                        self.atoms[atom2index-1].position)
        axis1 = (self.atoms[atom1index-1].position -
                 self.atoms[atom2index-1].position)
        axis2 = (self.atoms[atom4index-1].position -
                 self.atoms[atom3index-1].position)

        side1 = axis1 - axis * (np.dot(axis, axis1)/np.dot(axis, axis))
        side2 = axis2 - axis * (np.dot(axis, axis2)/np.dot(axis, axis))

        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1/lenside1
        side2 = 0.7 * lensides * side2/lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.dihedrals[name] = self.plot_arc(vertex, arcdots, color)

        return self.dihedrals[name]

    def remove_dihedral(self, atom1index, atom2index, atom3index, atom4index):
        """ Remove the dihedral angle between 4 atoms:

        atom1, atom2, atom3 and atom4

        Parameters
        ==========

        atom1index, atom2index, atom3index and atom4index: int
            Indexes of the three atoms that defines the angle.

        Output
        ======

        Return the dihedral angles
        """
        indexes = [atom1index, atom2index, atom3index, atom4index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.dihedrals.keys():
            for triangle in self.dihedrals[name]:
                self.viewer.view.remove_component(triangle)
            del self.dihedrals[name]
        return self.dihedrals

    def remove_all_dihedrals(self):
        """ remove all dihedral angles"""
        names = self.dihedrals.keys()

        for name in names:
            for triangle in self.dihedrals[name]:
                self.viewer.view.remove_component(triangle)
        self.dihedrals.clear()

    def download_image(self):
        self.viewer.view.download_image()

    def picked(self):
        return self.viewer.view.picked


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
