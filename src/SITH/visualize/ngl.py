import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import HBox
import numpy as np
from SITH.Utilities import color_distribution, create_colorbar


class MoleculeNGL:
    def __init__(self, atoms, alignment=None, axis=False,
                 xsize: int = 500, ysize: int = 500, n=5):
        """Set of graphic tools to see the distribution
        of energies in the different degrees of freedom
        (lengths, angles, dihedrals)

        alignment: list[int]
            list of three indexes corresponding to the
            indexes of the atoms in the xy plane. the first
            two atoms are set to the x axis.
        """
        if type(atoms) is list:
            self.is_trajectory = True
            self.atoms = [config.copy() for config in atoms]
        else:
            self.is_trajectory = False
            self.atoms = atoms.copy()

        if alignment is not None:
            index1, index2, index3 = alignment
            if type(atoms) is list:
                [self.xy_alignment(config, index1, index2, index3)
                 for config in self.atoms]
            else:
                self.xy_alignment(self.atoms, index1, index2, index3)

        self.viewer = view(self.atoms, viewer='ngl')
        # The keys of the next dictionaries are the names of the DOFs and they
        # are defined such that the name is invariant to the order of the
        # indexes, such that the dof i-j((-k)-l) is the same as (l-(k-))j-i
        self.bonds = {}
        self.n = n
        self.angles = {}
        self.dihedrals = {}
        self.all_dofs_parameters = {}
        self.shape = self.viewer.view.shape
        self.box = self.viewer
        if axis:
            self.add_axis()

    def add_bond(self, atom1index, atom2index,
                 color=None, radius=0.1):
        """Add a bond between two atoms:
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

        Return the bonds in the system
        """
        if self.is_trajectory:
            atoms = self.atoms[self.viewer.view.frame]
        else:
            atoms = self.atoms
        if color is None:
            color = [0.5, 0.5, 0.5]

        indexes = [atom1index, atom2index]
        if atom1index > atom2index:
            indexes = indexes[::-1]
        name = ''.join(str(i).zfill(3) for i in indexes)
        self.remove_bond(atom1index, atom2index)

        b = self.shape.add_cylinder(atoms[atom1index - 1].position,
                                    atoms[atom2index - 1].position,
                                    color,
                                    radius)

        self.bonds[name] = b
        self.all_dofs_parameters[name] = np.array([atom1index, atom2index,
                                                   0, 0])

        return self.bonds[name]

    def add_bonds(self, atoms1indexes, atoms2indexes, colors=None, radii=None):
        """Add a bond between each pair of atoms corresponding to
        two lists of atoms:
        atoms1 and atoms.

        Parameters
        ==========
        atom1index (and atom2index): int
            Indexes of the atoms to be connected according with g09
            convention.
        colors: list of color lists. Default all gray([0.5, 0.5, 0.5])
            RGB triplets for each of the bonds. It can be one a triplet
            in case of just one color in all bonds.
        radii: float or list of floats. Default 0.1
            radius of each bond.

        Output
        ======

        Return the bonds in the system
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
        """Remove a bond between two atoms:
        atoms1 and atoms2.

        Parameters
        ==========

        atom1index (and atom2index): int
            Indexes of the atoms that are connected. This bond
            will be removed.

        Output
        ======

        Return the bonds in the system
        """
        indexes = [atom1index, atom2index]
        indexes.sort()
        name = ''.join(str(i).zfill(3) for i in indexes)

        if name in self.bonds.keys():
            self.viewer.view.remove_component(self.bonds[name])
            del self.bonds[name]
            del self.all_dofs_parameters[name]
        return self.bonds

    def remove_bonds(self, atoms1indexes=None, atoms2indexes=None):
        """Remove several bonds in the plot between two list of atoms:
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
                del self.all_dofs_parameters[name]
            return self.bonds

        else:
            assert len(atoms1indexes) == len(atoms2indexes), \
                "The number of atoms in both lists must be the same"
            [self.remove_bond(index1, index2)
             for index1, index2 in
             zip(atoms1indexes, atoms2indexes)]
            return self.bonds

    def remove_all_bonds(self):
        """Remove all bonds"""
        return self.remove_bonds()

    def add_arc(self, vertex, arcdots, color):
        """Add an arc using triangles.

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
        for i in range(len(arcdots) - 1):
            vertexes = np.hstack((vertex,
                                  vertex + arcdots[i],
                                  vertex + arcdots[i + 1]))
            t = self.shape.add_mesh(vertexes, color)
            triangles.append(t)

        return triangles

    def add_angle(self, atom1index, atom2index, atom3index,
                  color=None, n=None):
        """Add an angle to between three atoms:
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
        if n is None:
            n = self.n
        if color is None:
            color = [0.5, 0.5, 0.5]

        if self.is_trajectory:
            atoms = self.atoms[self.viewer.view.frame]
        else:
            atoms = self.atoms

        indexes = [atom1index, atom2index, atom3index]
        if atom1index > atom3index:
            indexes = indexes[::-1]
        name = ''.join(str(i).zfill(3) for i in indexes)
        self.remove_angle(atom1index, atom2index, atom3index)
        self.angles[name] = []

        vertex = atoms[atom2index - 1].position
        side1 = atoms[atom1index - 1].position - vertex
        side2 = atoms[atom3index - 1].position - vertex
        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1 / lenside1
        side2 = 0.7 * lensides * side2 / lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.angles[name] = self.add_arc(vertex, arcdots, color)
        self.all_dofs_parameters[name] = np.array([atom1index, atom2index,
                                                   atom3index, 0])

        return self.angles[name]

    def intermedia_vectors(self, a, b, n):
        """Define the intermedia arc dots between two vectors

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

        theta_total = np.arccos(np.dot(a, b) / (lena * lenb))
        beta = np.arccos(np.dot(a, c) / (lena * lenc))
        intermedia = []

        for i in range(1, n):
            theta = i * theta_total / n
            gamma = beta - theta
            factor = (lena * np.sin(theta)) / (lenc * np.sin(gamma))
            dird = a + factor * c
            d = lend * dird / np.linalg.norm(dird)
            intermedia.append(d)
        return intermedia

    def remove_angle(self, atom1index, atom2index, atom3index):
        """Remove an angle if it exists

        Parameters
        ==========

        atom1index (and atom2 / 3index): int
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
            del self.all_dofs_parameters[name]

        return self.angles

    def remove_all_angles(self):
        """Remove all angles"""
        names = self.angles.keys()

        for name in names:
            for triangle in self.angles[name]:
                self.viewer.view.remove_component(triangle)
        self.angles.clear()

    def add_dihedral(self, atom1index, atom2index, atom3index,
                     atom4index, color=None, n=None):
        """Add an dihedral angle between four atoms:
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
        if n is None:
            n = self.n

        if self.is_trajectory:
            atoms = self.atoms[self.viewer.view.frame]
        else:
            atoms = self.atoms
        if color is None:
            color = [0.5, 0.5, 0.5]

        indexes = [atom1index, atom2index, atom3index, atom4index]
        if atom1index > atom4index:
            indexes = indexes[::-1]
        name = ''.join(str(i).zfill(3) for i in indexes)

        axis = (atoms[atom3index - 1].position -
                atoms[atom2index - 1].position)
        vertex = 0.5 * (atoms[atom3index - 1].position +
                        atoms[atom2index - 1].position)
        axis1 = (atoms[atom1index - 1].position -
                 atoms[atom2index - 1].position)
        axis2 = (atoms[atom4index - 1].position -
                 atoms[atom3index - 1].position)
        side1 = axis1 - axis * (np.dot(axis, axis1) / np.dot(axis, axis))
        side2 = axis2 - axis * (np.dot(axis, axis2) / np.dot(axis, axis))

        lenside1 = np.linalg.norm(side1)
        lenside2 = np.linalg.norm(side2)
        lensides = min(lenside1, lenside2)
        side1 = 0.7 * lensides * side1 / lenside1
        side2 = 0.7 * lensides * side2 / lenside2

        arcdots = [side1, side2]
        color = color * 3

        new = self.intermedia_vectors(side1,
                                      side2,
                                      n)

        if n != 0:
            [arcdots.insert(1, vert) for vert in new[::-1]]

        self.dihedrals[name] = self.add_arc(vertex, arcdots, color)
        self.all_dofs_parameters[name] = np.array([atom1index, atom2index,
                                                   atom3index, atom4index])

        return self.dihedrals[name]

    def remove_dihedral(self, atom1index, atom2index, atom3index, atom4index):
        """Remove the dihedral angle between 4 atoms:

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
            del self.all_dofs_parameters[name]
        return self.dihedrals

    def remove_all_dihedrals(self):
        """Remove all dihedral angles"""
        names = self.dihedrals.keys()

        for name in names:
            for triangle in self.dihedrals[name]:
                self.viewer.view.remove_component(triangle)
        self.dihedrals.clear()

    def add_dof(self, dof, color=None, n=5, radius=0.07):
        """Add the degree of freedom to the molecule image

        Parameters
        ==========

        dof: tuple
            label of the degree of freedom according with g09 convention.

        Example
        =======
            i=(1, 2) means a bond between atoms 1 and 2
            i=(1, 2, 3) means an angle between atoms 1, 2 and 3
            i=(1, 2, 3, 4) means a dihedral angle between atoms 1, 2, 3 and 4
        """

        if color is None:
            color = [0.5, 0.5, 0.5]

        types = ["bond", "angle", "dihedral"]
        type_dof = types[np.count_nonzero(dof) - 2]

        if type_dof == "bond":
            index1 = dof[0]
            index2 = dof[1]
            return self.add_bond(index1, index2, **kwargs)

        elif type_dof == "angle":
            index1 = dof[0]
            index2 = dof[1]
            index3 = dof[2]
            return self.add_angle(index1, index2, index3, **kwargs)

        elif type_dof == "dihedral":
            index1 = dof[0]
            index2 = dof[1]
            index3 = dof[2]
            index4 = dof[3]
            return self.add_dihedral(index1, index2, index3,
                                     index4, **kwargs)
        else:
            raise TypeError(f"{dof} is not an accepted degree of freedom.")

    def add_axis(self, length=1, radius=0.1):
        """Add xyz axis.

        Parameters
        ==========

        length: float
            indicates the length of the axis in the visualization. Default=1
        radius: float
            thickness of the xyz axis
        """
        self.axis = {}

        unit_vectors = np.array([[length, 0, 0],
                                 [0, length, 0],
                                 [0, 0, length]])
        for i in range(3):
            a = self.shape.add_cylinder([0, 0, 0],
                                        unit_vectors[i],
                                        unit_vectors[i] / length,
                                        radius)
            self.axis[str(i)] = a

    def remove_axis(self):
        """
        remove xyz axis
        """
        for name in self.axis.keys():
            self.viewer.view.remove_component(self.axis[name])
        self.axis.clear()

    def download_image(self, *args, **kwargs):
        self.viewer.view.download_image(*args, **kwargs)

    def picked(self):
        return self.viewer.view.picked

    # Alignment

    def rot_x(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        return R

    def rot_y(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, 0, s],
                      [0, 1, 0],
                      [-s, 0, c]])
        return R

    def rot_z(self, angle):
        c = np.cos(angle)
        s = np.sin(angle)
        R = np.array([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])
        return R

    def align_axis(self, vector):
        """Apply the necessary rotations to set a
        vector aligned with positive x axis
        """
        xyproj = vector.copy()
        xyproj[2] = 0
        phi = np.arcsin(vector[2] / np.linalg.norm(vector))
        theta = np.arccos(vector[0] / np.linalg.norm(xyproj))
        if vector[1] < 0:
            theta *= -1
        trans = np.dot(self.rot_y(phi), self.rot_z(-theta))
        return trans

    def align_plane(self, vector):
        """
        Rotation around x axis to set a vector in the xy plane
        """
        reference = vector.copy()
        reference[0] = 0
        angle = np.arccos(reference[1] / np.linalg.norm(reference))
        if reference[2] < 0:
            angle *= -1
        return self.rot_x(-angle)

    def apply_trans(self, atoms, trans, indexes=None):
        """Apply a transformation to all vector positions of the
        atoms object
        """
        if indexes is None:
            indexes = list(range(len(atoms)))

        new_positions = []
        for i, atom in enumerate(atoms):
            if i in indexes:
                new_positions.append(np.dot(trans, atom.position))
            else:
                new_positions.append(atom.position)
        atoms.set_positions(new_positions)

        return new_positions

    def xy_alignment(self, atoms, index1, index2, index3):
        """Transforme the positions of the atoms such that
        the atoms of indexes 1 and 2 are aligned in the
        x axis

        the atom 3 is in the xy plane"""
        # center
        center = (atoms[index1].position + atoms[index2].position) / 2
        atoms.set_positions(atoms.positions - center)
        axis = atoms[index2].position
        self.apply_trans(atoms, self.align_axis(axis))
        third = atoms[index3].position
        self.apply_trans(atoms, self.align_plane(third))

        return atoms.positions

    def show(self):
        """Show the molecule."""
        return self.box


class EnergiesNGL(MoleculeNGL):
    def __init__(self, sith_info, idef='all', alignment=None, axis=False,
                 background='#ffc', **kwargs):
        """Set of tools to show a molecule and the
        distribution of energies in the different DOF.

        Params
        ======

        sith_info :
            sith object or sith.utilities.ReadSummary object
        idef: int
            number of the deformation to be analized. Default=0
        alignment: list
            3 indexes to fix the correspondig atoms in the xy plane.
            The first atom is placed in the negative side of the x axis,
            the second atom is placed in the positive side of the x axis,
            and the third atom is placed in the positive side of the y axis.
        axis: bool
            add xyz axis
        background: color
            background color. Default: '#ffc'
        """
        self.idef = idef
        self.sith = sith_info
        if self.sith.energies is None:
            self._analize_energies(**kwargs)

        if idef == 'all':
            self.idef = None
            atoms = [config.atoms for config in self.sith.structures]
        else:
            self.idef = idef
            atoms = self.sith.structures[self.idef].atoms

        MoleculeViewer.__init__(self, atoms, alignment, axis)

        if self.idef is None:
            self.idef = self.viewer.view.frame

        self.viewer.view.background = background

        dims = self.sith.structures[0].dims
        self.nbonds = dims[1]
        self.nangles = dims[2]
        self.ndihedral = dims[3]
        self.kwargs_edofs = {'cmap': mpl.cm.get_cmap("Blues"),
                             'label': "Energy [Ha]",
                             'labelsize': 20,
                             'orientation': "vertical",
                             'div': 5,
                             'deci': 2,
                             'width': "700px",
                             'height': "600px",
                             'absolute': False}
        self.viewer.view.observe(self.update_frame, names='frame')

    def _analize_energies(self):
        """
        Execute JEDI method to obtain the energies of each
        DOFs

        see: https://doi.org/10.1063/1.4870334
        """
        self.sith.extract_data()
        self.sith.analyze()

    def add_dof(self, dof, color=[0.5, 0.5, 0.5], n=5, radius=0.07):
        """Add the degree of freedom to the molecule image

        Parameters
        ==========

        dof: tuple
            label of the degree of freedom according with g09 convention.

        Example
        =======
            i=(1, 2) means a bond between atoms 1 and 2
            i=(1, 2, 3) means an angle between atoms 1, 2 and 3
            i=(1, 2, 3, 4) means a dihedral angle between atoms 1, 2, 3 and 4
        """

        types = ["bond", "angle", "dihedral"]
        type_dof = types[len(dof)-2]
        type_dof = types[np.count_nonzero(dof) - 2]

        if type_dof == "bond":
            index1 = dof[0]
            index2 = dof[1]
            return self.add_bond(index1, index2, **kwargs)

        elif type_dof == "angle":
            index1 = dof[0]
            index2 = dof[1]
            index3 = dof[2]
            return self.add_angle(index1, index2, index3, **kwargs)

        elif type_dof == "dihedral":
            index1 = dof[0]
            index2 = dof[1]
            index3 = dof[2]
            index4 = dof[3]
            return self.add_dihedral(index1, index2, index3,
                                     index4, **kwargs)
        else:
            raise TypeError(f"{dof} is not an accepted degree of freedom.")

    def energies_bonds(self, **kwargs):
        """Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices[:self.nbonds]
        out = self.energies_some_dof(dofs, **kwargs)
        self.update_frame()
        return out

    def energies_angles(self, **kwargs):
        """Add the angles with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices[self.nbonds:self.nbonds +
                                                   self.nangles]
        out = self.energies_some_dof(dofs, **kwargs)
        self.update_frame()
        return out

    def energies_dihedrals(self, **kwargs):
        """Add the dihedral angles with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices[self.nbonds + self.nangles:]
        out = self.energies_some_dof(dofs, **kwargs)
        self.update_frame()
        return out

    def update_frame(self, change=None):
        """Function that is called when the frame is changed. Set the internal
        atribute self.kwargs_edofs to keep fixed the parameters to update the
        visualization of dofs.
        """
        self.idef = self.viewer.view.frame

        dofs = list(self.all_dofs_parameters.values())
        if len(dofs) != 0:
            self.energies_some_dof(dofs, **self.kwargs_edofs)

    def energies_all_dof(self, **kwargs):
        """Add all DOF with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices
        return self.energies_some_dof(dofs, **kwargs)

    def change_def(self, def_dict: dict, **kwargs) -> tuple:
        """
        This functions change the values stored in a dictionary and removes
        each one of the arguments from the kwargs.

        Parameters
        ==========
        def_dict: dict
            dictionary with the default values.
        **kwargs: all the arguments you want to change.

        Returns
        =======
        (dict, dict) modified dictionary withe the default values and set of
        kwargs without the used keys.
        """
        rem_keys = []
        for key, value in kwargs.items():
            if key in def_dict.keys():
                rem_keys.append(key)
                def_dict[key] = value

        for key in rem_keys:
            del kwargs[key]

        return def_dict, kwargs

    def energies_some_dof(self, dofs, **kwargs):
        """Add the bonds with a color scale that represents the
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
        self.kwargs_edofs, kwargs = self.change_def(self.kwargs_edofs,
                                                    **kwargs)
        cmap = self.kwargs_edofs['cmap']
        label = self.kwargs_edofs['label']
        labelsize = self.kwargs_edofs['labelsize']
        orientation = self.kwargs_edofs['orientation']
        div = self.kwargs_edofs['div']
        deci = self.kwargs_edofs['deci']
        width = self.kwargs_edofs['width']
        height = self.kwargs_edofs['height']
        absolute = self.kwargs_edofs['absolute']

        energies, normalize = color_distribution(self.sith,
                                                 dofs,
                                                 self.idef,
                                                 cmap,
                                                 absolute,
                                                 div)

        for i, dof in enumerate(dofs):
            color = cmap(normalize(energies[self.idef][i]))[:3]
            self.add_dof(dof, color=color, **kwargs)

        # Insert colorbar in view
        self.view._remote_call("setSize",
                               targe="Widget",
                               args=[width, height])
        self.fig, cbarwdg = create_colorbar(normalize, cmap, deci, label,
                                            labelsize, orientation, width,
                                            height)

        self.box = HBox([self.view, cbarwdg])

        return self.fig

    def show_bonds_of_DOF(self, dof, unique=False, color=None):
        """Show an specific dof.

        Parameters
        ==========
        dof: int.
            index in sith object that corresponds to the dof you want to show.
        unique: Bool. default False.
            True if you want to remove all the other bonds and only keeping
            these ones.
        color: list[3(int)]. default R G B for angles, distances, dihedrals.
            color that you want to use in this dof.
        """
        dof_indices = self.sith.structures[0].dim_indices[dof]
        if color is None:
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            color = colors[len(dof_indices) - 2]
        atoms1 = []
        atoms2 = []
        for i in range(len(dof_indices) - 1):
            atoms1.append(dof_indices[i])
            atoms2.append(dof_indices[i + 1])
        if unique:
            self.remove_all_bonds()

        return self.add_bonds(atoms1, atoms2, colors=color)

    def show_dof(self, dofs, **kwargs):
        """Show specific degrees of freedom.

        Parameters
        ==========
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
        """Show the bonds in the molecule.

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        dofs = self.sith.structures[0].dim_indices[:self.nbonds]
        self.show_dof(dofs, **kwargs)
