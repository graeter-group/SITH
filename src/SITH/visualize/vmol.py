from typing import Union
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from vmol.view import VMolecule
import vpython as vp
from SITH.SITH import SITH
from SITH.Utilities import color_distribution, create_colorbar


class EnergiesVMol(VMolecule):
    """
    Attributes
    =========
    atoms: ase.Atoms
        Atoms object with the molecule structure
    ax: plt.Axes
        matplotlib Axes that contains the colorbar.
    b_counter: vpython.button
        button counter that sets the stretching.
    color_scheme: dict
        color scheme of the atoms. Default = jml_colors
    canvaskwargs: dict
        set the values of the attributes in the escene.
    fig: plt.figure
        Matplotlib Figure that contains the colorbar.
    frame: int
        frame of the trajectory that is displayed
    hidden_objs: dict
        vpython does not delete objects, only hide them. This dictionary
        contains the deleted objects.
    idef: int
        index of the stretched structure.
    kwargs_edofs:
        set the parameters of the energies in dofs: cmap, label, labelsize,
        orientation, div, deci, width, height, absolute.
    master: vpython
        vpython that the user can use to add new objects.
    nangles: int
        number of angles.
    nbonds: int
        number of bonds.
    ndihedral: int
        number of dihedrals.
    scene: vpython.canvas
        canvas displaying the molecule.
    sith: SITH.SITH
        sith object with the energy distribution analysis.
    trajectory: list
        set of stretched configurations listed as ase.Atoms objects.
    vatoms: list
        set of spheres that represents the atoms in the displayed scene.
    """
    def __init__(self, sith_info: SITH,
                 idef: int = 0,
                 alignment: Union[list, tuple, np.ndarray] = None,
                 show_axis: bool = False,
                 background: Union[list, vp.color] = vp.color.white,
                 portion: float = 80,
                 **kwargs):
        r"""
        Set of tools to visualize a molecule and the distribution of energies
        in different DOFs.

        Parameters
        ==========
        sith_info: SITH
            sith object with the QM information to perform the analysis.
        idef: int
            initial deformation to be considered as reference. Default=0
        alignment: list
            3 indexes to fix the correspondig atoms in the xy plane.
            The first atom is placed in the negative side of the x axis,
            the second atom is placed in the positive side of the x axis,
            and the third atom is placed in the positive side of the y axis.
        show_axis: bool
            add xyz axis
        background: rgb list or vpython vector
            background color. Default=vpython.color.white
        portion: float
            percentage of the complete figure to be used to add the canvas. The
            rest of the space can be used to add a figure. Default=80
        \*\*kwargs: arguments for VMolecule
        """
        self.canvaskwargs = kwargs
        self.sith = sith_info

        atoms = [config.atoms for config in self.sith.structures]

        if idef < 0:
            assert abs(idef) <= len(atoms)
            self.idef = len(atoms) + idef
        else:
            self.idef = idef

        if 'height' in list(kwargs.keys()):
            height = kwargs['height']
        else:
            height = 500
            kwargs['height'] = height

        VMolecule.__init__(self, atoms,
                           show_axis=show_axis,
                           alignment=alignment,
                           frame=idef,
                           align='left',
                           js='<img src="colorbar.png"' +
                              'style="object-fit:fill;' +
                              f'height:{height}px;"/>',
                           portion=portion,
                           **kwargs)
        self.scene.background = self._asvector(background)
        self.traj_buttons()

        dims = self.sith.dims
        self.nbonds = dims[1]
        self.nangles = dims[2]
        self.ndihedral = dims[3]

        # matplotlib figure for colorbar
        self.fig = None

        self.kwargs_edofs = {'cmap': mpl.cm.get_cmap("Blues"),
                             'label': "Energy [Ha]",
                             'labelsize': 20,
                             'orientation': "vertical",
                             'div': 5,
                             'deci': 2,
                             'width': "700",
                             'height': "600",
                             'absolute': False}
        self.normalize, kwargs = self.create_figure(dofs, **kwargs)

        if 'img' not in kwargs.keys():
            kwargs['img'] = "./tmp_cbar.png"

        # Create scene
        VMolecule.__init__(self, atoms,
                           show_axis=show_axis,
                           alignment=alignment,
                           align='left',
                           frame=idef,
                           portion=portion,
                           **kwargs)

        self.scene.background = self._asvector(background)
        self.traj_buttons()

        # show energies in dofs
        self.energies_some_dof(dofs, **self.kwargs_edofs)

    
    def create_figure(self, dofs: list, **kwargs):
        if 'all' in dofs:
            dofs = self.sith.dim_indices
        else:
            if 'bonds' in dofs:
                bonds = self.sith.dim_indices[:self.nbonds]
                dofs.extend(bonds)
                dofs.remove('bonds')
            elif 'angles' in dofs:
                angles = self.sith.dim_indices[self.nbonds:self.nbonds +
                                                   self.nangles]
                dofs.extend(angles)
                dofs.remove('angles')
            elif 'dihedrals' in dofs:
                dihedrals = self.sith.dim_indices[self.ndihedral:]
                dofs.extend(dihedrals)
                dofs.remove('dihedrals')
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
        
        self.energies, normalize = color_distribution(self.sith,
                                                      dofs,
                                                 self.idef,
                                                 cmap,
                                                 absolute,
                                                 div)

        # Colorbar
        self.fig, _ = create_colorbar(normalize, cmap, deci, label, labelsize,
                                      orientation, int(width), int(height),
                                      dpi=300)

        self.fig.savefig("./tmp_cbar.png", dpi=300)

        return normalize, kwargs

    def energies_bonds(self, **kwargs) -> tuple:
        r"""
        Add the bonds with a color scale that represents the distribution of
        energy according to the JEDI method.

        Parameters
        ==========
        \*\*kwargs for EnergiesVMol.energies_some_dof

        Returns
        =======
        (tuple) DOFs and their computed energies.
        """
        dofs = self.sith.structures[0].dim_indices[:self.nbonds]
        out = self.energies_some_dof(dofs, **kwargs)

        return out

    def energies_angles(self, **kwargs) -> tuple:
        r"""
        Add the angles with a color scale that represents the distribution of
        energy according to the JEDI method.

        Parameters
        ==========
        \*\* kwargs for EnergiesVMol.energies_some_dof

        Returns
        =======
        (tuple) DOFs and their computed energies.
        """
        dofs = self.sith.structures[0].dim_indices[self.nbonds:self.nbonds +
                                                   self.nangles]
        out = self.energies_some_dof(dofs, **kwargs)
        return out

    def energies_dihedrals(self, **kwargs) -> tuple:
        r"""
        Add the dihedral angles with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========
        \*\* kwargs for EnergiesVMol.energies_some_dof

        Returns
        =======
        (tuple) DOFs and their computed energies.
        """
        dofs = self.sith.structures[0].dim_indices[self.nbonds + self.nangles:]
        out = self.energies_some_dof(dofs, **kwargs)
        return out

    def update_stretching(self, idef) -> None:
        """
        Function that is called when the frame is changed. Set the internal
        atribute self.kwargs_edofs to keep fixed the parameters to update the
        visualization of dofs.

        Parameters
        ==========
        idef: int
            index of the stretching to be updated to.
        """
        self.idef = idef
        self.frame = idef
        self.atoms = self.trajectory[self.frame]

        for i, atom in enumerate(self.vatoms):
            atom.pos = vp.vector(*self.atoms[i].position)

        udofs = [dof.indices for dof in self.dofs.values()]
        if len(udofs) != 0:
            self.energies_some_dof(udofs, **self.kwargs_edofs)

    def energies_all_dof(self, **kwargs) -> tuple:
        r"""
        Add all DOF with a color scale that represents the distribution of
        energy according to the JEDI method.

        Parameters
        ==========
        \*\*kwargs for EnergiesVMol.energies_some_dof

        Returns
        =======
        (tuple) DOFs and their computed energies.
        """
        dofs = self.sith.structures[0].dim_indices
        return self.energies_some_dof(dofs, **kwargs)

    def energies_some_dof(self, dofs: list, cmap=None, label: str = None,
                          labelsize: float = None, orientation: str = None,
                          div: int = None, deci: int = None, width: int = None,
                          height: int = None, absolute: int = None,
                          **kwargs) -> tuple:
        r"""
        Add the bonds with a color scale that represents the distribution of
        energy according to the JEDI method.

        Parameters
        ==========
        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.
        cmap: cmap.
            cmap used in the color bar. Default: mpl.cm.get_cmap("Blues")
        label: str.
            label of the color bar. Default: "Energy [Ha]"
        labelsize: float.
            size of the labels in the color bar defined as text points.
            Default=20
        orientation: "vertical" or "horizontal".
            orientation of the color bar. Default: "vertical"
        div: int.
            number of colors in the colorbar. Default=5
        deci: int
            number of decimals in the colorbar. Default=2
        width: int
            width (in pixels) of the space that will contain the scene and the
            color bar. Deault=700
        height: int
            height (in pixels) of the space that will contain the scene and the
            color bar. Deault=700
        absolute: bool
            True to define the color bar based on the maximum energy of the all
            the DOFS in all the stretching confs. False to define the color bar
            based on the maximum energy of the all the DOFS in the present
            stretching conf.
        \*\*kwargs of VMolecule.add_dof

        Returns
        =======
        (tuple) DOFs and their computed energies.
        """
        if cmap is None:
            cmap = self.kwargs_edofs['cmap']
        else:
            self.kwargs_edofs['cmap'] = cmap

        if label is None:
            label = self.kwargs_edofs['label']
        if labelsize is None:
            labelsize = self.kwargs_edofs['labelsize']
        if orientation is None:
            orientation = self.kwargs_edofs['orientation']
        if div is None:
            div = self.kwargs_edofs['div']
        else:
            self.kwargs_edofs['div'] = div
        if deci is None:
            deci = self.kwargs_edofs['deci']
        if width is None:
            width = self.kwargs_edofs['width']
        if height is None:
            height = self.kwargs_edofs['height']
        if absolute is None:
            absolute = self.kwargs_edofs['absolute']
        else:
            self.kwargs_edofs['absolute'] = absolute
        energies = []
        dof_ind = self.sith.structures[0].dim_indices
        for dof in dofs:
            for index, sithdof in enumerate(dof_ind):
                if (dof == sithdof).all():
                    energies.append(self.sith.dofs_energies[self.idef][index])

        assert len(dofs) == len(energies), "The number of DOFs " + \
            f"({len(dofs)}) does not correspond with the number of " + \
            f"energies ({len(energies)})"

        minval = min(energies)
        maxval = max(energies)

        # respect to the max-min dof energy of all stretching
        # TODO: can this be merged with the previous for?
        energies_ref = []
        if absolute:
            for dof in dofs:
                for index, sithdof in enumerate(dof_ind):
                    if (dof == sithdof).all():
                        energies_ref.append(self.sith.dofs_energies[:, index])
                all_ener = np.array(energies_ref).flatten()
                minval = min(all_ener)
                maxval = max(all_ener)
        # In case of all the energies are the same (e.g. 0 stretching)
        if minval == maxval:
            minval = 0
            maxval = 1

        if orientation == 'v' or orientation == 'vertical':
            rotation = 0
        else:
            rotation = 90

        boundaries = np.linspace(minval, maxval, div + 1)
        normalize = mpl.colors.BoundaryNorm(boundaries, cmap.N)

        # Costumize cbar
        if self.fig is None:
            self._create_colorbar(normalize, cmap, deci, label, labelsize,
                                  rotation)

        for i, dof in enumerate(dofs):
            color = cmap(normalize(energies[i]))[:3]
            self.add_dof(dof, color=color, **kwargs)

        return dofs, energies

    def _create_colorbar(self, normalize, cmap,
                         deci: int, label: str,
                         labelsize: float, rotation: str) -> None:
        """
        Creates the Color bar using matplotlib ans stores it in a png figure.

        Parameters
        ==========
        cmap: cmap.
            cmap used in the color bar.
        label: str.
            label of the color bar.
        labelsize: float.
            size of the labels in the color bar defined as text points.
        rotation: float
            Angle (in degrees) of the ticks.
        deci: int
            number of decimals in the colorbar.
        """
        # TODO: delete the present canvas and create a new
        # one with the created figure
        dpi = 300
        # labelsize is given in points, namely 1/72 inches
        # the width is here defined as 1.16 times the space occupied by
        # the ticks and labels(see below)
        width_inches = 0.07*labelsize
        height_inches = self.scene.height / dpi  # Convert pixels to inches

        self.fig, self.ax = plt.subplots(figsize=(width_inches, height_inches))
        cbar = self.fig.colorbar(mpl.cm.ScalarMappable(norm=normalize,
                                                       cmap=cmap),
                                 cax=self.ax, orientation='vertical',
                                 format='%1.{}f'.format(deci))
        cbar.set_label(label=label,
                       fontsize=labelsize,
                       labelpad=0.5 * labelsize)
        cbar.ax.tick_params(labelsize=0.8*labelsize,
                            length=0.2*labelsize,
                            pad=0.2*labelsize,
                            rotation=rotation)
        self.ax.set_position(Bbox([[0.01, 0.1],
                                   [0.99-0.06*labelsize/width_inches,
                                    0.9]]),
                             which='both')
        plt.savefig('colorbar.png', dpi=dpi)
        plt.close()

    def show_bonds_of_DOF(self, dof: list, unique: bool = False,
                          color: list = None) -> dict:
        """
        Show bonds of a specific dof.

        Parameters
        ==========
        dof: int.
            index in sith object that corresponds to the dof you want to show.
        unique: Bool. default False.
            True if you want to remove all the other bonds and only keeping
            these ones.
        color: list[3(int)]. default R G B for angles, distances, dihedrals.
            color that you want to use in this dof.

        Returns
        =======
        (dict) all the DOFs in the system. keys -> dof names,
        values -> vpython.objects
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

    def show_dof(self, dofs: list, **kwargs) -> dict:
        """
        Show specific degrees of freedom.

        Parameters
        ==========
        dofs: list of tuples.
            list of degrees of freedom defined according with g09 convention.

        Returns
        =======
        (dict) all the DOFs in the system. keys -> dof names,
        values -> vpython.objects

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        for dof in dofs:
            out = self.add_dof(dof, **kwargs)
        return out

    def show_bonds(self, **kwargs) -> None:
        """
        Show the bonds in the molecule.

        Notes
        -----
        The color is not related with the JEDI method. It
        could be changed with the kwarg color=rgb list.
        """
        dofs = self.sith.structures[0].dim_indices[:self.nbonds]
        self.show_dof(dofs, **kwargs)

    def traj_buttons(self) -> vp.button:
        """
        Create the buttons to move between stretching configurations.

        Returns
        =======
        (vpython.button) Button that shows the index of the stretched
        configuration displayed.
        """
        def counter(b):
            pass

        def go_up(b, vmol=self):
            frame = vmol.idef
            if frame < len(vmol.trajectory) - 1:
                vmol.update_stretching(frame + 1)
                vmol.b_counter.text = f"   {vmol.idef}   "

        def go_down(b, vmol=self):
            frame = vmol.idef
            if frame > 0:
                vmol.update_stretching(frame - 1)
                vmol.b_counter.text = f"   {vmol.idef}   "

        # button to go down
        vp.button(text='\u25C4',
                  pos=self.scene.title_anchor,
                  bind=go_down)

        # counter box
        self.b_counter = vp.button(text=f"   {self.idef}   ",
                                   pos=self.scene.title_anchor,
                                   bind=counter)
        # button to go up
        vp.button(text='\u25BA',
                  pos=self.scene.title_anchor,
                  bind=go_up)

        return self.b_counter
