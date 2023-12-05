from typing import Union, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import Output
import numpy as np
from SITH.visualize.view import VMolecule
import vpython as vp
from matplotlib.transforms import Bbox
from SITH.SITH import SITH


# TODO: add some documentation because I am using a modified version
""" of vpython. 

1) loook for vpython.py and modify the lines around

"from IPython.display import display, HTML, Javascript":

if _isnotebook:
            if 'js' in list(args.keys()):
                js = args['js']
            else:
                js = ''
            
            if 'height' in list(args.keys()):
                height = args['height']
            else:
                # because of this I just removed the
                # automatic creation of a canvas in __init__
                # just because it would mean a white unnecessary space 
                height = 500
                args['height'] = height
            
            # percentaje to of the 3d_canvas in the visualization
            if 'portion' in list(args.keys()):
                portion = args['portion']
            else:
                portion = 80

            from IPython.display import display, HTML, Javascript
            display(HTML('<body>'
                         f'<div style="width: 100%; height: {height}px;">'
                         f'<div style="width: {portion}%;" id="glowscript" class="glowscript"></div>'
                         f'<div style="float: right; width: {100-portion}%;"> '
                         f'{js}'
                         ' </div>'
                         '</div>'
                         '</body>'))

2) in /path/to/vpython/__init__.py change escene by:
scene = canvas(height=0)
"""


class EnergiesVMol(VMolecule):
    """
    Attributes
    =========
    scene"""
    def __init__(self, sith_info: SITH,
                 idef: int = 0,
                 alignment: Union[list, tuple, np.ndarray]=None,
                 show_axis: bool = False,
                 background: Union[list, vp.color] = vp.color.white,
                 portion: float = 80,
                 **kwargs):
        """Set of tools to visualize a molecule and the
        distribution of energies in different DOF.

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
        axis: bool
            add xyz axis
        background: rgb list or vpython vector
            background color. Default: vpython.color.white
        portion: float
            percentage of the complete figure to be used to add the canvas. The
            rest of the space can be used to add a figure.
        **kwargs: arguments for VMolecule
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
	                          'style="object-fit:fill;'
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
                             'width': "700px",
                             'height': "600px",
                             'absolute': False}

    # TODO: update function to last version of sith analyze
    def _analize_energies(self):
        """
        Execute JEDI method to obtain the energies of each
        DOFs

        see: https://doi.org/10.1063/1.4870334
        """
        return True
        #self.sith.extract_data()
        #self.sith.analyze()

    def energies_bonds(self, **kwargs):
        """Add the bonds with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices[:self.nbonds]
        out = self.energies_some_dof(dofs, **kwargs)

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
        return out

    def update_stretching(self, idef):
        """Function that is called when the frame is changed. Set the internal
        atribute self.kwargs_edofs to keep fixed the parameters to update the
        visualization of dofs.
        """
        self.idef = idef
        self.frame = idef
        self.atoms = self.trajectory[self.frame]

        for i, atom in enumerate(self.vatoms):
            atom.pos = vp.vector(*self.atoms[i].position)
        
        udofs = [dof.indices for dof in self.dofs.values()]
        if len(udofs) != 0:
            self.energies_some_dof(udofs, **self.kwargs_edofs)

    def energies_all_dof(self, **kwargs):
        """Add all DOF with a color scale that represents the
        distribution of energy according to the JEDI method.

        Parameters
        ==========

        optional kwargs for energies_some_dof
        """
        dofs = self.sith.structures[0].dim_indices
        return self.energies_some_dof(dofs, **kwargs)

    def energies_some_dof(self, dofs, cmap=None,
                          label=None, labelsize=None,
                          orientation=None, div=None, deci=None,
                          width=None, height=None, absolute=None, **kwargs):
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
                    energies.append(self.sith.dofs_energy[self.idef][index])

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
                        energies_ref.append(self.sith.dofs_energy[:, index])
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
            self.create_colorbar(normalize, cmap, deci, label,
                                        labelsize, rotation)

        for i, dof in enumerate(dofs):
            color = cmap(normalize(energies[i]))[:3]
            self.add_dof(dof, color=color, **kwargs)
        
        return dofs, energies
    
    def create_colorbar(self, normalize, cmap, deci, label,
                        labelsize, rotation):
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
                                 format='%1.{}f'.format(deci));
        cbar.set_label(label=label,
                       fontsize=labelsize,
                       labelpad=0.5 * labelsize);
        cbar.ax.tick_params(labelsize=0.8*labelsize,
                            length=0.2*labelsize,
                            pad=0.2*labelsize,
                            rotation=rotation);
        self.ax.set_position(Bbox([[0.01, 0.1],
                                   [0.99-0.06*labelsize/width_inches,
                                    0.9]]),
                                   which='both')
        plt.savefig('colorbar.png', dpi=dpi);
        plt.close()

        # vmol.update_stretching(frame - 1)



    def show_bonds_of_DOF(self, dof, unique=False, color=None):
        """Show bonds of a specific dof.

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
    

    def traj_buttons(self):
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
        
        return 
    
