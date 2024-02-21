import matplotlib as mpl
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
import numpy as np


def plot_hessian(hessian, ax=None, deci=2, orientation='vertical', cbar=True,
                 ticks=15):
    """
    Function that plots the a matrix using a divergent colormap to separate the
    negative from the positive values.

    Parameters
    ==========
    hessian: NxN numpy.array
        matrix to be ploted
    ax: plt.Axes
        Axis to add the plot. Default: None, in this case, the function creates
        a new Axis.
    deci: int
        number of decimals in the colorbar.
    orientation: str
        orientation of the colorbar. Default: 'vertical'.
    cbar: Bool
        True to show the colorbar. Default: True
    ticks: float
        ticks size.

    Return
    ======
    PathCollection
    """

    if orientation[0] == 'v':
        pad = 0.02
        shrink = 1
        rotation = 0
    else:
        pad = 0.15
        shrink = 0.9
        rotation = 90

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))
    if orientation[0] == 'v':
        pad = 0.02
        shrink = 0.85
        rotation = 0
    else:
        pad = 0.15
        shrink = 0.9
        rotation = 90

    cmap = mpl.cm.RdBu_r  # set the colormap to a divergent one

    indexes = np.arange(hessian.shape[0])

    x = [[i for i in indexes] for j in indexes]
    y = [[j for i in indexes] for j in indexes]

    lim = max(abs(min(hessian.flatten())), max(hessian.flatten()))

    im = ax.scatter(x, y, c=hessian.flatten(), marker='s',
                    cmap=cmap, vmin=-lim, vmax=lim)

    if cbar:
        cbar = plt.colorbar(im, ax=ax, format='%1.{}f'.format(deci),
                            orientation=orientation, pad=pad,
                            shrink=shrink)
        cbar.ax.tick_params(labelsize=ticks, rotation=rotation)
    return im


def hessian_blocks(hessian, dims, decis=[2, 2, 2, 2], orientation='vertical',
                   cbar=True, ticks=15, deltas=[1, 1, 1, 1]):

    fig, ax = plt.subplots(4, 3, figsize=(10, 12))
    """
    Plots the hessian matrix of the sith object separating it in blocks
    corresponding to the different degrees of freedom

    Parameters
    ==========
    hessian: NxN numpy.array
        matrix to be ploted.
    dims: numpy.array
        dimentions of the degrees of freedom subblocks.
    decis: list[ints]
        number of decimals in each colorbar.
    orientation: str
        orientation of the colorbar. Default: 'vertical'.
    cbar: Bool
        True to show the colorbar. Default: True
    ticks: float
        ticks size. Default: 15.
    deltas: list[float]
        deltas in the labels of the degrees of freedom. Default: [1, 1, 1, 1]

    Return
    ======
    PathCollection
    """
    if orientation[0] == 'v':
        pad = 0.02
        shrink = 1
        rotation = 0
    else:
        pad = 0.15
        shrink = 0.9
        rotation = 90
    ax[0][0].set_title('Bonds')
    plot_hessian(hessian[:dims[1], :dims[1]], ax=ax[0][0],
                 orientation='vertical', cbar=True, ticks=ticks, deci=decis[0])
    range_bonds = np.arange(1,
                            dims[1]+1,
                            deltas[0])
    ax[0][0].set_xticks(range_bonds - 1)
    ax[0][0].set_xticklabels(range_bonds)
    ax[0][0].set_yticks(range_bonds - 1)
    ax[0][0].set_yticklabels(range_bonds)

    ax[0][1].set_title('Angles')
    plot_hessian(hessian[dims[1]:dims[2]+dims[1], dims[1]:dims[2]+dims[1]],
                 ax=ax[0][1], orientation='vertical', cbar=True, ticks=ticks,
                 deci=decis[1])
    range_angles = np.arange(dims[1] + 1,
                             dims[1] + dims[2] + 1,
                             deltas[1])
    ax[0][1].set_xticks(range_angles - dims[1] - 1)
    ax[0][1].set_xticklabels(range_angles)
    ax[0][1].set_yticks(range_angles - dims[1] - 1)
    ax[0][1].set_yticklabels(range_angles)

    ax[0][2].set_title('Dihedrals')
    plot_hessian(hessian[dims[2]+dims[1]:, dims[2]+dims[1]:], ax=ax[0][2],
                 orientation='vertical', cbar=True, ticks=ticks, deci=decis[2])
    range_dihedrals = np.arange(dims[1] + dims[2] + 1,
                                dims[1] + dims[2] + dims[3] + 1,
                                deltas[2])
    ax[0][2].set_xticks(range_dihedrals - dims[1] - dims[2] - 1)
    ax[0][2].set_xticklabels(range_dihedrals)
    ax[0][2].set_yticks(range_dihedrals - dims[1] - dims[2] - 1)
    ax[0][2].set_yticklabels(range_dihedrals)

    ldx = ax[0][0].get_position().get_points()[0][0]
    ldy = ax[3][0].get_position().get_points()[0][1]
    rux = ax[0][2].get_position().get_points()[1][0]
    ruy = ax[1][2].get_position().get_points()[1][1]

    [[ax[i][j].set_visible(False) for i in range(1, 4)] for j in range(1, 3)]
    im = plot_hessian(hessian, ax=ax[1][0], cbar=False)
    ax[1][0].plot([dims[1]-0.5, dims[1]-0.5, -0.5, -0.5, dims[1]-0.5],
                  [-0.5, dims[1]-0.5, dims[1]-0.5, -0.5, -0.5], color='black',
                  lw=1)
    range_total = np.arange(1, dims[0]+1, deltas[3])
    ax[1][0].set_xticks(range_total - 1)
    ax[1][0].set_xticklabels(range_total)
    ax[1][0].set_yticks(range_total - 1)
    ax[1][0].set_yticklabels(range_total)

    ax[1][0].plot([dims[2]-0.5 + dims[1], dims[2]-0.5 + dims[1],
                   dims[1]-0.5, dims[1]-0.5,
                   dims[2]-0.5 + dims[1]],
                  [dims[1]-0.5, dims[2]-0.5 + dims[1],
                   dims[2]-0.5 + dims[1], dims[1]-0.5,
                   dims[1]-0.5], color='black', lw=1)

    ax[1][0].plot([dims[3]-0.5 + dims[1] + dims[2],
                   dims[3]-0.5 + dims[1] + dims[2],
                   dims[2]-0.5 + dims[1], dims[2]-0.5 + dims[1],
                   dims[3]-0.5 + dims[1]+dims[2]],
                  [dims[2]-0.5 + dims[1], dims[3]-0.5 + dims[1]+dims[2],
                   dims[3]-0.5 + dims[1]+dims[2], dims[2]-0.5 + dims[1],
                   dims[2]-0.5 + dims[1]], color='black', lw=1)

    cbar = fig.colorbar(im, cax=ax[3][2], format='%1.{}f'.format(decis[3]),
                        orientation=orientation, pad=pad, shrink=shrink)
    cbar.ax.tick_params(labelsize=ticks, rotation=rotation)

    ax[3][2].set_position(Bbox([[rux + 0.02, ldy], [rux + 0.05, ruy]]),
                          which='both')
    ax[2][0].set_visible(False)
    ax[3][0].set_visible(False)
    ax[3][2].set_visible(True)

    ax[1][0].set_position(Bbox([[ldx, ldy], [rux, ruy]]), which='both')
    ax[1][0].set_aspect('equal')
    print(im)

    return im
