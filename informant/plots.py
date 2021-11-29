import matplotlib.pyplot as plt
import numpy as np
import mdtraj

def _draw_ddsp_frame(ax, ref_trajectory, posl, posr,
                    offset=.05, lw=1, colorscheme=['r', 'y', 'b'], topbottom=True, **kwargs):
    #   - 'H' : Helix. Either of the 'H', 'G', or 'I' codes.
    #   - 'E' : Strand. Either of the 'E', or 'B' codes.
    #   - 'C' : Coil. Either of the 'T', 'S' or ' ' codes.

    dssp = mdtraj.compute_dssp(ref_trajectory)[0]
    dssp_ = np.zeros_like(dssp, dtype=int)
    dssp_[dssp == 'E'] = 1
    dssp_[dssp == 'C'] = 2
    dssp_col = np.array(colorscheme)[dssp_]

    label_posl = posl.copy()
    label_posl[:, 1] = label_posl[:, 1] - .5 / (ref_trajectory.topology.n_residues - 1)
    label_posl = np.vstack(
        [label_posl, [label_posl[-1][0], label_posl[-1][1] + 1 / (ref_trajectory.topology.n_residues - 1)]])
    for n in range(label_posl.shape[0] - 1):
        ax.plot(*(label_posl[n:n + 2] + [[0 - offset, 0], [0 - offset, 0]]).T, color=dssp_col[n], linewidth=lw,
                clip_on=False, solid_capstyle="butt", **kwargs)
        ax.plot(*(label_posl[n:n + 2] + [[1 + offset, 0], [1 + offset, 0]]).T, color=dssp_col[n], linewidth=lw,
                clip_on=False, solid_capstyle="butt", **kwargs)
    label_posr = posr.copy()
    label_posr[:, 0] = label_posr[:, 0] - .5 / (ref_trajectory.topology.n_residues - 1)
    label_posr = np.vstack(
        [label_posr, [label_posr[-1][0] + 1 / (ref_trajectory.topology.n_residues - 1), label_posr[-1][1]]])
    if topbottom:
        for n in range(label_posr.shape[0] - 1):
            ax.plot(*(label_posr[n:n + 2] + [[0, 0 - offset], [0, 0 - offset]]).T, color=dssp_col[n], linewidth=lw,
                    clip_on=False, solid_capstyle="butt", **kwargs)

            ax.plot(*(label_posr[n:n + 2] + [[0, 1 + offset], [0, 1 + offset]]).T, color=dssp_col[n], linewidth=lw,
                    clip_on=False, solid_capstyle="butt", **kwargs)

    return ax


def plot_directed_links(ref_trajectory,
                        te,
                        di_cc=None,
                        thres=5,
                        thresm=0,
                        direct_link_thres=.05,
                        labels=['MI', 'TE', 'ccDI'],
                        figsize=(18, 9)):
    """
    Convencience plot function for visualizing transfer entropy / directed information.
    :param ref_trajectory: mdtraj.Trajectory; reference for plotting residue numbers and secondary structure info
    :param te: dict; transfer entropy dictionary
    :param di_cc: dict, optional; causally conditioned directed information
    :param thres: float; multiplicity threshold for directionality
    It is considered a directional link if either forward-TE > thres * backward-TE or vice versa.
    :param thresm: float; Absolute threshold for plotting mutual information
    :param direct_link_thres: float; Threshold for directed links
    A directional link is considered direct if there is no ccDI < direct_link_thres * max(forward-TE, backward-TE)
    :param labels: list of str; plotting labels
    :return: matplotlib.ax instance
    """
    # directed links based on cutoff that is adjusted to TE
    fig, ax = plt.subplots(figsize=figsize)

    # plot properties
    offset = .5
    wspace = .25

    color = 'k'

    xmin = min([abs(d['te_forw'] - d['te_backw']) for d in te.values() if
                (d['te_forw'] / d['te_backw'] > thres or d['te_backw'] / d['te_forw'] > thres)])
    xmax = max([abs(d['te_forw'] - d['te_backw']) for d in te.values() if
                (d['te_forw'] / d['te_backw'] > thres or d['te_backw'] / d['te_forw'] > thres)])

    alphafactorm = 1 / (max([d['m'] for d in te.values()]) - thresm)

    posl = np.vstack(
        [np.zeros((ref_trajectory.topology.n_residues)) - offset, np.linspace(-1, 1, ref_trajectory.topology.n_residues)]).T
    posr = np.vstack(
        [np.zeros((ref_trajectory.topology.n_residues)) + offset, np.linspace(-1, 1, ref_trajectory.topology.n_residues)]).T

    # for dim, feat in enumerate(np.array(feat_desc)[dimmap]):
    for d in te.values():
        a = d['r1']
        b = d['r2']
        # if abs(d['te_forw'] - d['te_backw']) > thres:
        # check if a) directional and b) direct link
        if (d['te_forw'] / d['te_backw'] > thres or d['te_backw'] / d['te_forw'] > thres):

            arrow_aims_at = posr[a] if d['te_backw'] > d['te_forw'] else posr[b]
            arrow_starts_at = posl[b] if d['te_backw'] > d['te_forw'] else posl[a]

            ax.annotate(xy=arrow_aims_at + np.array([2 * offset + wspace, 0]),  # arrow points to xy
                        xytext=arrow_starts_at+ np.array([2 * offset + wspace, 0]),
                        s='',
                        arrowprops=dict(arrowstyle='->', color=color,
                                        alpha=(abs(d['te_forw'] - d['te_backw']) - xmin) / (xmax - xmin)))
            if di_cc is not None:
                if not np.any(abs(di_cc[str(d['r1']) + '_' + str(d['r2'])]['cc'][np.isfinite(
                        di_cc[str(d['r1']) + '_' + str(d['r2'])]['cc'])]) < direct_link_thres * max(d['te_forw'],
                                                                                                    d['te_backw'])):
                    ax.annotate(xy=arrow_aims_at + np.array([2*(2 * offset + wspace), 0]),  # arrow points to xy
                                xytext=arrow_starts_at + np.array([2*(2 * offset + wspace), 0]),
                                s='',
                                arrowprops=dict(arrowstyle='->', color=color,
                                                alpha=(abs(d['te_forw'] - d['te_backw']) - xmin) / (xmax - xmin)))

        if d['m'] > thresm:
            arrow_aims_at = posr[a]
            arrow_starts_at = posl[b]

            ax.annotate(xy=arrow_aims_at,  # arrow points to xy
                        xytext=arrow_starts_at,
                        s='',
                        arrowprops=dict(arrowstyle='<->', color=color, alpha=(d['m'] - thresm) * alphafactorm))

    ax.set_title('')
    for n in range(2 + int(di_cc is not None)):
        _draw_ddsp_frame(ax, ref_trajectory,
                         posl + n * np.array([2 * offset + wspace, 0]), posr + n * np.array([2 * offset + wspace, 0]),
                         topbottom=False, offset=.03, lw=3, alpha=.6)
        ax.text(n * np.array([2 * offset + wspace]), 1., labels[n], ha='center', fontsize=12, fontweight='bold')
        for pos in [posl, posr]:
            for _pos, _res in zip(pos, ref_trajectory.topology.residues):
                ax.annotate(xy=_pos + n * np.array([2 * offset + wspace, 0]),
                            s=str(_res.resSeq), fontsize=10, alpha=.8, ha='center')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(-1.1 * offset, n * 1.25 * (2 * offset + wspace))
    ax.axis('off')

    return ax