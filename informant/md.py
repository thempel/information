import pyemma
import numpy as np
import mdtraj
import itertools
from tqdm import tqdm_notebook
import informant

def discretize_residue_backbone_sidechains(topology_file,
                                           trajectory_files,
                                           tica_lag,
                                           nstates=3,
                                           stride=1,
                                           exclude_terminal_residues=5,
                                           periodic=True):
    """
    Convenience function. Discretize backbone and sidechain motion of every
    residue into n states; keep everything in a dictionary to be processed
    later on.

    :param topology_file: str; e.g. PDB-file. Passed to pyemma.coordinates.featurizer
    :param trajectory_files: str; e.g. list of XTC-files. Passed to pyemma.coordinates.load
    :param tica_lag: int; lag time for TICA in trajectory steps
    :param nstates: int; number of states to assume per residue
    :param stride: stride for loading the data (caution, this defines the output discrete trajectory timestep)
    :param exclude_terminal_residues: int; number of terminal residues to exclude from analysis
    :param periodic: Bool; uses periodic box? Passed e.g. to featurizer.add_backbone_torsions
    :return: dict, dictionary of discrete trajectories sorted by residue
    """
    ref_traj = mdtraj.load(topology_file)
    residue_blocks = np.arange(exclude_terminal_residues, ref_traj.topology.n_residues - exclude_terminal_residues)

    dtrajs = dict()
    for r1 in tqdm_notebook(residue_blocks):

        f = pyemma.coordinates.featurizer(topology_file)
        f.add_backbone_torsions(selstr='resid {}'.format(r1), cossin=True, periodic=periodic)
        if not f.topology.residue(r1).__str__()[:3] in ('ALA', 'GLY'):
            f.add_sidechain_torsions(selstr='resid {}'.format(r1), cossin=True, periodic=periodic)

        y = pyemma.coordinates.load(trajectory_files, features=f, stride=stride)
        tica = pyemma.coordinates.tica(y, tica_lag, dim=nstates-1)

        cl = pyemma.coordinates.cluster_kmeans(tica, k=nstates, max_iter=50)

        dtrajs[str(r1)] = cl.dtrajs

    return dtrajs


def compute_inter_residue_transfer_entropy(dtrajs_dictionary, msmlag, reversible=False):
    """
    Convenience function for computing transfer entropy between each pair of
    residues in a dataset.
    :param dtrajs_dictionary: dict; Discrete trajectories stored in a dictionary.
    :param msmlag: int; lag time for MSM estimation (in strided steps)
    :param reversible: Bool; MSM estimation (caution, reversible estimation not recommended)
    :return: dict; transfer entropy & mutual information estimates sorted by residue number pairs
    """
    te = dict()

    for r1, r2 in itertools.combinations([int(k) for k in dtrajs_dictionary.keys()], 2):
        te[str(r1) + '_' + str(r2)] = dict()
        te[str(r1) + '_' + str(r2)]['r1'] = r1
        te[str(r1) + '_' + str(r2)]['r2'] = r2

        A = dtrajs_dictionary[str(r1)]
        B = dtrajs_dictionary[str(r2)]

        p = informant.MSMProbabilities(msmlag=msmlag, reversible=reversible)
        p._dangerous_ignore_warnings_flag = True

        est = informant.TransferEntropy(p)
        est.symmetrized_estimate(A, B)

        te[str(r1) + '_' + str(r2)]['te_forw'] = est.d
        te[str(r1) + '_' + str(r2)]['te_backw'] = est.r

        est = informant.MutualInfoStationaryDistribution(p)
        est.estimate(A, B)

        te[str(r1) + '_' + str(r2)]['m'] = est.m

    return te