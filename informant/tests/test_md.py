import informant
import mdshare


def test_integration_simple():
    topology_file = mdshare.fetch('pentapeptide-impl-solv.pdb', working_directory='data', show_progress=False)
    trajectory_files = mdshare.fetch('pentapeptide-00-500ns-impl-solv.xtc', working_directory='data',
                                     show_progress=False)

    from pyemma.util import contexts
    with contexts.settings(show_progress_bars=False):
        dtrajs = informant.md.discretize_residue_backbone_sidechains(
            topology_file,
            trajectory_files,
            1,
            nstates=2,
            exclude_terminal_residues=1,
            show_progress=False
        )
    te = informant.md.compute_inter_residue_transfer_entropy(dtrajs, 1)

    for link in te.values():
        assert link['m'] > 0
        assert link['te_forw'] > 0
        assert link['te_backw'] > 0
