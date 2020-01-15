import numpy as np
import pyrosetta
import pyrosetta as pr
from datetime import datetime
from pyrosetta.rosetta.core.scoring.constraints import ConstraintSet
from pyrosetta.rosetta.protocols.constraint_movers import ConstraintSetMover
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType
from pyrosetta.rosetta.protocols.minimization_packing import MinMover
from pyrosetta.rosetta.protocols.moves import *
from pyrosetta.rosetta.protocols.simple_moves import SwitchResidueTypeSetMover, SmallMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from Bio import SeqIO
import time

from argparse import ArgumentParser


# Please make the information sources you use required!

def parse_arguments():
    parser = ArgumentParser(description='Run minimization on distograms')
    parser.add_argument('--fasta', help='fasta file',
                        required=True)
    parser.add_argument('--prediction', help='distogram,ss,angle predictions',
                        required=True)
    parser.add_argument('--mapping', help='input mapping, necessary to align decoy and PDB',
                        required=True)
    parser.add_argument('--native', help='native PDB',
                        required=True)
    parser.add_argument('--constraints', help='constraint file in ROSETTA format',
                        required=False)
    parser.add_argument('--frag3', help='3mers fragment file',
                        required=False)
    parser.add_argument('--frag9', help='9mers fragment file',
                        required=False)
    parser.add_argument('--photoAA', help='photoAA XL file',
                        required=False)
    parser.add_argument('--sulfoSDA', help='sulfoSDA XL file',
                        required=False)
    parser.add_argument('--nmr', help='sparseNMR file',
                        required=False)
    parser.add_argument('--output', help='output filename of decoy',
                        required=True)
    args = parser.parse_args()
    return args


def bit2deg(angles_bit):
    """ biternion ([cos, sin]) ->  degrees
    """
    return (np.rad2deg(np.arctan2(angles_bit[:, 1], angles_bit[:, 0])) + 360) % 360


def deg2bit(deg):
    rad = deg2rad(deg)
    return np.cos(rad), np.sin(rad)


def parse_fasta(fasta_fname):
    return str(SeqIO.read(fasta_fname, "fasta").seq)


class VonMisesMover(pyrosetta.rosetta.protocols.moves.Mover):
    """
    Custom mover sampling angles from the predicted von Mises distribution
    """

    def __init__(self, residues, phi, phi_kappa, psi, psi_kappa):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.residues = residues
        self.phi = phi
        self.phi_kappa = phi_kappa
        self.psi = psi
        self.psi_kappa = psi_kappa

    def get_name(self):
        return self.__class__.__name__

    def apply(self, pose):
        # each mover step, sample all angles once
        for res in self.residues:
            # sample phi
            mu_phi = np.deg2rad(float(bit2deg(self.phi)[res - 1]))
            kappa_phi = float(self.phi_kappa[res - 1][0])
            s_phi = np.random.vonmises(mu_phi, kappa_phi)

            # sample psi
            mu_psi = np.deg2rad(float(bit2deg(self.psi)[res - 1]))
            kappa_psi = float(self.psi_kappa[res - 1][0])
            s_psi = np.random.vonmises(mu_psi, kappa_psi)

            pose.set_phi(res, np.rad2deg(s_phi))
            pose.set_psi(res, np.rad2deg(s_psi))


def load_data(d):
    phi = np.transpose(d['phi'], [1, 0])
    phi_kappa = np.transpose(d['phi_kappa'], [1, 0])
    psi = np.transpose(d['psi'], [1, 0])
    psi_kappa = np.transpose(d['psi_kappa'], [1, 0])

    return phi, phi_kappa, psi, psi_kappa


def get_mapping(ss):
    pos = 0
    mapping = {}
    mapping_rosetta = pr.rosetta.std.map_unsigned_long_unsigned_long()
    for i, s in enumerate(ss):
        if s != '-':
            mapping[i + 1] = pos + 1
            mapping_rosetta[i + 1] = pos + 1
            pos += 1
    return mapping, mapping_rosetta


def main():
    args = parse_arguments()

    # mute some rosetta outputs
    pr.init("-mute core basic protocols")

    # prediction contains: ['distogram', 'ss', 'phi', 'phi_kappa', 'psi', 'psi_kappa']
    # raw distograms are in bins(!) not in A: bins = np.arange(4,42.5,0.5) the last bin being a catch all bin
    # constraints are already converted to A
    # secondary structures predictions are in Q8 (8 classes)
    # hec = { 'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, ' ': 7, 'C': 7 }
    # see also: https://en.wikipedia.org/wiki/DSSP_(hydrogen_bond_estimation_algorithm)
    # These eight types are usually grouped into three larger classes: helix (G, H and I), strand (E and B) and 
    # loop (S, T, and C, where C sometimes is represented also as blank space).
    # Q3 is the grouping into the three claseses ^^^^.

    prediction = np.load(args.prediction)

    ss = parse_fasta(args.mapping)

    N = len(ss)

    phi, phi_kappa, psi, psi_kappa = load_data(prediction)

    pose = pr.pose_from_sequence(parse_fasta(args.fasta))

    native = pr.io.pose_from_pdb(args.native)

    t0 = time.perf_counter()

    # we start minimizing in centroid mode
    to_centroid = SwitchResidueTypeSetMover("centroid")
    to_centroid.apply(pose)

    # set up scoring function
    score = pr.create_score_function("score4_smooth")

    cst = ConstraintSetMover()
    if args.constraints:
        score.set_weight(ScoreType.atom_pair_constraint, 1.0)
        cst.constraint_file(args.constraints)
        cst.apply(pose)

    # set up PyMOL observer, apply updates the structure in PyMOL
    # pm = pr.PyMOLMover()
    # pm.apply(pose)

    # we will be using gradient descent (LBFGS)
    minimizer = MinMover()
    minimizer.min_type("lbfgs_armijo_nonmonotone")

    # needs to be changed, if you want to keep some parts of the structure fixed
    mm = MoveMap()
    mm.set_bb_true_range(1, N)

    minimizer.movemap(mm)
    minimizer.score_function(score)

    # residues are 1-indexed in rosetta!
    vmm = VonMisesMover(range(1, N + 1), phi, phi_kappa, psi, psi_kappa)

    # alternative mover sequence with Monte Carlo

    # kT = 10
    # smallmoves = 100
    # smallmover = SmallMover(mm, kT, smallmoves)

    # mc = pyrosetta.rosetta.protocols.moves.MonteCarlo(pose, score, 0) # 0 to always restart from the best
    # perturbation_mover = SequenceMover()
    # perturbation_mover.add_mover(minimizer)
    # perturbation_mover.add_mover(smallmover)
    # perturbation_mover.add_mover(minimizer)
    # tm = pyrosetta.rosetta.protocols.moves.TrialMover(perturbation_mover, mc) 

    # setting up fragments

    # if args.frag3 and args.frag9:
    #     fragset3mer = ConstantLengthFragSet(3, args.frag3)
    #     mover_3mer = ClassicFragmentMover(fragset3mer,movemap)
    #     fragset9mer = ConstantLengthFragSet(9, args.frag9)
    #     mover_9mer = ClassicFragmentMover(fragset9mer,movemap)

    best_pose = None
    best_score = float('inf')

    # sample starting conformation
    vmm.apply(pose)

    # current protocol: sample angles once at random from predicted phi/psi distribution, minimize
    s = score(pose)
    for it in range(
            3):  # in principal, minimizer should converge after one application, but sometimes it runs out of minimization steps
        # tm.apply(pose)
        minimizer.apply(pose)
        s_ = score(pose)
        # smallmover.apply(pose)
        # print(it,s_)
        if int(s_) == int(s):
            break
        s = s_

    # switch to higher resolution = more expensive scoring function
    to_fa = SwitchResidueTypeSetMover("fa_standard")
    to_fa.apply(pose)
    sf = pr.create_score_function("ref2015")  # full atom

    # play with constraint weights in relax
    if args.constraints:
        sf.set_weight(ScoreType.atom_pair_constraint, 5.0)
        cst.apply(pose)

    # relax usually helps, fixes side chains etc. but it's very expensive/ time-consuming!
    relax = pr.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf)
    relax.apply(pose)

    t1 = time.perf_counter()

    # compute RMSD, in case parts of the native are missing, compute mapping between the two
    mapping, mapping_rosetta = get_mapping(ss)
    s = sf(pose)
    sf.set_weight(ScoreType.atom_pair_constraint, 0)
    s_ = sf(pose)
    score_rsmd = pr.rosetta.core.scoring.CA_rmsd(pose, native, mapping_rosetta)
    print(args.fasta, score_rsmd, s, s_)
	
    # @jonny datestring kann gelöscht werden wenn der nervt
    dt_string = datetime.now().strftime("%d%m%Y_%H%M")
    name = args.output + dt_string
    pose.dump_pdb(name + '.pdb')
    with open(name + '.args', 'w') as f:
        f.write(str(args) + '\n')
        f.write("Results: \n" + ','.join([str(i) for i in [args.fasta, score_rsmd, s, s_, t1-t0]]))
        f.close()


if __name__ == "__main__":
    main()
