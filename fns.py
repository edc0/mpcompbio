import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib
from itertools import product
from sklearn.neighbors import KernelDensity
from scipy.special import kl_div as kl_divergence
import matplotlib.pylab as pl
import pyrosetta as pr
import sys, os


ANGSTROM_BINS = np.arange(4,43,0.5)
additive_photo = np.load("dist_close.npy")
additive_sulfo = np.load("dist_medium.npy")


ANGSTROM_BINS = np.arange(4,43,0.5)

class Protein:

    def __init__(self, dirname, name):
        self.name = name

        self.file_native = dirname + "pdb/" + name + ".pdb"
        self.native = pr.io.pose_from_pdb(dirname + "pdb/" + name + ".pdb")
        self.sequence = self.native.sequence()

        self.file_contacts = dirname + "contacts/" + name + ".contacts"
        self.contacts = pd.read_csv(self.file_contacts, sep=" ", names=["i", "j", "_1", "_2", "p"])[["i", "j", "p"]]

        self.file_photo = dirname + "photoAA_xl/" + name + "_FDR10"
        self.xl_photoAA = pd.read_csv(self.file_photo, sep=" ", names=["i", "j"])

        self.file_sulfo = dirname + "sulfo-SDA_xl/" + name + ".xl"
        self.xl_sulfo = pd.read_csv(self.file_sulfo, sep=" ", names=["i", "j"])

        self.file_numpy_dist = dirname + 'distograms/' + name + '.npz'
        self.dist_info = np.load(self.file_numpy_dist)

        self.file_fasta = dirname + 'fasta/' + name + '.fasta'
        self.file_map = 'mapping/' + name

        self.file_distogram_cst = 'distogram_cst/' + name + '.cst'

        self.distogram = self.dist_info['distogram']
        self.ground_truth_dist = self._compute_ground_truth_dist()
        self.ground_truth_dist_clipped = self._compute_ground_truth_dist_clipped()

        print("Initialized " + self.name)

    def _compute_ground_truth_dist_clipped(self):
        return self.ground_truth_dist.clip(4,42.5)


    def _compute_ground_truth_dist(self):
        pose = self.native
        def xyz(pose,res):

            return pose.residue(res).xyz("N")
        residues = pose.total_residue()
        dist = np.zeros((residues,residues))
        for i, j in product(range(1,residues+1),range(1,residues+1)):
            dist[i-1,j-1] = (xyz(pose,i) - xyz(pose,j)).norm()

        return dist

    def get_mapping(self):
        with open(self.file_map,'r') as f:
            f.readline()
            ss = f.readline().strip()
            pos = 0
            mapping = np.zeros(self.ground_truth_dist.shape[1])
            for i, s in enumerate(ss):
                if s != '-':
                    mapping[pos] = i
                    pos += 1

        return mapping.astype(int)

    def map_distogram_for_native(self,dist_2d):
        m = self.get_mapping()
        z = dist_2d[:,m]
        return z[m,:]


    @staticmethod
    def estimate_dist_argmax(dist_3d):
        return np.argmax(dist_3d,axis=0) + 4 # count starts only at four angstrom

    @staticmethod
    def estimate_dist_weighted(dist_3d):
        a = np.sum(ANGSTROM_BINS[:,np.newaxis] * dist_3d.reshape(dist_3d.shape[0],-1), axis=0)
        return a.reshape(int(np.sqrt(a.shape[0])),-1)

    def __repr__(self):
        return self.name

def calc_distance_estimation_sulfo(p,filter, bandwidth=3,kernel='gaussian'):
    residues = range(p.distogram.shape[1])
    sulfo = p.xl_sulfo[["i", "j"]].values.astype(np.int32) - 1
    if filter:
        s = p.distogram[np.logical_and(ANGSTROM_BINS>=10, ANGSTROM_BINS<= 25)][:, sulfo[:,0], sulfo[:,1]]
        sulfo = sulfo[s.sum(axis=0)>0.1]
    sulfo_all = np.vstack((sulfo,np.vstack((sulfo[:,1],sulfo[:,0])).T))
    if sulfo_all.shape[0] == 0:
        print('not sulfo Xlinks available')
        return np.zeros((len(residues),len(residues)))
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(sulfo_all)
    X,Y = np.meshgrid(residues,residues)
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    return np.exp(kde.score_samples(xy).reshape(len(residues),len(residues)))

def calc_distance_estimation_photo(p,filter,bandwidth=3,kernel='gaussian'):
    residues = range(p.distogram.shape[1])
    photo = p.xl_photoAA[["i", "j"]].values.astype(np.int32) - 1
    if filter:
        s = p.distogram[ANGSTROM_BINS<10][:, photo[:,0], photo[:,1]]
        photo = photo[s.sum(axis=0)>0.1]
    photo_all = np.vstack((photo,np.vstack((photo[:,1],photo[:,0])).T))
    if photo_all.shape[0] == 0:
        print('not photo Xlinks available')
        return np.zeros(len(residues),len(residues))
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(photo_all)
    X,Y = np.meshgrid(residues,residues)
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    return np.exp(kde.score_samples(xy).reshape(len(residues),len(residues)))

def adapt_for_greater_than(dist, sulfo_Z, photo_Z, alpha_sulfo=1000, alpha_photo=1000):
    add = np.tile(additive_photo.reshape((-1, 1, 1)), (1, dist.shape[1], dist.shape[2]))
    weighted_add = photo_Z.reshape(-1, photo_Z.shape[0], photo_Z.shape[1]) * add * alpha_photo
    dist_ = dist + weighted_add
    dist = dist_ / dist_.sum(axis=0)

    add = np.tile(additive_sulfo.reshape((-1, 1, 1)), (1, dist.shape[1], dist.shape[2]))
    weighted_add = sulfo_Z.reshape(-1, sulfo_Z.shape[0], sulfo_Z.shape[1]) * add * alpha_sulfo
    dist_ = dist + weighted_add
    dist_ /= dist_.sum(axis=0)

    return dist_

def jensen(p,q):
    m = (p + q) / 2
    return ((kl_divergence(p,m) + kl_divergence(q,m))/2)


def create_distogram(p, filter, bandwidth, alpha_sulfo, alpha_photo):
    p_sulfo = calc_distance_estimation_sulfo(p, filter, bandwidth)
    p_photo = calc_distance_estimation_photo(p, filter, bandwidth)
    d = adapt_for_greater_than(p.distogram,
                      p_sulfo, p_photo, alpha_sulfo, alpha_photo
                      )
    return d

def save_to_rosetta_format_cmd(p, distogram_file, outname):
    # for converting to rosetta
    python = sys.executable
    snippet = 'code/distograms_to_rosetta.py'
    params = ' '.join(['--prediction', distogram_file,
    '--fasta',  p.file_fasta,
    '--cst_folder', outname.replace('.','_'),
    '--cst_file', outname ])
    return ' '.join([python, snippet, params])


def minimize_cmd(p, dist_numpy,dist_file_rosetta, outname):
    python = sys.executable
    snippet = "minimize.py"
    params = ' '.join(['--fasta', p.file_fasta,
    '--prediction', dist_numpy,
    '--mapping' , p.file_map,
    '--native', p.file_native,
    '--output', outname,
    '--constraint', dist_file_rosetta
                      ])
    return ' '.join([python, snippet, params])


def save_all(p,new_dist,numpy_dist_name):
    np.savez(numpy_dist_name,
             distogram=new_dist,
             ss=p.dist_info['ss'],
             phi=p.dist_info['phi'],
             phi_kappa=p.dist_info['phi_kappa'] ,
             psi=p.dist_info['psi'] ,
             psi_kappa=p.dist_info['ss'])
