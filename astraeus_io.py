#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm


#######################################
# Functions to read the configuration #
#######################################

def read_inifile(fname):
    from configparser import ConfigParser
    cfg = ConfigParser(inline_comment_prefixes=['#'])
    cfg.read(fname)
    return cfg


######################################
# Functions to read the GALAXY trees #
######################################

class read_snapshot_memmap:
    """
    Read the horizontal outputs from ASTRAEUS
    """

    def __init__(self, path='./', iout=46, outputsnapnumber=73, h=0.6777, metals=True):

        filename = path + '/Galsnap_' + str(iout) + '.dat'
        print(filename)

        if metals:
            dt = str(43 + 2 * outputsnapnumber) + 'f'
        else:
            dt = str(25 + outputsnapnumber) + 'f'

        data = np.memmap(filename, dtype=dt, mode='r')

        counter = 0
        # Read values (be carefult to masses expressed in units h^-1!)
        self.aexp = data[0, 0]
        self.pos_x = data[:, 1]  # x coordinate
        self.pos_y = data[:, 2]  # y coordinate
        self.pos_z = data[:, 3]  # z coordinate
        self.vel_x = data[:, 4]  # x velocity
        self.vel_y = data[:, 5]  # y velocity
        self.vel_z = data[:, 6]  # z velocity
        self.Mvir = data[:, 7] / h  # halo mass
        self.Mvir_prog = data[:, 8] / h  # halo mass progenitor
        self.Rvir = data[:, 9] / h  # virial radius
        self.spin = data[:, 12]
        self.Mgas_ini = data[:, 14] / h  # INITIAL gas mass
        self.frac_Mgas_merger = data[:, 15]

        if metals:
            self.Mgas_new = data[:, 16] / h  # gas mass
            self.Mgas_ej = data[:, 17] / h  # gas mass
            counter = counter + 2

        self.Mgas = data[:, 16 + counter] / h  # gas mass
        self.Mstar = data[:, 17 + counter] / h  # stellar mass
        self.fesc = data[:, 18 + counter]
        self.Nion = data[:, 19 + counter]
        self.fej = data[:, 20 + counter]
        self.feff = data[:, 21 + counter]  # feff
        self.fg = data[:, 22 + counter]  # fg (see evol_gal.c)
        self.zreion = data[:, 23 + counter]  # reionization redshift
        self.photHI_bg = data[:, 24 + counter]
        self.stellarmasshistory = data[
                                      :, 25 + counter:25 + outputsnapnumber + counter] / h  # stellar mass history array

        if metals:
            self.MZ = data[:, 27 + outputsnapnumber] / h
            self.MZ_O = data[:, 28 + outputsnapnumber] / h
            self.MZ_Fe = data[:, 29 + outputsnapnumber] / h

            self.MZ_ini = data[:, 30 + outputsnapnumber] / h
            self.MZ_O_ini = data[:, 31 + outputsnapnumber] / h
            self.MZ_Fe_ini = data[:, 32 + outputsnapnumber] / h

            self.MZ_new = data[:, 33 + outputsnapnumber] / h
            self.MZ_O_new = data[:, 34 + outputsnapnumber] / h
            self.MZ_Fe_new = data[:, 35 + outputsnapnumber] / h

            self.MZ_frac_merger = data[:, 36 + outputsnapnumber]
            self.MZ_O_frac_merger = data[:, 37 + outputsnapnumber]
            self.MZ_Fe_frac_merger = data[:, 38 + outputsnapnumber]

            self.MZ_ej = data[:, 39 + outputsnapnumber] / h
            self.MZ_O_ej = data[:, 40 + outputsnapnumber] / h
            self.MZ_Fe_ej = data[:, 41 + outputsnapnumber] / h

            self.metalmasshistory = data[:, 42 + outputsnapnumber:42 + 2 * outputsnapnumber] / h

            self.Mdust = data[:, 42 + 2 * outputsnapnumber] / h


def read_snapshot_raw(iout, path='./', outputsnapnumber=73, metals=False):
    """
    Read the Galsnap files ("horizontal" outputs) and return a structured array
    """

    gfile = f'{path}/Galsnap_{iout:d}.dat'
    names = ['aexp', 'pos_x', 'pos_y', 'pos_z',
             'vel_x', 'vel_y', 'vel_z',
             'Mvir', 'Mvir_prog',
             'Rvir', 'velDisp', 'velMax', 'spin', 'aexplastmerger',
             'Mgas_ini', 'frac_Mgas_merger']
    if metals:
        names += ['Mgas_new', 'Mgas_ej']
    names += ['Mgas', 'Mstar', 'fesc', 'Nion', 'fej', 'feff', 'fg',
              'zreion', 'photHI_bg',
              'stellarmasshistory']
    if metals:
        names += ['MZ', 'MZ_O', 'MZ_Fe',
                  'MZ_ini', 'MZ_O_ini', 'MZ_Fe_ini',
                  'MZ_new', 'MZ_O_new', 'MZ_Fe_new',
                  'MZ_frac_merger', 'MZ_O_frac_merger', 'MZ_Fe_frac_merger',
                  'MZ_ej', 'MZ_O_ej', 'MZ_Fe_ej',
                  'metalmasshistory',
                  'Mdust']

    formats = ['f4', 'f4', 'f4', 'f4',
               'f4', 'f4', 'f4',
               'f4', 'f4',
               'f4', 'f4', 'f4', 'f4', 'f4',
               'f4', 'f4']
    if metals:
        formats += ['f4', 'f4']
    formats += ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4',
                f'{outputsnapnumber}f4']
    if metals:
        formats += ['f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    f'{outputsnapnumber}f4',
                    'f4']

    dt = np.dtype({'names': names, 'formats': formats})

    gals = np.fromfile(gfile, dtype=dt)
    return gals


def read_snapshot(iout, path='./', outputsnapnumber=73, metals=True):
    """
    Read the Galsnap files ("horizontal" outputs) in a Pandas DataFrame !!! DOES NOT PROPERLY WORK
    """

    # Reads the raw data
    gals = read_snapshot_raw(iout, path, outputsnapnumber, metals=metals)
    # Remove the stellarmasshistory field from the list of names
    names = list(gals.dtype.names)
    if 'stellarmasshistory' in names:
        names.remove('stellarmasshistory')
    if 'metalmasshistory' in names:
        names.remove('metalmasshistory')

    # return pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in gals.items() ]))

    return pd.DataFrame(gals[names])


def read_treefile_raw(fname, metals=True, rockstarID=True):
    """
    Read a single treefile
    """

    if rockstarID:
        names = ['ID', 'descID']
    else:
        names = []
    names += ['localID', 'localDescID', 'numprog', 'snapnum',
              'aexp', 'pos_x', 'pos_y', 'pos_z',
              'vel_x', 'vel_y', 'vel_z',
              'Mvir', 'Mvir_prog',
              'Rvir', 'velDisp', 'velMax', 'spin', 'aexplastmerger',
              'Mgas_ini', 'frac_Mgas_merger']
    if metals:
        names += ['Mgas_new', 'Mgas_ej']
    names += ['Mgas', 'Mstar', 'fesc', 'Nion', 'fej', 'feff', 'fg',
              'zreion', 'photHI_bg']
    if metals:
        names += ['MZ', 'MZ_O', 'MZ_Fe',
                  'MZ_ini', 'MZ_O_ini', 'MZ_Fe_ini',
                  'MZ_new', 'MZ_O_new', 'MZ_Fe_new',
                  'MZ_frac_merger', 'MZ_O_frac_merger', 'MZ_Fe_frac_merger',
                  'MZ_ej', 'MZ_O_ej', 'MZ_Fe_ej',
                  'Mdust']
    if rockstarID:
        names += ['Padding']

    if rockstarID:
        formats = ['i8', 'i8']
    else:
        formats = []
    formats += ['i4', 'i4', 'i4', 'i4',
                'f4', 'f4', 'f4', 'f4',
                'f4', 'f4', 'f4',
                'f4', 'f4',
                'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4']
    if metals:
        formats += ['f4', 'f4']
    formats += ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                'f4', 'f4']
    if metals:
        formats += ['f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4', 'f4', 'f4',
                    'f4']
    if rockstarID:
        formats += ['f4']

    dt = np.dtype({'names': names, 'formats': formats})

    ngtree, = np.fromfile(fname, dtype='i', count=1)
    offset = np.dtype('i').itemsize

    with open(fname, 'rb') as fnameID:
        fnameID.seek(offset)
        ngal = np.fromfile(fnameID, dtype='i', count=ngtree)
        # ngal = np.fromfile(fname, dtype='i', offset=offset, count=ngtree)
        offset += np.dtype('i').itemsize * ngtree
        fnameID.seek(offset)
        data = np.fromfile(fnameID, dtype=dt)
    # data = np.fromfile(fname, dtype=dt, offset=offset)
    return ngtree, ngal, data


def read_treefile(fname, metals=True, rockstarID=False):
    """
    Read a single treefile and return it a Pandas DataFrame
    """
    ngtree, ngal, data = read_treefile_raw(fname, metals, rockstarID)
    treeID = np.repeat(np.arange(ngtree), ngal)
    df = pd.DataFrame(data)
    df['treeID'] = treeID

    return df


def read_all_treefiles(pattern, metals=True, rockstarID=False):
    """
    Read all treefiles in a Pandas DataFrame
    """
    fnames = sorted(glob(pattern))
    df_trees = []
    ngtrees = []
    ngals = []

    print(fnames)

    for fname in fnames:
        ngtree, ngal, data = read_treefile_raw(fname, metals, rockstarID)
        ngtrees.append(ngtree)
        ngals.append(ngal)
        df_trees.append(pd.DataFrame(data))

    ngals = np.concatenate(ngals)
    treeID = np.repeat(np.arange(np.sum(ngtrees)), ngals)
    df = pd.concat(df_trees)
    df['treeID'] = treeID

    return df


##########################################
# Functions to read the HALO trees (ICs) #
##########################################
# /net/eos/data/users/delphi/MD_simulations/VSMD_160mpc_3840/trees_z4.5_localhorizontal/trees_z4.5_localhorizontal.%d.dat
def read_halotree_raw(fname):
    """
    Read the Galsnap files ("horizontal" outputs) and return a structured array
    """

    names = ['ID', 'descID', 'localID', 'localDescID', 'numProg',
             'snapnumber', 'aexp', 'aexpdesc',
             'pos_x', 'pos_y', 'pos_z',
             'vel_x', 'vel_y', 'vel_z',
             'Mvir', 'Rvir', 'velDisp', 'velMax',
             'R50', 'spin', 'spinBullock', 'aexplastmerger']
    formats = ['i8', 'i8', 'i4', 'i4', 'i4',
               'i4', 'f4', 'f4',
               'f4', 'f4', 'f4',
               'f4', 'f4', 'f4',
               'f4', 'f4', 'f4', 'f4',
               'f4', 'f4', 'f4', 'f4']

    dt = np.dtype({'names': names, 'formats': formats})

    numtrees, = np.fromfile(fname, dtype='i', count=1)
    offset = np.dtype('i').itemsize
    data = []
    numhalos = []

    print(numtrees)

    print(f"Reading {numtrees} trees")
    # for i in tqdm(range(numtrees)):
    # nh, = np.fromfile(fname, dtype='i', count=1, offset=offset)
    # numhalos.append(nh)
    # offset += np.dtype('i').itemsize
    # tree = np.fromfile(fname, dtype=dt, offset=offset, count=nh)
    # offset += dt.itemsize*nh
    ##data.append(pd.DataFrame(tree))
    # data.append(tree)

    with open(fname, 'rb') as fnameID:
        for i in tqdm(range(numtrees)):
            fnameID.seek(offset)
            nh, = np.fromfile(fname, dtype='i', count=numtrees)
            offset += np.dtype('i').itemsize
            fnameID.seek(offset)
            tree = np.fromfile(fname, dtype=dt, count=nh)
            data.append(tree)
            offset += dt.itemsize * nh

    print(np.shape(numhalos))
    print(np.shape(data))

    return numtrees, np.array(numhalos), np.concatenate(data)


def read_all_halotrees(pattern):
    """
    Read all treefiles in a Pandas DataFrame
    """
    fnames = sorted(glob(pattern))
    trees = []
    ngtrees = []
    nhalos = []
    nhalostot = []

    print(fnames)

    for fname in fnames:
        print(f"Reading {fname}")
        ngtree, nhalo, data = read_halotree_raw(fname)
        ngtrees.append(ngtree)
        nhalos.append(nhalo)
        # df_trees.append(pd.DataFrame(data))
        trees.append(data)
        nhalostot.append(np.sum(nhalo))

    nhalos = np.concatenate(nhalos)
    treeID = np.repeat(np.arange(np.sum(ngtrees)), nhalos)
    fileID = np.repeat(np.arange(len(fnames)), nhalostot)
    trees = np.concatenate(trees)
    df = pd.DataFrame(trees)
    df['treeID'] = treeID
    df['fileID'] = fileID

    return df


####################################
# Functions to read the CIFOG data #
####################################
def read_cube(fname, inifile, dtype='float'):
    # Read the parameters from the config file
    cfg = read_inifile(inifile)
    N = cfg.getint('Simulation', 'gridsize')
    # Read the data
    data = np.fromfile(fname, dtype=dtype).reshape((N, N, N))
    return data


if __name__ == '__main__':
    print('Please import the module')
