
import subprocess
from glob import glob
from os.path import join

import numpy as np

import seisflows.seistools.specfem3d as solvertools
from seisflows.seistools.shared import load

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import exists, setdiff
from seisflows.tools.config import findpath, ParameterObj
from seisflows.tools.io import loadbin, savebin

PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')

import system
import preprocess


class base(object):
    """ Python interface for SPECFEM3D

      eval_func, eval_grad, apply_hess
        These methods deal with evaluation of the misfit function or its
        derivatives and provide the primary interface between the solver and
        other workflow components.

      forward, adjoint
        These methods allow direct access to individual SPECFEM3D components.
        Together, they provide a secondary interface users can employ for
        specialized tasks not covered by high level methods.

      prepare_solver, prepare_data, prepare_model
        SPECFEM3D requires a particular directory structure in which to run and
        particular file formats for models, data, and parameter files. These
        methods help put in place all these prerequisites.

      load, save
        For reading and writing SPECFEM3D models and kernels. On the disk,
        models and kernels are stored as binary files, and in memory, as
        dictionaries with different keys corresponding to different material
        parameters.

      split, merge
        In the solver routines, it is possible to store models as dictionaries,
        but for the optimization routines, it is necessary to merge all model
        values together into a single vector. Two methods, 'split' and 'merge',
        are used to convert back and forth between these two representations.

      combine, smooth
        Utilities for combining and smoothing kernels, meant to be called from
        external postprocessing routines.
    """

    # model parameters
    model_parameters = []
    model_parameters += ['rho']
    model_parameters += ['vp']
    model_parameters += ['vs']

    # inversion parameters
    inversion_parameters = []
    inversion_parameters += ['vp']
    inversion_parameters += ['vs']

    kernel_map = {
        'rho': 'rho_kernel',
        'vp': 'alpha_kernel',
        'vs': 'beta_kernel'}


    def check(self):
        """ Checks parameters, paths, and dependencies
        """
        # check time stepping parameters
        if 'NT' not in PAR:
            raise Exception

        if 'DT' not in PAR:
            raise Exception

        if 'F0' not in PAR:
            raise Exception

        # check paths
        if 'GLOBAL' not in PATH:
            raise Exception

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        if 'SOLVER' not in PATH:
            if PATH.LOCAL:
                setattr(PATH, 'SOLVER', join(PATH.LOCAL, 'solver'))
            else:
                setattr(PATH, 'SOLVER', join(PATH.GLOBAL, 'solver'))


    def setup(self):
        """ Prepares solver for inversion or migration

          As input for an inversion or migration, users can choose between
          supplying data or providing a target model from which data are
          generated on the fly. In both cases, all necessary SPECFEM3D input
          files must be provided.
        """
        unix.rm(self.getpath)

        # prepare data
        if PATH.DATA:
            self.initialize_solver_directories()
            src = glob(PATH.DATA +'/'+ self.getname +'/'+ '*')
            dst = 'traces/obs/'
            unix.cp(src, dst)

        else:
            self.generate_data(
                model_path=PATH.MODEL_TRUE,
                model_name='model_true',
                model_type='gll')

        # prepare model
        self.generate_mesh(
            model_path=PATH.MODEL_INIT,
            model_name='model_init',
            model_type='gll')

        self.initialize_adjoint_traces()
        self.initialize_io_machinery()


    def generate_data(self, **model_kwargs):
        """ Generates data
        """
        self.generate_mesh(**model_kwargs)

        unix.cd(self.getpath)
        solvertools.setpar('SIMULATION_TYPE', '1')
        solvertools.setpar('SAVE_FORWARD', '.true.')
        self.mpirun('bin/xspecfem3D')

        unix.mv(self.wildcard, 'traces/obs')
        self.export_traces(PATH.OUTPUT, 'traces/obs')


    def generate_mesh(self, model_path=None, model_name=None, model_type='gll'):
        """ Performs meshing and database generation
        """
        raise NotImplementedError('Must be implemented by subclass.')



    ### high-level solver interface

    def eval_func(self, path='', export_traces=False):
        """ Evaluates misfit function by carrying out forward simulation and
            making measurements on observations and synthetics.
        """
        unix.cd(self.getpath)
        self.import_model(path)

        self.forward()
        unix.mv(self.wildcard, 'traces/syn')
        preprocess.prepare_eval_grad(self.getpath)

        self.export_residuals(path)
        if export_traces:
            self.export_traces(path, prefix='traces/syn')


    def eval_grad(self, path='', export_traces=False):
        """ Evaluates gradient by carrying out adjoint simulation. Adjoint traces
            must be in place prior to calling this method.
        """
        unix.cd(self.getpath)

        self.adjoint()

        self.export_kernels(path)
        if export_traces:
            self.export_traces(path, prefix='traces/syn')


    def apply_hess(self, path='', hessian='Newton'):
        """ Evaluates action of Hessian on a given model vector.
        """
        unix.cd(self.getpath)
        self.imprt(path, 'model')

        self.forward()
        unix.mv(self.wildcard, 'traces/lcg')
        preprocess.prepare_apply_hess(self.getpath)
        self.adjoint()

        self.export_kernels(path)



    ### low-level solver interface

    def forward(self):
        """ Calls SPECFEM3D forward solver
        """
        raise NotImplementedError('Must be implemented by subclass.')


    def adjoint(self):
        """ Calls SPECFEM3D adjoint solver
        """
        raise NotImplementedError('Must be implemented by subclass.')



    ### model input/output

    def load(self, dirname, type='model', verbose=False):
        """ reads SPECFEM3D model
        """
        if type == 'model':
            mapping = lambda key: key
        elif type == 'kernel':
            mapping = lambda key: self.kernel_map[key]
        else:
            raise ValueError

        if verbose:
            logfile = PATH.SUBMIT +'/'+ 'output.minmax'
        else:
            logfile = None

        return load(dirname, self.model_parameters, mapping, PAR.NPROC, logfile)


    def save(self, dirname, parts):
        """ writes SPECFEM3D model
        """
        unix.mkdir(dirname)

        # write database files
        for key in self.model_parameters:
            nn = len(parts[key])
            for ii in range(nn):
                filename = 'proc%06d_%s.bin' % (ii, key)
                savebin(parts[key][ii], join(dirname, filename))


    ### vector/dictionary conversion

    def merge(self, parts):
        """ merges dictionary into vector
        """
        v = np.array([])
        for key in self.inversion_parameters:
            for iproc in range(PAR.NPROC):
                v = np.append(v, parts[key][iproc])
        return v


    def split(self, v):
        """ splits vector into dictionary
        """
        parts = {}
        nrow = len(v)/(PAR.NPROC*len(self.inversion_parameters))
        j = 0
        for key in self.model_parameters:
            parts[key] = []
            if key in self.inversion_parameters:
                for i in range(PAR.NPROC):
                    imin = nrow*PAR.NPROC*j + nrow*i
                    imax = nrow*PAR.NPROC*j + nrow*(i + 1)
                    i += 1
                    parts[key].append(v[imin:imax])
                j += 1
            else:
                for i in range(PAR.NPROC):
                    proc = '%06d' % i
                    parts[key].append(
                        np.load(PATH.GLOBAL +'/'+ 'mesh' +'/'+ key +'/'+ proc))
        return parts



    ### postprocessing utilities

    def combine(self, path=''):
        """ combines SPECFEM3D kernels
        """
        unix.cd(self.getpath)

        # create temporary files and directories
        dirs = unix.ls(path)
        with open('kernels_list.txt', 'w') as file:
            file.write('\n'.join(dirs) + '\n')
        unix.mkdir('INPUT_KERNELS')
        unix.mkdir('OUTPUT_SUM')
        for dir in dirs:
            src = path +'/'+ dir
            dst = 'INPUT_KERNELS' +'/'+ dir
            unix.ln(src, dst)

        # sum kernels
        self.mpirun(PATH.SOLVER_BINARIES +'/'+ 'xsum_kernels')
        unix.mv('OUTPUT_SUM', path +'/'+ 'sum')

        # remove temporary files and directories
        unix.rm('INPUT_KERNELS')
        unix.rm('kernels_list.txt')

        unix.cd(path)


    def smooth(self, path='', tag='gradient', span=0.):
        """ smooths SPECFEM3D kernels
        """
        unix.cd(self.getpath)

        # list kernels
        kernels = []
        for name in self.model_parameters:
            if name in self.inversion_parameters:
                flag = True
            else:
                flag = False
            kernels = kernels + [[name, flag]]

        # smooth kernels
        for name, flag in kernels:
            if flag:
                print ' smoothing', name
                self.mpirun(
                    PATH.SOLVER_BINARIES +'/'+ 'xsmooth_sem '
                    + str(span) + ' '
                    + str(span) + ' '
                    + name + ' '
                    + path +'/'+ tag + '/ '
                    + path +'/'+ tag + '/ ')

        # move kernels
        src = path +'/'+ tag
        dst = path +'/'+ tag + '_nosmooth'
        unix.mkdir(dst)
        for name, flag in kernels:
            if flag:
                unix.mv(glob(src+'/*'+name+'.bin'), dst)
            else:
                unix.cp(glob(src+'/*'+name+'.bin'), dst)
        unix.rename('_smooth', '', glob(src+'/*'))
        print ''

        unix.cd(path)



    ### file transfer utilities

    def import_model(self, path):
        src = join(path, 'model')
        dst = self.databases

        if system.getnode()==0:
            self.save(dst, self.load(src, verbose=True))
        else:
            self.save(dst, self.load(src))

    def import_traces(self, path):
        src = glob(join(path, 'traces', self.getname, '*'))
        dst = join(self.getpath, 'traces/obs')
        unix.cp(src, dst)

    def export_model(self, path):
        if system.getnode() == 0:
            for name in self.model_parameters:
                src = glob(join(self.databases, '*_'+name+'.bin'))
                dst = path
                unix.mkdir(dst)
                unix.cp(src, dst)

    def export_kernels(self, path):
        unix.mkdir_gpfs(join(path, 'kernels'))
        unix.mkdir(join(path, 'kernels', self.getname))
        for name in self.kernel_map.values():
            src = join(glob(self.databases  +'/'+ '*'+ name+'.bin'))
            dst = join(path, 'kernels', self.getname)
            unix.mv(src, dst)
        try:
            name = 'rhop_kernel'
            src = join(glob(self.databases +'/'+ '*'+ name+'.bin'))
            dst = join(path, 'kernels', self.getname)
            unix.mv(src, dst)
        except:
            pass

    def export_residuals(self, path):
        unix.mkdir_gpfs(join(path, 'residuals'))
        src = join(self.getpath, 'residuals')
        dst = join(path, 'residuals', self.getname)
        unix.mv(src, dst)

    def export_traces(self, path, prefix='traces/obs'):
        unix.mkdir_gpfs(join(path, 'traces'))
        src = join(self.getpath, prefix)
        dst = join(path, 'traces', self.getname)
        unix.cp(src, dst)


    ### setup utilities

    def initialize_solver_directories(self):
        """ Creates directory structure expected by SPECFEM3D, copies 
          executables, and prepares input files. Executables must be supplied 
          by user as there is currently no mechanism to automatically compile 
          from source.
        """
        unix.mkdir(self.getpath)
        unix.cd(self.getpath)

        # create directory structure
        unix.mkdir('bin')
        unix.mkdir('DATA')

        unix.mkdir('traces/obs')
        unix.mkdir('traces/syn')
        unix.mkdir('traces/adj')

        unix.mkdir(self.databases)

        # copy exectuables
        src = glob(PATH.SOLVER_BINARIES +'/'+ '*')
        dst = 'bin/'
        unix.cp(src, dst)

        # copy input files
        src = glob(PATH.SOLVER_FILES +'/'+ '*')
        dst = 'DATA/'
        unix.cp(src, dst)

        src = 'DATA/' + self.prefix + '_' + self.getname
        dst = 'DATA/' + self.prefix
        unix.cp(src, dst)


    def initialize_adjoint_traces(self):
        """ Adjoint traces must be initialized by writing zeros for all 
          components. This is because when reading traces at the start of an
          adjoint simulation, SPECFEM3D expects that all components exist.
          Components actually in use during an inversion or migration will
          be overwritten with nonzero values later on.
        """
        _, h = preprocess.load('traces/obs')
        zeros = np.zeros((h.nt, h.nr))
        for channel in ['x', 'y', 'z']:
            preprocess.writer(zeros, h, channel=channel, prefix='traces/adj')


    def initialize_io_machinery(self):
        """ Writes mesh files expected by input/output methods
        """
        if system.getnode() == 0:
            parts = self.load(PATH.MODEL_INIT)
            try:
                path = PATH.GLOBAL +'/'+ 'mesh'
            except:
                raise Exception
            if not exists(path):
                for key in self.model_parameters:
                    if key not in self.inversion_parameters:
                        unix.mkdir(path +'/'+ key)
                        for proc in range(PAR.NPROC):
                            with open(path +'/'+ key +'/'+ '%06d' % proc, 'w') as file:
                                np.save(file, parts[key][proc])

            try:
                path = PATH.OPTIMIZE +'/'+ 'm_new'
            except:
                return
            if not exists(path):
                savenpy(path, self.merge(parts))
            #if not exists(path):
            #    for key in inversion_set:
            #        unix.mkdir(path +'/'+ key)
            #        for proc in range(PAR.NPROC):
            #            with open(path +'/'+ key +'/'+ '%06d' % proc, 'w') as file:
            #                np.save(file, parts[key][proc])


    ### input file writers

    def write_parameters(self):
        unix.cd(self.getpath)
        solvertools.write_parameters(vars(PAR))

    def write_receivers(self):
        unix.cd(self.getpath)
        key = 'use_existing_STATIONS'
        val = '.true.'
        solvertools.setpar(key, val)
        _, h = preprocess.load('traces/obs')
        solvertools.write_receivers(h.nr, h.rx, h.rz)

    def write_sources(self):
        unix.cd(self.getpath)
        _, h = preprocess.load(dir='traces/obs')
        solvertools.write_sources(vars(PAR), h)


    ### utility functions

    def mpirun(self, script, output='/dev/null'):
        """ Wrapper for mpirun
        """
        with open(output,'w') as f:
            subprocess.call(
                system.mpiargs() + script,
                shell=True,
                stdout=f)

    @property
    def getname(self):
        """name of current source"""
        if not hasattr(self, 'sources'):
            # generate list of all sources
            paths = glob(PATH.SOLVER_FILES +'/'+ self.prefix + '_*')
            self.sources = []
            for path in paths:
                self.sources += [unix.basename(path).split('_')[-1]]
            self.sources.sort()
        return self.sources[system.getnode()]

    @property
    def getpath(self):
        """path of current source"""
        return join(PATH.SOLVER, self.getname)

    @property
    def wildcard(self):
        raise NotImplementedError('Must be implemented by subclass.')

    @property
    def databases(self):
        return join(self.getpath, 'OUTPUT_FILES/DATABASES_MPI')

    @property
    def prefix(self):
        raise NotImplementedError('Must be implemented by subclass.')




