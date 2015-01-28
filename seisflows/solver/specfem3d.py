
import subprocess
from glob import glob
from os.path import join

import numpy as np

import seisflows.seistools.specfem3d as solvertools
from seisflows.seistools.shared import load

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import exists, setdiff
from seisflows.tools.config import findpath, loadclass, ParameterObj
from seisflows.tools.io import loadbin, savebin

PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')

import system
import preprocess


class specfem3d(loadclass('solver','base')):
    """ Python interface for SPECFEM3D

      For detailed method descriptions, see base class.
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
        super(specfem3d, self).check()


    def generate_mesh(self, model_path=None, model_name=None, model_type='gll'):
        """ Performs meshing and database generation
        """
        assert(model_name)
        assert(model_type)

        self.initialize_solver_directories()
        unix.cd(self.getpath)

        if model_type == 'gll':
            assert (exists(model_path))
            unix.cp(glob(model_path +'/'+ '*'), self.databases)
        elif model_type == 'sep':
            pass
        elif model_type == 'default':
            pass
        elif model_type == 'tomo':
            pass

        self.mpirun('bin/xmeshfem3D')
        self.mpirun('bin/xgenerate_databases')
        self.export_model(PATH.OUTPUT +'/'+ model_name)


    def forward(self):
        """ Calls SPECFEM3D forward solver
        """
        solvertools.setpar('SIMULATION_TYPE', '1')
        solvertools.setpar('SAVE_FORWARD', '.true.')
        self.mpirun('bin/xgenerate_databases')
        self.mpirun('bin/xspecfem3D')
        unix.mv(self.wildcard, 'traces/syn')


    def adjoint(self):
        """ Calls SPECFEM3D adjoint solver
        """
        solvertools.setpar('SIMULATION_TYPE', '3')
        solvertools.setpar('SAVE_FORWARD', '.false.')
        unix.rm('SEM')
        unix.ln('traces/adj', 'SEM')
        self.mpirun('bin/xspecfem3D')


    @property
    def wildcard(self):
        return glob('OUTPUT_FILES/*SU')

    @property
    def prefix(self):
        return 'FORCESOLUTION'
