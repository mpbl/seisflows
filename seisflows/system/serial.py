from os.path import abspath, join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.config import ConfigObj, ParameterObj

OBJ = ConfigObj('SeisflowsObjects')
PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')

save_objects = OBJ.save
save_parameters = PAR.save
save_paths = PATH.save


class serial(object):
    """ An interface through which to submit workflows, run tasks in serial or 
      parallel, and perform other system functions.

      By hiding environment details behind a python interface layer, these 
      classes provide a consistent command set across different computing
      environments.

      For more informations, see 
      http://seisflows.readthedocs.org/en/latest/manual/manual.html#system-interfaces
    """

    def check(self):
        """ Checks parameters and paths
        """

        if 'TITLE' not in PAR:
            setattr(PAR, 'TITLE', unix.basename(abspath('.')))

        if 'SUBTITLE' not in PAR:
            setattr(PAR, 'SUBTITLE', unix.basename(abspath('..')))

        # check parameters
        if 'NTASK' not in PAR.General["System"]:
            Par.General["System"]["NTASK"] = 1

        if 'NPROC' not in PAR.General["System"]:
            PAR.Genera["Sytem"]["NPROC"] = 1

        if 'VERBOSE' not in PAR:
            setattr(PAR, 'VERBOSE', 1)

        # check paths
        if 'GLOBAL' not in PATH:
            setattr(PATH, 'GLOBAL', join(abspath('.'), 'scratch'))

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', '')

        if 'SUBMIT' not in PATH:
            setattr(PATH, 'SUBMIT', unix.pwd())

        if 'OUTPUT' not in PATH:
            setattr(PATH, 'OUTPUT', join(PATH.SUBMIT, 'output'))

        if 'SYSTEM' not in PATH:
            setattr(PATH, 'SYSTEM', join(PATH.GLOBAL, 'system'))


    def submit(self, workflow):
        """ Submits job
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)

        # save current state
        save_parameters('SeisflowsParameters.json')
        save_paths('SeisflowsPaths.json')

        workflow.main()


    def run(self, classname, funcname, hosts='all', **kwargs):
        """ Runs tasks in serial or parallel on specified hosts
        """
        unix.mkdir(PATH.SYSTEM)

        if hosts == 'all':
            for itask in range(PAR.General["System"]["NTASK"]):
                self.setnode(itask)
                self.progress(itask)
                func = getattr(__import__(classname), funcname)
                func(**kwargs)
            print ''

        elif hosts == 'head':
            self.setnode(0)
            func = getattr(__import__(classname), funcname)
            func(**kwargs)

        else:
            task(**kwargs)


    def getnode(self):
        """Gets number of running task"""
        return int(np.loadtxt(PATH.SYSTEM + '/' + 'nodenum'))

    def setnode(self, itask):
        """Sets number of running task"""
        np.savetxt(PATH.SYSTEM + '/' + 'nodenum', [itask])

    def mpiexec(self):
        """Wrapper for mpiexec"""
        return 'mpiexec -np %d '%PAR.General["System"]["NPROC"]

    def progress(self, itask=None):
        """Prints status updates"""
        if PAR.VERBOSE and PAR.General["System"]["NTASK"] > 1:
            print ' task ' + '%02d'%(itask + 1) + ' of ' 
                  + '%02d'%PAR.General["System"]["NTASK"]
