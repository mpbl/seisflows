
import os
import math
import sys
import subprocess
import time
from os.path import abspath, join

from seisflows.tools import unix
from seisflows.tools.code import saveobj
from seisflows.tools.config import findpath, ConfigObj, ParameterObj

OBJ = ConfigObj('SeisflowsObjects')
PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')


class slurm_lg(object):
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
            raise Exception

        if 'NPROC' not in PAR.General["System"]:
            raise Exception

        if 'NPROC_PER_NODE' not in PAR.General["System"]:
            raise Exception

        if 'WALLTIME' not in PAR.General["System"]:
            PAR.General["System"]["WALLTIME"] = 30.

        if 'STEPTIME' not in PAR:
            setattr(PAR, 'STEPTIME', 30.)

        if 'SLEEPTIME' not in PAR:
            PAR.SLEEPTIME = 1.

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
        """ Submits workflow
        """
        unix.mkdir(PATH.OUTPUT)
        unix.cd(PATH.OUTPUT)
        unix.mkdir(PATH.SUBMIT+'/'+'output.slurm')

        self.save_objects()
        self.save_parameters()
        self.save_paths()

        # prepare sbatch arguments
        args = ('sbatch '
                + '--job-name=%s ' % PAR.TITLE
                + '--output %s ' % (PATH.SUBMIT+'/'+'output.log')
                + '--ntasks-per-node=%d ' % PAR.General["System"]["NPROC_PER_NODE"]
                + '--nodes=%d ' % 1
                + '--time=%d ' % PAR.General["System"]["WALLTIME"]
                + findpath('system') +'/'+ 'slurm/wrapper_sbatch '
                + PATH.OUTPUT)

        subprocess.call(args, shell=1)


    def run(self, classname, funcname, hosts='all', **kwargs):
        """  Runs tasks in serial or parallel on specified hosts.
        """

        self.save_objects()

        self.save_kwargs(classname, funcname, kwargs)
        jobs = self.launch(classname, funcname, hosts)
        while 1:
            time.sleep(60.*PAR.SLEEPTIME)
            self.timestamp()
            isdone, jobs = self.task_status(classname, funcname, jobs)
            if isdone:
                return


    def launch(self, classname, funcname, hosts='all'):
        unix.mkdir(PATH.SYSTEM)

        ntask = PAR.General["System"]["NTASK"]
        nproc = PAR.General["System"]["NPROC"]
        ppn = PAR.General["System"]["NPROC_PER_NODE"]
        walltime = PAR.General["System"]["WALLTIME"]
        
        # Find if system parameters have been overloaded for this class 
        # * Find which class we are dealing with
        #   -> Look for "System" 
        #      -> Look for overloaded parameters
        operation_type = classname.capitalize()
        if operation_type in PAR.keys():
            operation_param = PAR[operation_type]
            if "System" in operation_param: 
                operation_system = operation_param["System"]
                if "NTASK" in operation_system:
                    ntask = operation_system["NTASK"]
                if "NPROC" in operation_system:
                    nproc = operation_system["NPROC"]
                if "NPROC_PER_NODE" in operation_system:
                    ppn = operation_system["NPROC_PER_NODE"]
                if "WALLTIME" in operation_system:
                    walltime = operation_system["WALLTIME"]

        # prepare sbatch arguments
        if hosts == 'all':
            args = ('--array=%d-%d ' % (0, ntask-1)                             
                   +'--output %s ' % (PATH.SUBMIT+'/'+'output.slurm/'+'%A_%a'))

        elif hosts == 'head':
            args = ('--array=%d-%d ' % (0,0)
                   +'--output=%s ' % (PATH.SUBMIT+'/'+'output.slurm/'+'%j'))
            #args = ('--export=SEISFLOWS_TASK_ID=%s ' % 0
            #       +'--get-user-env '
            #       +'--output=%s ' % (PATH.SUBMIT+'/'+'output.slurm/'+'%j'))

        args = ('sbatch '
                + '--job-name=%s ' % PAR.TITLE
                + '--nodes=%d ' % (math.ceil(nproc / float(ppn)))
                + '--ntasks-per-node=%d ' % ppn
                + '--time=%d ' % walltime 
                + args
                + findpath('system') +'/'+ 'slurm/wrapper_srun '
                + PATH.OUTPUT + ' '
                + classname + ' '
                + funcname + ' ')

        if PAR.VERBOSE >= 2:
            print "Launching:..........................."
            print args 
            print ".....................................\n"

        # submit jobs
        with open(PATH.SYSTEM+'/'+'job_id', 'w') as f:
            subprocess.call(args, shell=1, stdout=f)

        # return job ids
        with open(PATH.SYSTEM+'/'+'job_id', 'r') as f:
            line = f.readline()
            job = line.split()[-1].strip()
        if hosts == 'all' and PAR.General["System"]["NTASK"] > 1:
            nn = range(PAR.General["System"]["NTASK"])
            return [job+'_'+str(ii) for ii in nn]
        else:
            return [job]


    def task_status(self, classname, funcname, jobs):
        # query slurm database
        for job in jobs:
            state = self.getstate(job)

            states = []
            if state in ['COMPLETED']:
                states += [1]
            else:
                states += [0]
            if state in ['FAILED', 'NODE_FAIL', 'TIMEOUT']:
                raise Exception

        isdone = all(states)

        return isdone, jobs


    def mpiargs(self):
        return 'srun '

    def getstate(self, jobid):
        """ Retrives job state from SLURM database
        """
        with open(PATH.SYSTEM+'/'+'job_status', 'w') as f:
            subprocess.call('sacct -n -o state -j '+jobid, shell=True, stdout=f)

        with open(PATH.SYSTEM+'/'+'job_status', 'r') as f:
            line = f.readline()
            state = line.strip()

        return state

    def getnode(self):
        """ Gets number of running task
        """
        try:
            return int(os.getenv('SEISFLOWS_TASK_ID'))
        except:
            try:
                return int(os.getenv('SLURM_ARRAY_TASK_ID'))
            except:
                raise Exception("TASK_ID environment variable not defined.")

    def timestamp(self):
        with open(PATH.SYSTEM+'/'+'timestamps', 'a') as f:
            line = time.strftime('%H:%M:%S')+'\n'
            f.write(line)


    ### utility functions

    def save_kwargs(self, classname, funcname, kwargs):
        kwargspath = join(PATH.OUTPUT, 'SeisflowsObjects', classname+'_kwargs')
        kwargsfile = join(kwargspath, funcname+'.p')
        unix.mkdir(kwargspath)
        saveobj(kwargsfile, kwargs)

    def save_objects(self):
        OBJ.save(join(PATH.OUTPUT, 'SeisflowsObjects'))

    def save_parameters(self):
        PAR.save('SeisflowsParameters.json')

    def save_paths(self):
        PATH.save('SeisflowsPaths.json')

