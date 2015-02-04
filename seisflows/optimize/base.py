
from os.path import join

import numpy as np

from seisflows.tools import unix
from seisflows.tools.array import loadnpy, savenpy
from seisflows.tools.code import loadtxt, savetxt
from seisflows.tools.config import ParameterObj
from seisflows.tools.io import OutputWriter
from seisflows.optimize import lib

PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')


class base(object):
    """ Nonlinear optimization base class.

     Available nonlinear optimization algorithms include gradient descent,
     nonlinear conjugate gradient, and a quasi-Newton method
     (limited-memory BFGS).

     Available line search algorithms include a backtracking line search based
     on quadratic interpolation and a bracketing and interpolation procedure
     (abbreviated as 'Backtrack' and 'Bracket' respectively.)

     To reduce memory overhead, input vectors are read from the directory
     cls.path rather than passed from a calling routine. At the start of each
     search direction computation, the current model and gradient are read from
     'm_new' and 'g_new'; the resulting search direction is written to 'p_new'.
     As the optimization procedure progresses, other information is stored in
     the cls.path directory.
    """

    def check(cls):
        """ Checks parameters, paths, and dependencies
        """
        # check parameters
        if 'BEGIN' not in PAR.Workflow:
            raise Exception
        if 'END' not in PAR.Workflow:
            raise Exception

        # check paths
        if 'SUBMIT' not in PATH:
            raise Exception

        if 'OPTIMIZE' not in PATH:
            setattr(PATH, 'OPTIMIZE', join(PATH.GLOBAL, 'optimize'))

        # search direction parameters
        if 'SCHEME' not in PAR.Optimization:
            PAR.Optimization['SCHEME'] = 'QuasiNewton'

        if 'NLCGMAX' not in PAR.Optimization:
            PAR.Optimization['NLCGMAX'] = 10

        if 'NLCGTHRESH' not in PAR.Optimization:
            PAR.Optimization['NLCGTHRESH'] = 0.5

        if 'LBFGSMAX' not in PAR.Optimization:
            PAR.Optimization['LBFGSMAX'] = 6

        # line search parameters
        if 'SRCHTYPE' not in PAR.Optimization:
            PAR.Optimization['SRCHTYPE'] = 'Backtrack'

        if 'SRCHMAX' not in PAR.Optimization:
            PAR.Optimization['SRCHMAX'] = 10

        if 'STEPLEN' not in PAR.Optimization:
            PAR.Optimization['STEPLEN'] = 0.05

        if 'STEPMAX' not in PAR.Optimization:
            PAR.Optimization['STEPMAX'] = 0.

        if 'ADHOCSCALING' not in PAR.Optimization:
            PAR.Optimization['ADHOCSCALING'] = 0.



    def setup(cls):
        """ Sets up directory in which to store optimization vectors
        """
        cls.path = PATH.OPTIMIZE
        unix.mkdir(cls.path)

        # prepare algorithm machinery
        if PAR.Optimization["SCHEME"] in ['ConjugateGradient']:
            cls.NLCG = lib.NLCG(cls.path, PAR.NLCGTHRESH, PAR.NLCGMAX)

        elif PAR.Optimization["SCHEME"] in ['QuasiNewton']:
            cls.LBFGS = lib.LBFGS(cls.path, PAR.Optimization["LBFGSMAX"],
                                  PAR.Workflow["BEGIN"])

        # prepare output writer
        cls.writer = OutputWriter(PATH.SUBMIT + '/' + 'output.optim',
            ['iter', 'step', 'misfit'])


    ### search direction methods

    def compute_direction(cls):
        """ Computes model update direction from stored function and gradient 
          values
        """
        unix.cd(cls.path)
        m_new = loadnpy('m_new')
        f_new = loadtxt('f_new')
        g_new = loadnpy('g_new')

        if PAR.Optimization["SCHEME"] == 'GradientDescent':
            p_new = -g_new

        elif PAR.Optimization["SCHEME"] == 'ConjugateGradient':
            # compute NLCG udpate
            p_new = cls.NLCG.compute()

        elif PAR.Optimization["SCHEME"] == 'QuasiNewton':
            # compute L-BFGS update
            if cls.iter == 1:
                p_new = -g_new
            else:
                cls.LBFGS.update()
                p_new = -cls.LBFGS.solve()

        # save results
        unix.cd(cls.path)
        savenpy('p_new', p_new)
        savetxt('s_new', np.dot(g_new, p_new))


    ### line search methods

    def initialize_search(cls):
        """ Determines initial step length for line search
        """
        unix.cd(cls.path)
        if cls.iter == 1:
            s_new = loadtxt('s_new')
            f_new = loadtxt('f_new')
        else:
            s_old = loadtxt('s_old')
            s_new = loadtxt('s_new')
            f_old = loadtxt('f_old')
            f_new = loadtxt('f_new')
            alpha = loadtxt('alpha')

        m = loadnpy('m_new')
        p = loadnpy('p_new')

        # reset search history
        cls.search_history = [[0., f_new]]
        cls.isdone = 0
        cls.isbest = 0
        cls.isbrak = 0

        # determine initial step length
        len_m = max(abs(m))
        len_d = max(abs(p))
        cls.step_ratio = float(len_m/len_d)

        if cls.iter == 1:
            assert PAR.Optimization["STEPLEN"] != 0.
            alpha = PAR.Optimization["STEPLEN"]*cls.step_ratio
        elif PAR.Optimization["SRCHTYPE"] in ['Bracket']:
            alpha *= 2.*s_old/s_new
        elif PAR.Optimization["SCHEME"] in ['GradientDescent', 'ConjugateGradient']:
            alpha *= 2.*s_old/s_new
        else:
            alpha = 1.

        # ad hoc scaling
        if PAR.Optimization["ADHOCSCALING"]:
            alpha *= PAR.Optimization["ADHOCSCALING"]

        # limit maximum step length
        if PAR.Optimization["STEPMAX"] > 0.:
            if alpha/cls.step_ratio > PAR.Optimization["STEPMAX"]:
                alpha = PAR.Optimization["STEPMAX"]*cls.step_ratio

        # write trial model
        savenpy('m_try', m + p*alpha)
        savetxt('alpha', alpha)

        cls.writer(cls.iter, 0., f_new)


    def search_status(cls):
        """ Determines status of line search
        """
        unix.cd(cls.path)
        f0 = loadtxt('f_new')
        g0 = loadtxt('s_new')
        x_ = loadtxt('alpha')
        f_ = loadtxt('f_try')

        if np.isnan(f_):
            raise ValueError

        cls.search_history += [[x_, f_]]

        x = cls.step_lens()
        f = cls.func_vals()

        # is current step length the best so far?
        vals = cls.func_vals(sort=False)
        if np.all(vals[-1] < vals[:-1]):
            cls.isbest = 1

        # are stopping criteria satisfied?
        if PAR.Optimization["SRCHTYPE"] == 'Backtrack':
            if any(f[1:] < f[0]):
                cls.isdone = 1

        elif PAR.Optimization["SRCHTYPE"] == 'Bracket':
            if cls.isbrak:
                cls.isbest = 1
                cls.isdone = 1
            elif any(f[1:] < f[0]) and (f[-2] < f[-1]):
                cls.isbrak = 1

        elif PAR.Optimization["SRCHTYPE"] == 'Fixed':
            if any(f[1:] < f[0]) and (f[-2] < f[-1]):
                cls.isdone = 1

        cls.writer([], x_, f_)

        return cls.isdone, cls.isbest


    def compute_step(cls):
        """ Computes next trial step length
        """
        unix.cd(cls.path)
        m0 = loadnpy('m_new')
        p = loadnpy('p_new')
        f0 = loadtxt('f_new')
        g0 = loadtxt('s_new')

        x = cls.step_lens()
        f = cls.func_vals()

        # compute trial step length
        if PAR.Optimization["SRCHTYPE"] == 'Backtrack':
            alpha = lib.backtrack2(f0, g0, x[1], f[1], b1=0.1, b2=0.5)

        elif PAR.Optimization["SRCHTYPE"] == 'Bracket':
            FACTOR = 2.
            if any(f[1:] < f[0]) and (f[-2] < f[-1]):
                alpha = lib.polyfit2(x, f)
            elif any(f[1:] < f[0]):
                alpha = loadtxt('alpha')*FACTOR
            else:
                alpha = loadtxt('alpha')*FACTOR**-1

        elif PAR.Optimization["SRCHTYPE"] == 'Fixed':
            alpha = cls.step_ratio*(step + 1)*PAR.Optimization["STEPLEN"]

        else:
            raise ValueError

        # write trial model
        savetxt('alpha', alpha)
        savenpy('m_try', m0 + p*alpha)


    def finalize_search(cls):
        """ Cleans working directory and writes updated model
        """
        unix.cd(cls.path)
        m0 = loadnpy('m_new')
        p = loadnpy('p_new')

        x = cls.step_lens()
        f = cls.func_vals()

        # clean working directory
        unix.rm('alpha')
        unix.rm('m_try')
        unix.rm('f_try')

        if cls.iter > 1:
            unix.rm('m_old')
            unix.rm('f_old')
            unix.rm('g_old')
            unix.rm('p_old')
            unix.rm('s_old')

        unix.mv('m_new', 'm_old')
        unix.mv('f_new', 'f_old')
        unix.mv('g_new', 'g_old')
        unix.mv('p_new', 'p_old')
        unix.mv('s_new', 's_old')

        # write updated model
        alpha = x[f.argmin()]
        savetxt('alpha', alpha)
        savenpy('m_new', m0 + p*alpha)
        savetxt('f_new', f.min())

        cls.writer([], [], [])


    ### line search utilities

    def step_lens(cls, sort=True):
        x, f = zip(*cls.search_history)
        x = np.array(x)
        f = np.array(f)
        f_sorted = f[abs(x).argsort()]
        x_sorted = x[abs(x).argsort()]
        if sort:
            return x_sorted
        else:
            return x


    def func_vals(cls, sort=True):
        x, f = zip(*cls.search_history)
        x = np.array(x)
        f = np.array(f)
        f_sorted = f[abs(x).argsort()]
        x_sorted = x[abs(x).argsort()]
        if sort:
            return f_sorted
        else:
            return f
