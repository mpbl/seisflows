
import numpy as np

from seisflows.tools import unix
from seisflows.tools.code import Struct
from seisflows.tools.config import ParameterObj

from seisflows.seistools import adjoint, misfit, sbandpass, smute, readers, writers

PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')


class base(object):
    """ Data preprocessing class
    """

    def check(self):
        """ Checks parameters, paths, and dependencies
        """

        if 'MISFIT' not in PAR.Preprocessing:
            Par.Preprocessing['MISFIT'] = 'wav'
        if 'FORMAT' not in PAR.Preprocessing:
            raise Exception
        if 'CHANNELS' not in PAR.Preprocessing:
            raise Exception
        if 'NORMALIZE' not in PAR.Preprocessing:
            Par.Preprocessing['NORMALIZE'] = True

        # mute settings
        if 'MUTE' not in PAR.Preprocessing:
            PAR.Preprocessing['MUTE'] = False
        if 'MUTESLOPE' not in PAR.Preprocessing:
            PAR.Preprocessing['MUTESLOPE'] = 0.
        if 'MUTECONST' not in PAR.Preprocessing:
            PAR.Preprocessing['MUTECONST'] = 0.

        # filter settings
        if 'BANDPASS' not in PAR.Preprocessing:
            PAR.Preprocessing['BANDPASS'] = False
        if 'HIGHPASS' not in PAR.Preprocessing:
            PAR.Preprocessing['HIGHPASS'] = False
        if 'LOWPASS' not in PAR.Preprocessing:
            PAR.Preprocessing['LOWPASS'] = False
        if 'FREQLO' not in PAR.Preprocessing:
            PAR.Preprocessing['FREQLO'] = 0.
        if 'FREQHI' not in PAR.Preprocessing:
            PAR.Preprocessing['FREQHI'] = 0.


    def setup(self):
        """ Performs any required setup tasks

          Called at beginning of an inversion, prior to any model update 
          iterations.
        """
        self.reader = getattr(readers, PAR.Preprocessing["FORMAT"])
        self.writer = getattr(writers, PAR.Preprocessing["FORMAT"])
        self.channels = [char for char in PAR.Preprocessing["CHANNELS"]]


    def prepare_eval_grad(self, path='.'):
        """ Prepares solver for gradient evaluation by writing residuals and
          adjoint traces
        """
        unix.cd(path)

        d, h = self.load(prefix='traces/obs/')
        s, _ = self.load(prefix='traces/syn/')

        d = self.apply(self.process_traces, [d], [h])
        s = self.apply(self.process_traces, [s], [h])

        self.apply(self.write_residuals, [s, d], [h], inplace=False)

        s = self.apply(self.generate_adjoint_traces, [s, d], [h])
        self.save(s, h, prefix='traces/adj/')


    def process_traces(self, s, h):
        """ Performs data processing operations on traces
        """
        # filter data
        if PAR.Preprocessing["BANDPASS"]:
            s = sbandpass(s, h, PAR.Preprocessing["FREQLO"], PAR.Preprocessing["FREQHI"])

        if PAR.Preprocessing["HIGHPASS"]:
            s = shighpass(s, h, PAR.Preprocessing["FREQLO"])

        if PAR.Preprocessing["HIGHPASS"]:
            s = slowpass(s, h, PAR.Preprocessing["FREQHI"])

        # mute direct arrival
        if PAR.Preprocessing["MUTE"] == 1:
            vel = PAR.Preprocessing["MUTESLOPE"]
            off = PAR.Preprocessing["MUTECONST"]
            s = smute(s, h, vel, off, constant_spacing=False)

        elif PAR.Preprocessing["MUTE"] == 2:
            import system
            vel = PAR.Preprocessing["MUTESLOPE"]*(PAR.Workflow["NREC"] + 1) \
                / (PAR.Solver["XMAX"] - PAR.Solver["XMIN"])
            off = PAR.Preprocessing["MUTECONST"]
            src = system.getnode()
            s = smute(s, h, vel, off, src, constant_spacing=True)

        return s


    def write_residuals(self, s, d, h):
        """ Computes residuals from observations and synthetics
        """
        r = []
        for i in range(h.nr):
            r.append(self.call_misfit(s[:,i], d[:,i], h.nt, h.dt))

        # write residuals
        np.savetxt('residuals', r)

        return np.array(r)


    def generate_adjoint_traces(self, s, d, h):
        """ Generates adjoint traces from observed and synthetic traces
        """
        # generate adjoint traces
        for i in range(h.nr):
            s[:,i] = self.call_adjoint(s[:,i], d[:,i], h.nt, h.dt)

        # normalize traces
        if PAR.Preprocessing["NORMALIZE"]:
            for ir in range(h.nr):
                w = np.linalg.norm(d[:,ir], ord=2)
                if w > 0: 
                    s[:,ir] /= w

        return s


    ### misfit/adjoint wrappers

    def call_adjoint(self, wsyn, wobs, nt, dt):
        """ Wrapper for generating adjoint traces
        """
        if PAR.Preprocessing["MISFIT"] in ['wav', 'wdiff']:
            # waveform difference
            w = adjoint.wdiff(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['tt', 'wtime']:
            # traveltime
            w = adjoint.wtime(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['ampl', 'wampl']:
            # amplitude
            w = adjoint.wampl(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['env', 'ediff']:
            # envelope
            w = adjoint.ediff(wsyn, wobs, nt, dt, eps=0.05)
        elif PAR.Preprocessing["MISFIT"] in ['cdiff']:
            # cross correlation
            w = adjoint.cdiff(wsyn, wobs, nt, dt)
        else:
            w = wobs
        return w

    def call_misfit(self, wsyn, wobs, nt, dt):
        """ Wrapper for evaluating misfit function
        """
        if PAR.Preprocessing["MISFIT"] in ['wav', 'wdiff']:
            # waveform difference
            e = misfit.wdiff(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['tt', 'wtime']:
            # traveltime
            e = misfit.wtime(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['ampl', 'wampl']:
            # amplitude
            e = misfit.wampl(wsyn, wobs, nt, dt)
        elif PAR.Preprocessing["MISFIT"] in ['env', 'ediff']:
            # envelope
            e = misfit.ediff(wsyn, wobs, nt, dt, eps=0.05)
        elif PAR.Preprocessing["MISFIT"] in ['cdiff']:
            # cross correlation
            e = misfit.cdiff(wsyn, wobs, nt, dt)
        else:
            e = 0.
        return float(e)


    ### input/output

    def load(self, prefix=''):
        """ Reads seismic data from disk
        """
        h = Struct()
        f = Struct()

        for channel in self.channels:
            f[channel], h[channel] = self.reader(prefix=prefix, channel=channel)

        # check headers
        h = self.check_headers(h)

        return f, h

    def save(self, s, h, prefix='traces/adj/', suffix=''):
        """ Writes seismic data to disk
        """
        for channel in self.channels:
            self.writer(s[channel], h, channel=channel, prefix=prefix, suffix=suffix)


    ### utility functions

    def apply(self, func, arrays, input, inplace=True):
        """ Applies function to multi-component data
        """
        if inplace:
            output = Struct(arrays[0])
        else:
            output = Struct()

        for channel in self.channels:
            args = [array[channel] for array in arrays] + input
            output[channel] = func(*args)

        return output

    def check_headers(self, headers):
        """ Checks headers for consistency
        """
        h = headers.values()[0]

        if 'DT' in PAR.Solver:
            if h.dt != PAR.Solver["DT"]:
                h.dt = PAR.Solver["DT"]

        if 'NT' in PAR.Solver:
            if h.nt != PAR.Solver["NT"]:
                print 'Warning: h.nt != PAR.Solver["NT"]'

        if 'NREC' in PAR.Workflow:
            if h.nr != PAR.Workflow["NREC"]:
                print 'Warning: h.nr != PAR.Workflow["NREC"]'

        return h


