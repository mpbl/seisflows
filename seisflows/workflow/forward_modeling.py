from seisflows.tools.config import ConfigObj, ParameterObj

OBJ = ConfigObj('SeisflowsObjects')
PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')

import system
import solver


class forward_modeling(object):
    """ Forward modeling base class
    """

    def check(self):
        """ Checks parameters, paths, and dependencies
        """

        # check paths
        if 'GLOBAL' not in PATH:
            raise Exception

        if 'LOCAL' not in PATH:
            setattr(PATH, 'LOCAL', None)

        # check input settings
        if 'MODEL' not in PATH:
            raise Exception

        # check output settings
        if 'OUTPUT' not in PATH:
            raise Exception

        # check dependencies
        if 'solver' not in OBJ:
            raise Exception("Undefined Exception")

        if 'system' not in OBJ:
            raise Exception("Undefined Exception")


    def main(self):
        """ Generates seismic data
        """

        print 'Running solver...'

        system.run('solver', 'generate_data',
                   hosts='all',
                   model_path=PATH.MODEL,
                   model_type='gll',
                   model_name='model')

        print "Finished"
