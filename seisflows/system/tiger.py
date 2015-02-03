
from seisflows.tools.config import ConfigObj, ParameterObj, loadclass

OBJ = ConfigObj('SeisflowsObjects')
PAR = ParameterObj('SeisflowsParameters')
PATH = ParameterObj('SeisflowsPaths')


# ensure number of processers per source is defined
if 'NPROC' not in PAR.General["System"]:
    raise Exception

# there are 16 processers per node on tiger
if 'NPROC_PER_NODE' in PAR.General["System"]:
    assert(PAR.General["System"]["NPROC_PER_NODE"] == 16)
else:
    PAR.General["System"]["NPROC_PER_NODE"] = 16

# if nproc per source exceeds nproc per node, use tiger_lg
# otherwise, use tiger_sm
if (PAR.General["System"]["NPROC"] >
    PAR.General["System"]["NPROC_PER_NODE"]):
    tiger = loadclass('system','tiger_lg')
else:
    tiger = loadclass('system','tiger_sm')

