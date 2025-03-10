
from SeqSNN.network.snn import TSLIF_base
import numpy as np
def reset_states(model):
    for m in model.modules():
        if hasattr(m, 'reset'):
            if not isinstance(m, TSLIF_base.MemoryModule):
                print(f'Trying to call `reset()` of {m}, which is not base.MemoryModule')
            m.reset()




