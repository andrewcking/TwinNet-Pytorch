"""
HELPER FUNCTIONS
"""
def get_n_params(model):
    """
    Get the number of parameters in model
    :param model: pytorch model
    :return: number of parameters
    """
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp
