"""
Title: relay.py
Purpose: Contains the relay object, which acts as a messenger between
            the agent and the objects that the agent manages.
Notes:
"""

# Global container for method-specific parameters
modeParams = {}


#============================================
#                   Relay
#============================================
class Relay:
    """
    Doc string.

    Attributes:
    -----------
        pass

    Methods:
    --------
        pass
    """
    #-----
    # constructor
    #-----
    def __init(self):
        pass


#============================================
#                  get_mode
#============================================
def get_mode(d, baseKey=''):
    """
    Doc string (see practice/class_attr_adding/add.py)

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    for k, v in d.items():
        if k == 'mode':
            for modeKey, modeVal in v.items():
                if modeVal['enable']:
                    mode = modeKey
                    if len(modeVal.keys()) > 1:
                        modeParams[baseKey] = {}
                        for modeParam, paramValue in modeVal.items():
                            if modeParam != 'enable':
                                mode_params[baseKey][modeParam] = paramValue
                    break
            d[k] = mode
        elif isinstance(v, dict):
            baseKey = k
            d[k] = get_mode(v, baseKey)
        else:
            continue
    return d


#============================================
#                  update
#============================================
def update(d):
    """
    Doc string. NOTE: calling dict.update on a nested dictionary
    doesn't work. As such, this assumes that the deepest nesting
    is two layers. That is, e.g., d['memory']['mode'] doesn't have any
    further modes. If future Jared wants to mess with that kind of
    recursion, be my guest.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    for k, v in d.items():
        if k in mode_params.keys():
            d[k].update(mode_params[k])
    return d


#============================================
#              dict_to_class
#============================================
def recurse(cls, d):
    """
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    for k, v in d.items():
        if isinstance(v, dict):
            r = Relay()
            setattr(cls, k, recurse(r, v))
        else:
            setattr(cls, k, v)
    return cls


#============================================
#               get_new_relay
#============================================
def get_new_relay(clArgs, params):
    """
    Doc string.

    Parameters:
    -----------
        pass

    Raises:
    -------
        pass

    Returns:
    --------
        pass
    """
    # Prune the params dictionary. Having method specific params filed
    # under that method's mode heading is nice for the param file, but
    # miserable for using in the code, so those need to be moved and
    # the chosen mode needs to be set as such
    prunedParams = get_mode(params)
    prunedParams = update(prunedParams)
    # Now convert from the dictionary interface to the class interface,
    # because it's cleaner
    relay = Relay()
    relay = dict_to_class(relay, prunedParams)
    # Now add the clArgs Namespace object as an attribute of relay
    setattr(relay, 'clArgs', clArgs)
    return relay
