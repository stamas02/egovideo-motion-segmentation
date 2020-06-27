def segment(observations):
    """ Segment a sequence of observations.

    Given a sequence of observations extract the start and end index of consecutive observations.

    Example:
    --------
    >>> observations = [1,1,1,2,2,2,1,1,4,4,4]
    >>> segments = segment(observations)
    >>> print(segmetns)
    result:
    >>> segments = {'1':[[0,2],[6,7]],
    >>>             '2':[[3,5]],
    >>>             '4':[8,10]}

    Parameters
    ----------
    observations: list,
        A list of observations.

    Returns
    -------
        A dictionary where each key represents a unique type of observations and the values
        are a list of [start, end] style list representing the satrt end end frame of a segment with
        the belonging observation.

    """
    segmetns = {}
    p_o = observations[0]
    segmetns[p_o] = [[0]]
    for i, o in enumerate(observations):
        if p_o != o:
            segmetns[p_o][-1].append(i-1)
            if o not in segmetns.keys():
                segmetns[o] = []
            segmetns[o].append([i])
            p_o = o

    segmetns[p_o][-1].append(i)
    return segmetns


