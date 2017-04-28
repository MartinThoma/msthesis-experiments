#!/usr/bin/env python


"""Find all possible pure convolutional architectures with const parameters."""


from copy import deepcopy


def get_params(ns):
    """
    Get the number of parameters of a sequence of convolutional layers.

    All layers have biases and all layers use 3x3 convolutions.
    ns[0] denotes the number of input feature maps, n[i] denotes the number of
    filters of layer i.

    Examples
    --------
    >>> get_params([3, 32])
    896
    """
    params = 0
    for ni1, ni in zip(ns, ns[1:]):
        params += (ni1 * 3**2 + 1) * ni
    return params


def determine_max_depth(max_params, n0, min_width):
    """
    Get the maximum depth of a CNN with a fixed parameter budget.

    Example
    -------
    >>> determine_max_length(896, 3)
    88
    """
    n = [n0]
    params = get_params(n)
    while params < max_params:
        n.append(min_width)
        params = get_params(n)
    return len(n) - 2


def get_possible_layers(queue, max_params, min_params, min_width):
    possible = set([])
    while len(queue) > 0:
        current, pos = queue.pop()
        params = get_params(current)
        if min_params <= params and params <= max_params:
            possible.add(tuple(current))
        if params <= max_params:
            current = deepcopy(current)
            current[pos] += 1
            queue.append((current, pos))
    return possible


def get_all(min_params, max_params, n0, min_width=1):
    possible = set()
    if min_params == 0:
        possible.add([n0])
    max_depth = determine_max_depth(max_params, n0, min_width)
    print("Maximum depth width {} params: {}".format(max_params, max_depth))

    # Each element in the queue is (architecture, position), where
    # at position could be added another filter to architecture. Hence
    # architecture might be under min_params but is not over max_params
    queue = []
    for depth in range(max_depth + 1):
        # Create minimal network of this depth
        minimal = [n0]
        for _ in range(depth):
            minimal.append(min_width)

        params = get_params(minimal)
        if params > max_params:
            continue
        for pos in range(depth + 1):
            if pos == 0:
                continue
            current = deepcopy(minimal)
            queue.append((current, pos))

    possible = get_possible_layers(queue, max_params, min_params, min_width)
    possible = sorted(list(possible), key=lambda n: len(n))
    for i, el in enumerate(possible, start=1):
        print("{}: {} ({} params)".format(i, el, get_params(el)))
    return len(possible)

if __name__ == '__main__':
    print(get_all(800, max_params=896, n0=3, min_width=3))
