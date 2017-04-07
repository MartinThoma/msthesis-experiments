#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimize confusion matrix.

For more information, see

* http://cs.stackexchange.com/q/70627/2914
* http://datascience.stackexchange.com/q/17079/8820
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
random.seed(0)
import logging
import sys
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import silhouette_score

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


def get_accuracy(cm):
    """Get the accuaracy by the confusion matrix cm."""
    return float(sum([cm[i][i] for i in range(len(cm))])) / float(cm.sum())


def calculate_score(cm, weights):
    """
    Calculate a score how close big elements of cm are to the diagonal.

    Examples
    --------
    >>> cm = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> weights = calculate_weight_matrix(3)
    >>> calculate_score(cm, weights)
    32
    """
    return int(np.tensordot(cm, weights, axes=((0, 1), (0, 1))))


def calculate_weight_matrix(n):
    """
    Calculate the weights for each position.

    The weight is the distance to the diagonal.
    """
    weights = np.abs(np.arange(n) - np.arange(n)[:, None])
    return weights


def swap(cm, i, j):
    """
    Swap row and column i and j in-place.

    Examples
    --------
    >>> cm = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> swap(cm, 2, 0)
    array([[8, 7, 6],
           [5, 4, 3],
           [2, 1, 0]])
    """
    # swap columns
    copy = cm[:, i].copy()
    cm[:, i] = cm[:, j]
    cm[:, j] = copy
    # swap rows
    copy = cm[i, :].copy()
    cm[i, :] = cm[j, :]
    cm[j, :] = copy
    return cm


def move_1d(perm, from_start, from_end, insert_pos):
    """Move a block in a list."""
    assert insert_pos < from_start or insert_pos > from_end
    if insert_pos > from_end:
        p_new = (list(range(from_end + 1, insert_pos + 1)) +
                 list(range(from_start, from_end + 1)))
    else:
        p_new = (list(range(from_start, from_end + 1)) +
                 list(range(insert_pos, from_start)))
    p_old = sorted(p_new)
    perm[p_old] = perm[p_new]
    return perm


def move(cm, from_start, from_end, insert_pos):
    """
    Move rows from_start - from_end to insert_pos in-place.

    Examples
    --------
    >>> cm = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1], [2, 3, 4, 5]])
    >>> move(cm, 1, 2, 0)
    array([[5, 6, 4, 7],
           [9, 0, 8, 1],
           [1, 2, 0, 3],
           [3, 4, 2, 5]])
    """
    assert insert_pos < from_start or insert_pos > from_end
    if insert_pos > from_end:
        p_new = (list(range(from_end + 1, insert_pos + 1)) +
                 list(range(from_start, from_end + 1)))
    else:
        p_new = (list(range(from_start, from_end + 1)) +
                 list(range(insert_pos, from_start)))
    # print(p_new)
    p_old = sorted(p_new)
    # swap columns
    cm[:, p_old] = cm[:, p_new]
    # swap rows
    cm[p_old, :] = cm[p_new, :]
    return cm


def swap_1d(perm, i, j):
    """
    Swap two elements of a 1-D numpy array in-place.

    Examples
    --------
    >>> perm = np.array([2, 1, 2, 3, 4, 5, 6])
    >>> swap_1d(perm, 2, 6)
    array([2, 1, 6, 3, 4, 5, 2])
    """
    perm[i], perm[j] = perm[j], perm[i]
    return perm


def apply_permutation(cm, perm):
    """
    Apply permutation to a matrix.

    Examples
    --------
    >>> cm = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    >>> perm = np.array([2, 0, 1])
    >>> apply_permutation(cm, perm)
    array([[8, 6, 7],
           [2, 0, 1],
           [5, 3, 4]])
    """
    return cm[perm].transpose()[perm].transpose()


def simulated_annealing(current_cm,
                        current_perm=None,
                        score=calculate_score,
                        steps=2 * 10**5,
                        temp=100.0,
                        cooling_factor=0.99,
                        deterministic=False):
    """
    Optimize current_cm by randomly swapping elements.

    Parameters
    ----------
    current_cm : numpy array
    current_perm : None or iterable, optional (default: None)
    steps : int, optional (default: 2 * 10**4)
    temp : float > 0.0, optional (default: 100.0)
        Temperature
    cooling_factor: float in (0, 1), optional (default: 0.99)
    """
    assert temp > 0.0
    assert cooling_factor > 0.0
    assert cooling_factor < 1.0
    n = len(current_cm)

    # Load the initial permutation
    if current_perm is None:
        current_perm = list(range(n))
    current_perm = np.array(current_perm)

    # Pre-calculate weights
    weights = calculate_weight_matrix(n)

    # Apply the permutation
    current_cm = apply_permutation(current_cm, current_perm)
    current_score = score(current_cm, weights)

    best_cm = current_cm
    best_score = current_score
    best_perm = current_perm

    print("## Starting Score: {:0.2f}%".format(current_score))
    for step in range(steps):
        tmp_cm = np.array(current_cm, copy=True)

        swap_prob = 0.5
        make_swap = random.random() < swap_prob
        if make_swap:
            # Choose what to swap
            i = random.randint(0, n - 1)
            j = i
            while j == i:
                j = random.randint(0, n - 1)
            # Define permutation
            perm = swap_1d(current_perm.copy(), i, j)
            # Define values after swap
            tmp_cm = swap(tmp_cm, i, j)
        else:
            block_len = n
            while block_len >= n - 1:
                from_start = random.randint(0, n - 3)
                from_end = random.randint(from_start + 1, n - 2)
                block_len = from_start - from_end
            insert_pos = from_start
            while not (insert_pos < from_start or insert_pos > from_end):
                insert_pos = random.randint(0, n - 1)
            perm = move_1d(current_perm.copy(),
                           from_start, from_end, insert_pos)

            # Define values after swap
            tmp_cm = move(tmp_cm, from_start, from_end, insert_pos)
        tmp_score = score(tmp_cm, weights)

        # Should be swapped?
        if deterministic:
            chance = 1.0
        else:
            chance = random.random()
            temp *= 0.99
        hot_prob_thresh = min(1, np.exp(-(tmp_score - current_score) / temp))
        if chance <= hot_prob_thresh:
            changed = False
            if best_score > tmp_score:  # minimize
                best_perm = perm
                best_cm = tmp_cm
                best_score = tmp_score
                changed = True
            current_score = tmp_score
            current_cm = tmp_cm
            current_perm = perm
            if changed:
                logging.info(("Current: %0.2f (best: %0.2f, "
                              "hot_prob_thresh=%0.4f%%, step=%i, swap=%s)"),
                             current_score,
                             best_score,
                             (hot_prob_thresh * 100),
                             step,
                             str(make_swap))
    return {'cm': best_cm, 'perm': best_perm}


def plot_cm(cm, zero_diagonal=False, labels=None):
    """Plot a confusion matrix."""
    n = len(cm)
    if zero_diagonal:
        for i in range(n):
            cm[i][i] = 0
    if n > 20:
        size = int(n / 4.)
    else:
        size = 5
    fig = plt.figure(figsize=(size, size), dpi=80, )
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    if labels is None:
        labels = [i for i in range(len(cm))]
    x = [i for i in range(len(cm))]
    plt.xticks(x, labels, rotation='vertical')
    y = [i for i in range(len(cm))]
    plt.yticks(y, labels)  # , rotation='vertical'
    res = ax.imshow(np.array(cm), cmap=plt.cm.viridis,
                    interpolation='nearest')
    width, height = cm.shape

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(res, cax=cax)
    plt.tight_layout()

    plt.savefig('confusion_matrix.tmp.png', format='png')


def extract_clusters(cm, labels, steps=10**4, lambda_=0.013):
    """
    Find clusters in cm.

    Parameters
    ----------
    lambda_ : float
        The closer to 0, the more groups
        The bigger, the bigger groups

    Idea:
        mininmize lambda (error between clusters) - (count of clusters)
        s.t.: Each inter-cluster accuracy has to be lower than the overall
              accuracy
    """
    def create_weight_matrix(grouping):
        n = len(grouping) + 1
        weight_matrix = np.zeros((n, n))
        for i in range(n):
            seen_1 = False
            for j in range(i + 1, n):
                if seen_1:
                    weight_matrix[i][j] = 1
                elif grouping[j - 1] == 1:
                    seen_1 = True
                    weight_matrix[i][j] = 1
        return weight_matrix + weight_matrix.transpose()

    def get_score(cm, grouping, lambda_):
        inter_cluster_err = 0.0
        weights = create_weight_matrix(grouping)
        inter_cluster_err = calculate_score(cm, weights)
        return lambda_ * inter_cluster_err - sum(grouping)

    def find_thres(cm, percentage):
        """
        Find a threshold for grouping.

        Parameters
        ----------
        cm : numpy array
        percentage : float
            Probability that two neighboring classes belong togehter
        """
        n = int(len(cm) * (1.0 - percentage))
        con = sorted(get_neighboring_connectivity(cm))
        return con[n]

    def find_thres_interactive(cm, labels):
        """
        Find a threshold for grouping.

        The threshold is the minimum connection strength for two classes to be
        within the same cluster.

        Parameters
        ----------
        cm : numpy array
        percentage : float
            Probability that two neighboring classes belong togehter
        """
        n = len(cm)
        con = sorted(zip(get_neighboring_connectivity(cm),
                         zip(range(n - 1), range(1, n))))
        # pos_low = 0
        pos_str = None

        # Lowest position from which we know that they are connected
        pos_up = n - 1

        # Highest position from which we know that they are not connected
        neg_low = 0
        # neg_up = n - 1
        while pos_up - 1 > neg_low:
            print("pos_up={}, neg_low={}, pos_str={}"
                  .format(pos_up, neg_low, pos_str))
            pos = int((pos_up + neg_low) / 2)
            con_str, (i1, i2) = con[pos]
            should_be_conn = raw_input('Should {} and {} be in one cluster?'
                                       ' (y/n): '
                                       .format(labels[i1], labels[i2]))
            if should_be_conn == 'n':
                neg_low = pos
            elif should_be_conn == 'y':
                pos_up = pos
                pos_str = con_str
            else:
                print("Please type only 'y' or 'n'. You typed {}."
                      .format(should_be_conn))
        return pos_str

    def get_neighboring_connectivity(cm):
        con = []
        n = len(cm)
        for i in range(n - 1):
            con.append(cm[i][i + 1] + cm[i + 1][i])
        return con

    def split_at_con_thres(cm, thres, labels, interactive):
        """
        Two classes are not in the same group if they are not connected strong.

        Minimum connection strength is thres. The bigger this value, the more
        clusters / the smaller clusters you will get.
        """
        con = get_neighboring_connectivity(cm)
        grouping = []
        for i, el in enumerate(con):
            if el == thres and interactive:
                should_conn = '-'
                while should_conn not in ['y', 'n']:
                    should_conn = raw_input('Should {} and {} be in one '
                                            'cluster? (y/n): '
                                            .format(labels[i], labels[i + 1]))
                    if should_conn == 'y':
                        grouping.append(0)
                    elif should_conn == 'n':
                        grouping.append(1)
                    else:
                        print("please type either 'y' or 'n'")
            else:
                grouping.append(el < thres)
        return grouping

    method = 'local-connectivity'
    interactive = False

    if method == 'energy':
        n = len(cm)
        grouping = np.zeros(n - 1)
        minimal_score = get_score(cm, grouping, lambda_)
        best_grouping = grouping.copy()
        for i in range(steps):
            pos = random.randint(0, n - 2)
            grouping = best_grouping.copy()
            grouping[pos] = (grouping[pos] + 1) % 2
            current_score = get_score(cm, grouping, lambda_)
            if current_score < minimal_score:
                best_grouping = grouping
                minimal_score = current_score
                logging.info("Best grouping: {} (score: {})"
                             .format(grouping, minimal_score))
    elif method == 'local-connectivity':
        if interactive:
            thres = find_thres_interactive(cm, labels)
        else:
            thres = find_thres(cm, 0.1)  # hyper-parameter
        logging.info("Found threshold for local connection: {}".format(thres))
        best_grouping = split_at_con_thres(cm, thres, labels,
                                           interactive=interactive)
    logging.info("Found {} clusters".format(sum(best_grouping) + 1))
    return best_grouping


def apply_grouping(labels, grouping):
    """Return list of grouped labels."""
    groups = []
    current_group = [labels[0]]
    for label, cut in zip(labels[1:], grouping):
        if cut:
            groups.append(current_group)
            current_group = [label]
        else:
            current_group.append(label)
    groups.append(current_group)
    return groups


def _remove_single_element_groups(hierarchy):
    h_new = []
    for el in hierarchy:
        if len(el) > 1:
            h_new.append(el)
        else:
            h_new.append(el[0])
    return h_new


def main(cm_file, perm_file, steps, labels_file, limit_classes=None):
    """Run optimization and generate output."""
    # Load confusion matrix
    with open(cm_file) as f:
        cm = json.load(f)
        cm = np.array(cm)

    n = len(cm)
    make_max = float('inf')
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cm[i][j] = min(cm[i][j], make_max)

    cm_orig = cm.copy()

    # Load permutation
    if os.path.isfile(perm_file):
        with open(perm_file) as data_file:
            perm = json.load(data_file)
    else:
        perm = list(range(len(cm)))

    # Load labels
    if os.path.isfile(labels_file):
        with open(labels_file, "r") as f:
            labels = json.load(f)
    else:
        labels = list(range(len(cm)))

    weights = calculate_weight_matrix(len(cm))
    print("Score: {}".format(calculate_score(cm, weights)))
    result = simulated_annealing(cm, perm,
                                 score=calculate_score,
                                 deterministic=True,
                                 steps=steps)
    print("Score: {}".format(calculate_score(result['cm'], weights)))
    print("Perm: {}".format(list(result['perm'])))
    labels = [labels[i] for i in result['perm']]
    class_indices = list(range(len(labels)))
    class_indices = [class_indices[i] for i in result['perm']]
    print("Symbols: {}".format(labels))
    acc = get_accuracy(cm_orig)
    print("Accuracy: {:0.2f}%".format(acc * 100))
    start = 0
    if limit_classes is None:
        limit_classes = len(cm)
    plot_cm(result['cm'][start:limit_classes, start:limit_classes],
            zero_diagonal=True, labels=labels[start:limit_classes])
    grouping = extract_clusters(result['cm'], labels)
    y_pred = [0]
    cluster_i = 0
    for el in grouping:
        if el == 1:
            cluster_i += 1
        y_pred.append(cluster_i)
    print("silhouette_score={}".format(silhouette_score(cm, y_pred)))
    # Store grouping as hierarchy
    with open('hierarchy.tmp.json', 'w') as outfile:
        hierarchy = apply_grouping(class_indices, grouping)
        hierarchy = _remove_single_element_groups(hierarchy)
        str_ = json.dumps(hierarchy,
                          indent=4, sort_keys=True,
                          separators=(',', ':'), ensure_ascii=False)
        outfile.write(str_)

    # Print nice
    for group in apply_grouping(labels, grouping):
        print("\t{}: {}".format(len(group), [str(el) for el in group]))


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--cm",
                        dest="cm_file",
                        help=("path of a json file with a confusion matrix"),
                        metavar="cm.json",
                        required=True)
    parser.add_argument("--perm",
                        dest="perm_file",
                        help=("path of a json file with a permutation to "
                              "start with"),
                        metavar="perm.json",
                        default="")
    parser.add_argument("--labels",
                        dest="labels_file",
                        help=("path of a json file with a list of label "
                              "names"),
                        metavar="labels.json",
                        default="")
    parser.add_argument("-n",
                        dest="n",
                        default=4 * 10**5,
                        type=int,
                        help="number of steps to iterate")
    parser.add_argument("--limit",
                        dest="limit_classes",
                        type=int,
                        help="Limit the number of classes in the output")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    import doctest
    doctest.testmod()
    main(args.cm_file, args.perm_file, args.n, args.labels_file,
         args.limit_classes)
