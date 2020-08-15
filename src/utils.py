import numpy as np
import pandas as pd

def visit_sparse_segmentation_to_df(labels, min_length):
    segmentation = np.where(np.diff(labels) != 0)[0]+1
    segmentation = np.insert(segmentation, 0, 0)
    segmentation = np.insert(segmentation, len(segmentation), len(labels))
    while np.any(np.diff(segmentation) < min_length):
        to_delete = np.where(np.diff(segmentation) < min_length)[0][0]
        segmentation = np.delete(segmentation, to_delete)
        if to_delete != len(segmentation)-1:
            segmentation = np.delete(segmentation, to_delete)
    segmentation = segmentation[1:-1]
    segmentation = np.repeat(segmentation, 2)
    segmentation = np.insert(segmentation, 0, 0)
    segmentation = np.insert(segmentation, len(segmentation), len(labels))
    segmentation = np.reshape(segmentation, (-1, 2))

    new_labels = []
    for s in segmentation:
        new_labels.append("visit" if labels[s[0]] else "transition")

    d = {'Start frame': segmentation[:, 0], 'End frame': segmentation[:, 1], "Type": new_labels}
    df = pd.DataFrame(data=d)
    return df


def view_sparse_segmentation_to_df(labels, min_length):
    segmentation = np.where(np.diff(labels) != 0)[0]+1
    segmentation = np.repeat(segmentation, 2)
    segmentation = np.insert(segmentation, 0, 0)
    segmentation = np.insert(segmentation, len(segmentation), len(labels))
    segmentation = np.reshape(segmentation, (-1, 2))
    to_delete = []
    for i, s in enumerate(segmentation):
        if s[1]-s[0] < min_length:
            to_delete.append(i)
    segmentation = np.delete(segmentation, to_delete,axis=0)
    segmentation = np.reshape(segmentation, (-1, 2))

    d = {'Start frame': segmentation[:, 0], 'End frame': segmentation[:, 1]}
    df = pd.DataFrame(data=d)
    return df


