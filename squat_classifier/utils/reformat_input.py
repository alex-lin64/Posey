from bodyparts import Bodyparts
import tensorflow as tf


def get_center_point(keypoints, left_bp, right_bp):
    """
    Calculates center point of two opposing body parts

    :params:

    :returns:
    
    """
    l = tf.gather(keypoints, left_bp, axis=1)
    r = tf.gather(keypoints, right_bp, axis=1)
    center = 0.5 * (l + r)

    return center