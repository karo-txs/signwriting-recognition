import tensorflow as tf
import math


def rotate_to_normal(pose, normal, around):
    z_axis = normal
    y_axis = tf.linalg.cross(tf.constant(
        [1.0, 0.0, 0.0], dtype=tf.float32), z_axis)
    x_axis = tf.linalg.cross(z_axis, y_axis)
    axis = tf.stack([x_axis, y_axis, z_axis])
    return tf.tensordot(pose - around, axis, axes=[[1], [1]])


def get_hand_normal(pose):
    plane_points = [0, 17, 5]
    triangle = tf.gather(pose, plane_points, axis=0)
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = tf.linalg.cross(v1, v2)
    normal /= tf.norm(normal)
    return normal, triangle[0]


def get_hand_rotation(pose):
    p1 = pose[0, :2]  # Extract x, y coordinates
    p2 = pose[9, :2]
    vec = p2 - p1
    angle_rad = tf.atan2(vec[1], vec[0])  # atan2 returns angle in radians
    # Convert radians to degrees
    angle_deg = tf.cast(angle_rad * (180.0 / tf.constant(math.pi)), tf.float32)
    return 90.0 + angle_deg


def rotate_hand(pose, angle):
    radians = angle * math.pi / 180.0  # Convert degrees to radians
    cos_val = tf.cos(radians)
    sin_val = tf.sin(radians)
    rotation_matrix = tf.stack([
        [cos_val, -sin_val, 0.0],
        [sin_val, cos_val, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return tf.matmul(pose, rotation_matrix)


def scale_hand(pose, size=200.0):
    p1 = pose[0]
    p2 = pose[9]
    current_size = tf.norm(p2 - p1)
    scale_factor = size / current_size
    return pose * scale_factor - pose[0]


def apply_all_normalizations(pose):
    normal, base = get_hand_normal(pose)
    pose = rotate_to_normal(pose, normal, base)
    angle = get_hand_rotation(pose)
    pose = rotate_hand(pose, angle)
    pose = scale_hand(pose, 200)

    min_vals = tf.reduce_min(pose, axis=0)
    max_vals = tf.reduce_max(pose, axis=0)
    pose = (pose - min_vals) / (max_vals - min_vals)

    return pose
