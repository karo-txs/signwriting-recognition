import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import numpy as np
import mediapipe as mp
import random
import math
import cv2

# Inicializa o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def rotate_to_normal(pose, normal, around):
    z_axis = normal
    y_axis = tf.linalg.cross(tf.constant([1.0, 0.0, 0.0], dtype=tf.float32), z_axis)
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
    p1 = pose[0, :2]
    p2 = pose[9, :2]
    vec = p2 - p1
    angle_rad = tf.atan2(vec[1], vec[0])
    angle_deg = tf.cast(angle_rad * (180.0 / tf.constant(math.pi)), tf.float32)
    return 90.0 + angle_deg

def rotate_hand(pose, angle):
    radians = angle * math.pi / 180.0
    cos_val = tf.cos(radians)
    sin_val = tf.sin(radians)
    rotation_matrix = tf.stack([
        [cos_val, -sin_val, 0.0],
        [sin_val, cos_val, 0.0],
        [0.0, 0.0, 1.0]
    ])
    return tf.matmul(pose, rotation_matrix)

def scale_hand(pose, size=1.0):
    # Calcula o centroide da mão (média de todos os pontos)
    centroid = tf.reduce_mean(pose, axis=0)
    
    # Calcula o fator de escala com base no tamanho desejado
    p1 = pose[0]
    p2 = pose[9]
    current_size = tf.norm(p2 - p1)
    scale_factor = size / current_size
    
    # Aplica o fator de escala e centraliza em relação ao centroide
    pose = (pose - centroid) * scale_factor + centroid
    return pose

def norm(pose):
    min_vals = tf.reduce_min(pose, axis=0)
    max_vals = tf.reduce_max(pose, axis=0)
    pose = (pose - min_vals) / (max_vals - min_vals)
    return pose

def normalize_hand_data(pose):
    # Passo 1: Alinhar a mão com o vetor normal
    normal, base = get_hand_normal(pose)
    pose = rotate_to_normal(pose, normal, base)
    
    # Passo 2: Ajustar a orientação principal para 90 graus
    angle = get_hand_rotation(pose)
    pose = rotate_hand(pose, angle)
    
    # Passo 3: Escalar a mão para um tamanho padrão
    pose = scale_hand(pose, size=1.0)
    
    # Passo 4: Normalizar com Min-Max para o intervalo [0, 1]
    pose = norm(pose)
    
    return pose

def augment_with_rotate_finger(hand_data, max_samples=1):
    max_variations = np.arange(10, 100.0, 0.1)
    random.shuffle(max_variations)
    max_variations = max_variations[:max_samples]

    finger_indices = [[2, 3, 4], [6, 7, 8], [
        10, 11, 12], [14, 15, 16], [18, 19, 20]]

    samples = []
    for max_variation in max_variations:
        for finger in finger_indices:
            new_hand_data = rotate_finger(hand_data, finger, max_variation)
            samples.append(new_hand_data)
    return samples


def augment_with_perturb_points(hand_data, max_samples=1):
    max_variations = np.arange(0.01, 0.05, 0.01)
    random.shuffle(max_variations)
    max_variations = max_variations[:max_samples]

    samples = []
    for max_variation in max_variations:
        new_hand_data = perturb_points(hand_data, max_variation)
        samples.append(new_hand_data)
    return samples


def rotate_finger(hand_landmarks, finger_indices, max_angle_degrees=50):
    base_idx, mid_idx, tip_idx = finger_indices
    base = hand_landmarks[base_idx]
    mid = hand_landmarks[mid_idx]
    tip = hand_landmarks[tip_idx]

    # Calcula o vetor base-tip e base-mid
    base_to_mid = mid - base
    base_to_tip = tip - base

    # Gera um ângulo aleatório dentro do limite especificado
    angle = tf.random.uniform([], -max_angle_degrees,
                              max_angle_degrees) * math.pi / 180

    # Matriz de rotação no plano XY
    cos_val = tf.cos(angle)
    sin_val = tf.sin(angle)
    rotation_matrix = tf.stack([
        [cos_val, -sin_val, 0.0],
        [sin_val, cos_val, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Rotaciona os vetores
    rotated_base_to_mid = tf.tensordot(base_to_mid, rotation_matrix, axes=1)
    rotated_base_to_tip = tf.tensordot(base_to_tip, rotation_matrix, axes=1)

    # Atualiza as posições do mid e tip
    new_mid = base + rotated_base_to_mid
    new_tip = base + rotated_base_to_tip

    # Substitui os landmarks originais pelos rotacionados
    hand_landmarks_updated = tf.tensor_scatter_nd_update(
        hand_landmarks,
        indices=[[mid_idx], [tip_idx]],
        updates=[new_mid, new_tip]
    )

    return hand_landmarks_updated


def perturb_points(hand_data, perturbation_range=1):
    perturbation = tf.random.uniform(
        tf.shape(hand_data), -perturbation_range, perturbation_range)
    perturbed_hand = hand_data + perturbation
    return perturbed_hand

def plot_hand_landmarks(hand_data, title, filename, elev=90, azim=0):
    """
    Plota os landmarks da mão em um gráfico 3D e salva como imagem.
    
    Parâmetros:
    - hand_data: np.array com landmarks 3D da mão.
    - title: título para o gráfico.
    - filename: caminho para salvar a imagem.
    - elev, azim: ângulos para a visualização 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plota os pontos dos landmarks
    ax.scatter(hand_data[:, 0], hand_data[:, 1], hand_data[:, 2], c='blue', s=50)
    
    # Desenha conexões da estrutura da mão
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
        (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
        (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
        (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
        (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
    ]
    for start, end in connections:
        ax.plot([hand_data[start, 0], hand_data[end, 0]], 
                [hand_data[start, 1], hand_data[end, 1]], 
                [hand_data[start, 2], hand_data[end, 2]], 'gray')

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=elev, azim=azim)
    
    # Remove grade e eixos para deixar o gráfico mais limpo
    ax.grid(False)
    ax.set_axis_off()
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Função para gerar e salvar imagens das amostras aumentadas
def generate_augmented_images(hand_data, max_samples=3):
    hand_data = normalize_hand_data(hand_data)
    # Gera amostras aumentadas com rotação de dedo
    rotated_samples = augment_with_rotate_finger(hand_data, max_samples=max_samples)
    for i, sample in enumerate(rotated_samples):
        plot_hand_landmarks(
            sample.numpy(),
            title=f"Augment Rotate Finger {i+1}",
            filename=f"augmented_rotate_finger_{i+1}.png"
        )

    # Gera amostras aumentadas com perturbação de pontos
    perturbed_samples = augment_with_perturb_points(hand_data, max_samples=max_samples)
    for i, sample in enumerate(perturbed_samples):
        plot_hand_landmarks(
            sample.numpy(),
            title=f"Augment Perturb Points {i+1}",
            filename=f"augmented_perturb_points_{i+1}.png"
        )
def process_image(image_path):
    """
    Processa uma imagem e retorna os pontos de referência da mão e a orientação (mão esquerda ou direita).
    
    Parâmetros:
        image_path (str): Caminho da imagem de entrada.
    
    Retorna:
        landmarks (np.array): Array numpy com os 21 pontos de referência em 3D da mão.
        handedness (str): String indicando se é "Left" ou "Right".
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks and results.multi_handedness:
        landmarks = []
        for landmark in results.multi_hand_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
        handedness = results.multi_handedness[0].classification[0].label
        return np.array(landmarks), handedness
    else:
        return None, None

# Exemplo de uso
image_path = "K:/Master/repositories/SignWritingAI/sw-modeling/assets/raw_data/signwriting_org_v2/images/S1a0/right.jpg"  # Substitua pelo caminho da imagem
landmarks, handedness = process_image(image_path)
landmarks = tf.convert_to_tensor(landmarks, dtype=tf.float32)
if landmarks is not None:
    generate_augmented_images(landmarks)
else:
    print("Nenhuma mão detectada na imagem.")