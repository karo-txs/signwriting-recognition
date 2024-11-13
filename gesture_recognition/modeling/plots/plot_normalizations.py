import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inicializa o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funções de normalização fornecidas
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

def save_landmark_image(pose, title, filename, elev=90, azim=0, figsize=(15, 15), lim=1):
    """
    Salva uma imagem dos landmarks da mão em um gráfico 3D.
    
    Parâmetros:
    - pose: pontos 3D da mão.
    - title: título da imagem.
    - filename: caminho para salvar a imagem.
    - elev, azim: ângulos de visualização.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Desenha landmarks e conexões
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=np.linspace(0, 1, 21), cmap='viridis', s=100)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Polegar
        (0, 5), (5, 6), (6, 7), (7, 8),  # Indicador
        (0, 9), (9, 10), (10, 11), (11, 12),  # Médio
        (0, 13), (13, 14), (14, 15), (15, 16),  # Anelar
        (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
    ]
    for start, end in connections:
        ax.plot([pose[start, 0], pose[end, 0]], [pose[start, 1], pose[end, 1]], [pose[start, 2], pose[end, 2]], 'gray')
    
    ax.set_title(title)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.view_init(elev=elev, azim=azim)
    
    # Remove plano de fundo, grade e eixos
    ax.grid(False)
    ax.set_axis_off()
    
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

# Pipeline para aplicar cada transformação e salvar imagens por etapa
def apply_all_normalizations_and_save_images(pose):
    # Converte pose para tensores do TensorFlow
    pose = tf.convert_to_tensor(pose, dtype=tf.float32)
    
    # Inverte o eixo Y para corrigir a orientação
    pose = tf.concat([pose[:, 0:1], 1 - pose[:, 1:2], pose[:, 2:]], axis=1)
    
    # Estado inicial
    save_landmark_image(pose.numpy(), "Posição Inicial", "step_1_initial.png")

    # Etapa 1: Calcular o vetor normal da mão e centralizar
    normal, base = get_hand_normal(pose)
    pose = rotate_to_normal(pose, normal, base)
    save_landmark_image(pose.numpy(), "Rotação para Alinhar com Normal", "step_2_rotate_to_normal.png")

    # Etapa 2: Ajustar a orientação principal (90 graus)
    angle = get_hand_rotation(pose)
    pose = rotate_hand(pose, angle)
    save_landmark_image(pose.numpy(), "Rotação Principal (90 graus)", "step_3_rotate_hand.png")

    # Etapa 3: Escalar a mão para tamanho padrão
    pose = scale_hand(pose, 0.5)
    save_landmark_image(pose.numpy(), "Escala para Tamanho Padrão", "step_4_scale_hand.png", lim=2)

    # Etapa 4: Normalização Min-Max para o intervalo [0, 1]
    pose = norm(pose)
    save_landmark_image(pose.numpy(), "Normalização Min-Max", "step_5_min_max_normalization.png")

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

if landmarks is not None:
    apply_all_normalizations_and_save_images(landmarks)
else:
    print("Nenhuma mão detectada na imagem.")
