import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Inicializa o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Conexões entre pontos para a estrutura da mão
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Polegar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Indicador
    (0, 9), (9, 10), (10, 11), (11, 12),   # Médio
    (0, 13), (13, 14), (14, 15), (15, 16), # Anelar
    (0, 17), (17, 18), (18, 19), (19, 20)  # Mínimo
]
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

def plot_hand_landmarks(landmarks, handedness, save_paths=["hand_plot1.png", "hand_plot2.png"]):
    """
    Gera dois gráficos 3D dos pontos da mão e salva como imagens.
    
    Parâmetros:
        landmarks (np.array): Array numpy com shape (21, 3) representando os 21 pontos da mão.
        handedness (str): String indicando se é "Left" ou "Right".
        save_paths (list): Lista com os caminhos para salvar as imagens dos gráficos.
    """
    fig = plt.figure(figsize=(10, 5))

    # Primeiro ângulo de visão
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.linspace(0, 1, 21), cmap='viridis', s=100)
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([landmarks[start, 0], landmarks[end, 0]],
                [landmarks[start, 1], landmarks[end, 1]],
                [landmarks[start, 2], landmarks[end, 2]], 'gray')
    ax.set_title(f'Hand: {handedness}')
    # ax.view_init(elev=90, azim=-90)
    ax.set_xticklabels([])  # Remove números do eixo X
    ax.set_yticklabels([])  # Remove números do eixo Y
    ax.set_zticklabels([])  # Remove números do eixo Z

    # Segundo ângulo de visão
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c=np.linspace(0, 1, 21), cmap='viridis', s=100)
    for connection in HAND_CONNECTIONS:
        start, end = connection
        ax.plot([landmarks[start, 0], landmarks[end, 0]],
                [landmarks[start, 1], landmarks[end, 1]],
                [landmarks[start, 2], landmarks[end, 2]], 'gray')
    ax.set_title(f'Hand: {handedness}')
    # ax.view_init(elev=90, azim=0)
    ax.set_xticklabels([])  # Remove números do eixo X
    ax.set_yticklabels([])  # Remove números do eixo Y
    ax.set_zticklabels([])  # Remove números do eixo Z

    plt.tight_layout()
    plt.savefig(save_paths[0])
    plt.savefig(save_paths[1])
    plt.show()



# Exemplo de uso
image_path = "K:/Master/repositories/SignWritingAI/sw-modeling/assets/raw_data/signwriting_org_v2/images/S1a0/left.jpg"  # Substitua pelo caminho da imagem
landmarks, handedness = process_image(image_path)

if landmarks is not None:
    plot_hand_landmarks(landmarks, handedness)
else:
    print("Nenhuma mão detectada na imagem.")
