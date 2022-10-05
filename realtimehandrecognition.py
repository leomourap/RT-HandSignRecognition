#!/usr/bin/env python
# coding: utf-8

# Desafio Bônus - Reconhecimento de ações em tempo real!

# Importando os pacotes necessários para reconhecimento dos sinais utilizando Python OpenCV
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# iniciando o MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Carregando o modelo de classificação dos sinais desenvolvido anteriormente
model = load_model('cnn_sibi')
# Carregando as classes - criei este arquivo manualmente!
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Iniciando a WebCam para reconhecer os sinais
cap = cv2.VideoCapture(0)
while True:
    # Ler cada frame da WebCam
    _, frame = cap.read()
    x , y, c = frame.shape
    # Girar o frame verticalmente
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Obter a classificação do frame a partir do "esqueleto" da mão - os Hands Landmarks.
    result = hands.process(framergb)
    
    className = ''
    
    # pós processamento dos resultados
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                lmz = int(lm.z)
                landmarks.append(lmx)
                landmarks.append(lmy)
                landmarks.append(lmz)
            # Desenhando os Landmarks nos frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            # Classificando o sinal
            prediction = model.predict([landmarks])
            predictions = np.array(prediction)
            print(predictions)
            classID = np.argmax(predictions)
            print(classID)
            className = classNames[classID].capitalize()
            # Exibindo a classificação no frame
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    # Exibindo o output final
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Apertar a tecla "Q" para finalizar a janela da WebCam.
cap.release()
cv2.destroyAllWindows()