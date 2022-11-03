![Real-Time_Hand_Sign_Recognition](https://user-images.githubusercontent.com/105673165/199635985-2081100e-75da-4f78-8325-e8dbe3c83d8c.png)

## Desafio 2 Hand Talk - Reconhecimento de Ações em Vídeo + Desafio Bônus (Real Time Classification).

### Objetivo
Você precisará criar um sistema que reconhece determinadas ações em um vídeo, escolha pelo menos 20 ações diferentes (quantidade de classes). Utilize qualquer base de dados disponível na web, mas o seu sistema terá que ser validado, então certifique-se de que ele seja capaz de reconhecer a ação de qualquer entrada de vídeo nova, inclusive de uma webcam.

### Desafio bônus
Uma vez cumprido o desafio, nós trazemos para você uma desafio bônus, aquele que não é obrigatório fazer, maaaaas irá encher os olhos do avalador com uma ⭐. Então bora lá…. o desafio bônus é: que o seu sistema seja capaz de reconhecer ações em tempo real.

### Requisitos
Python 3+
Tensorflow 2.x
Necessário processar o frame com Mediapipe Holistic antes de enviar para o modelo.

*---------------------------------------------------------------------------------------------------------------------------------*

Bom, primeiramente tentei implementar uma CNN simples, utilizando um dataset de Hand Signs estrangeiro, porém acabei treinando o modelo com dados contendo informação dos pixels das imagens do dataset, e portanto, para minha ideia inicial de utilizar o MediaPipe para captar "Hand Landmarks" a partir da minha WebCam, este modelo não funcionou bem.

Então precisei construir outro modelo trabalhasse com previsões a partir de dados no formato que o MediaPipe obtinha de minha WebCam.

Minha referência inicial foi um projeto (Referência 1) onde o desenvolvedor utilizou o MediaPipe Hands para extrair informações e classificar os sinais em tempo real, porém utilizando um modelo pré-treinado (também da MediaPipe).
* O MediaPipe Hands estava extraindo 3 coordenadas (x,y,z) de 21 pontos (portando uma array 21,3) da "Hand Landmark", então eu precisava treinar meu modelo para realizar previsões com dados similarmente formatados.

Então eu construí um modelo baseado em CNN, relativamente simples, alimentado com features (extraídas com MediaPipe) resultantes de imagens de um dataset disponível no Kaggle:
 https://www.kaggle.com/datasets/mlanangafkaar/datasets-lemlitbang-sibi-alphabets
 
Este dataset possui imagens de sinais de mão (padrão da indonésia) representando as letras do alfabeto. 
Ex:
### A
![image](https://user-images.githubusercontent.com/105673165/194180319-37a3c9ab-cdc8-4e9f-9159-3fb342eb38c1.png)

### L
![image](https://user-images.githubusercontent.com/105673165/194180391-f2aa671b-9fc8-4afa-9e64-ab8d76cfefca.png)

O modelo construído performou super bem após 50 epochs de treinamento (arquivo "handrecognitionSIBI.ipynb" do repositório) - aproximadamente 90% de acurácia.
![image](https://user-images.githubusercontent.com/105673165/194180533-7285afb6-3132-46ad-a614-688388414df2.png)

Então partí para a implementação do modelo, no script que construí para realizar a extração de informações da WebCam utilizando o MediaPipe (arquivo "real time hand recognition.ipynb" do repositório), para realizar a classificação de sinais em tempo real.

Postei um vídeo no YouTube com o teste que realizei executando o script (arquivo "realtimehandrecognition.py") diretamente do prompt de comando. O teste correu muito bem e o modelo funcionou na classificação dos frames em tempo real!

#### LINK : https://youtu.be/74Z6kt5ojNw

Ex:

![image](https://user-images.githubusercontent.com/105673165/194181503-1d806074-c475-4cb1-9b5a-74e5ce40f623.png)



*---------------------------------------------------------------------------------------------------------------------------------*

### Referências:
#### 1 - Real-time Hand Gesture Recognition using TensorFlow & OpenCV - TechVidvan:
https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

#### 2 - CNN with Mediapipe for Sign Language Recognition - M. Lanang Afkaar:
https://www.kaggle.com/code/mlanangafkaar/cnn-with-mediapipe-for-sign-language-recognition
