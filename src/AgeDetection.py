from FaceDetection import FaceDetection

from ultralytics import YOLO

import torch


class AgeDetection:

    # Gemini sugeriu um bgl assim
    """ def __init__(self):
        # 1. Instancia a estrutura do modelo (substitua pelo nome real da sua classe)
        self.age_model = MinhaRedeCnn() 
        
        # 2. Carrega o dicionário de pesos (OrderedDict)
        pesos = torch.load("models/age_model.pt", map_location=torch.device('cpu'))
        
        # 3. Injeta os pesos na estrutura do modelo
        self.age_model.load_state_dict(pesos)
        
        # 4. Agora sim, coloca em modo de inferência
        self.age_model.eval() """

    def __init__(self):
        self.age_model = torch.load("models/age_model.pt")
        self.age_model.eval()  # Coloca o modelo em modo de inferência
        self.fd = FaceDetection()


    def detect_by_img(self, img_path):
        # pega a imagem com o rosto
        face = self.fd.img_capture(img_path)

        # aplica a detecção de idade/faixa etária
        with torch.no_grad():
            age_img = self.age_model(face)

        # plota a imagem
        self.fd.img_plot(age_img)







