from FaceDetection import FaceDetection

# bibliotecas utilizadas pra converter o modelo tensorflow (.h5) pra pytorch (.pt)
import tf2onnx
import onnx
from onnx2pytorch import ConvertModel

from ultralytics import YOLO


class AgeDetection:
    def __init__(self):
        self.tf_model = "models/age_model_acc_0.762.h5"
        self.model_convert()
        self.fd = FaceDetection()

    def model_convert(self):
        # converte o modelo pro formato ONNX
        self.onnx_model, _ = tf2onnx.convert.from_keras(self.tf_model)

        # aqui ele faz onnx_model = onnx.load_model("tf_model.onnx") 
        # mas acho q esse arquivo só é criado se rodar o comando acima o que eu não consegui
        self.onnx_model = onnx.load_model(self.onnx_model)

        # finalmente, converte de onnx pra pytorch
        self.pt_model = ConvertModel(self.onnx_model)

    def detect_by_img(self, img_path):
        # pega a imagem com o rosto
        face = self.fd.img_capture(img_path)

        # aplica a detecção de idade/faixa etária
        age_img = self.pt_model(source = face, verbose = False)

        # plota a imagem
        self.fd.img_plot(age_img)







