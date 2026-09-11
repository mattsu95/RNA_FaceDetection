import cv2
from ultralytics import YOLO

class FaceDetection:
    def __init__(self):
        # usa um modelo pré-treinado de reconhecimento facial
        self.face_model = YOLO("models/yolov11n-face.pt")

        self.window_name = "Face Detecion"

    # detecção de rosto através da webcam
    def webcam_capture(self):
        camera_object = cv2.VideoCapture(0)

        while True:
            success, bgr_frame = camera_object.read()
            if not success:
                break

            # diminui o tamanho do frame
            # bgr_frame = self.escalonar_frame(bgr_frame, 25)

            # aplica a detecção de rostos no frame 
            detection_result = self.face_model(bgr_frame, verbose = False)

            # "recria" a imagem do frame unida com a detecção de rostos
            result_frame = detection_result[0].plot()


            # mostra a imagem da câmera com a detecção de rostos aplicada
            cv2.imshow(self.window_name, result_frame)

            # fecha com esc
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # para a execução caso feche a janela
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break


        # termina a conexão com a webcam e destroi a janela
        camera_object.release()
        cv2.destroyAllWindows()


    def img_capture(self, img_path):
        # lê a imagem
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem no caminho: {img_path}")

        # aplica o reconhecimento facial nela
        image_result = self.face_model(source = image, verbose = False)

        return image_result

    def img_plot(self, img):
        while True:
            cv2.imshow(self.window_name, img[0].plot())

            # fecha com esc
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # para a execução caso feche a janela
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

        