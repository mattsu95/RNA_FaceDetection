import os

import cv2
from ultralytics import YOLO

from AgeDetection import AgeDetection

class FaceDetection:
    def __init__(self, age_model: AgeDetection = None):
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "yolov11n-face.pt")
        self.face_model = YOLO(model_path)
        self.camera = None
        self.ad: AgeDetection = age_model

        self.window_name = "Face Detecion"


    def open_camera(self, camera_index=0):
        self.release_camera()
        self.camera = cv2.VideoCapture(camera_index)
        if not self.camera.isOpened():
            self.release_camera()
            raise RuntimeError("Não foi possível acessar a webcam.")
        return self.camera

    def read_camera_frame(self):
        if self.camera is None:
            raise RuntimeError("A webcam ainda não foi iniciada.")
        success, bgr_frame = self.camera.read()
        if not success:
            return None, 0
        return self.process_frame(bgr_frame)

    def process_frame(self, bgr_frame):
        detection_result = self.face_model(bgr_frame, verbose=False)
        result = detection_result[0]

        result_frame = result.plot()

        if self.ad and len(result.boxes) > 0:
            alt_img, larg_img = bgr_frame.shape[:2]
            margem = 0.30

            for box in result.boxes:
                #pega as coord.
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                #calcula margme
                offset_x = int((x2 - x1) * margem)
                offset_y = int((y2 - y1) * margem)

                #aplica margem
                novo_x1 = max(0, x1 - offset_x)
                novo_y1 = max(0, y1 - offset_y)
                novo_x2 = min(larg_img, x2 + offset_x)
                novo_y2 = min(alt_img, y2 + offset_y)

                #Recorte do rosto
                face_img = bgr_frame[novo_y1:novo_y2, novo_x1:novo_x2]

                #Calcula a idade
                idade_predita = self.ad.detect_by_img(face_img)

                if idade_predita:
                    cv2.putText(result_frame, f"Idade: {idade_predita}", (x1, max(20, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        face_count = len(result.boxes)
        return result_frame, face_count

    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    # detecção de rosto através da webcam
    def webcam_capture(self, camera_index=0):
        camera_object = self.open_camera(camera_index)

        try:
            while True:
                success, bgr_frame = camera_object.read()
                if not success:
                    break

                result_frame, _ = self.process_frame(bgr_frame)

                cv2.imshow(self.window_name, result_frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break

                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.release_camera()
            cv2.destroyAllWindows()


    def img_capture(self, img_path):
        # lê a imagem
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem no caminho: {img_path}")

        # aplica a detecção facial nela
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

        