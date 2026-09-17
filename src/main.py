#from FaceDetection import FaceDetection
from AgeDetection import AgeDetection
from FaceDetection import FaceDetection

if __name__ == "__main__":
    #fd = FaceDetection()
    #fd.webcam_capture()
    #img = fd.img_capture("teste.png")
    #fd.img_plot(img)

    ad = AgeDetection()

    fd = FaceDetection(age_model=ad)
    fd.webcam_capture(camera_index=0)

    #ad.detect_by_img("teste.png")