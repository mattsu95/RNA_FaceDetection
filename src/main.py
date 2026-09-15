#from FaceDetection import FaceDetection
from AgeDetection import AgeDetection
if __name__ == "__main__":
    #fd = FaceDetection()
    #fd.webcam_capture()
    #img = fd.img_capture("teste.png")
    #fd.img_plot(img)

    ad = AgeDetection()
    ad.detect_by_img("teste.png")