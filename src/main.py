from FaceDetection import FaceDetection

if __name__ == "__main__":
    fd = FaceDetection()
    fd.webcam_capture()
    #img = fd.img_capture("teste.png")
    #fd.img_plot(img)