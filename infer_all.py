from UNet.inference import infer
from yolov5.detect import infer_detect

def infer_all(path):

    seg_img = infer(path)
    detect_img,class_count = infer_detect(seg_img)
    return detect_img,class_count
if __name__=="__main__":
    img = infer_all('图片测试/0002.jpg')