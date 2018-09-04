#coding:utf-8
import sys
import traceback
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
import requests
from download import download
import time
import json
import hashlib

#0: test 1: online
run_mode = 0
test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
img_server_url="http://192.168.0.1:10001"
path="temp_image"
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []

def call_4_urls(num=10):
    r = requests.get(imgurl+"?type=1&num=10")
    content = r.content
    arr = []
    if content !='':
        jsonObj = json.loads(content)
        data = jsonObject['data']
        urls = json.loads(data)
        for url in urls:
            arr.append(url)
    #TODO parse content
    return arr
def dectect_callback(url, status):
    for i in range(3):
        r = requests.get(imgurl+"?type=2&imgurl="+url+"&status="+str(status))
        if r.code == 200:
            break
def faked_call_4_urls(num=10):
    #arr = ["http://img.mxtrip.cn/fadd1b80f8f62eb335cca0a1ffb777f1.jpeg"]
    urls=[]
    f = open("urls.txt")               # 返回一个文件对象 
    line = f.readline()               # 调用文件的 readline()方法 
    while line:
        urls.append(line.replace('\n',''))
        if len(urls) == 10:
            print(line)
            yield urls
            urls=[]
        line = f.readline() 
    f.close()  

    yield urls
def faked_callback(url, status):
    print("call back set url=",url," status=",status)
    file = "detect_result.txt"
    with open(file, 'a+') as f:
        f.write(url+"\t"+str(status)+"\n")
    f.close()

def run_file():
    urls_gen = faked_call_4_urls()
    for urls in urls_gen:
        gt_imdb = []
        for url in urls:
            md5=hashlib.md5(url.encode('utf-8')).hexdigest()
            name = path+"/"+md5+'.jpg';
            if os.path.exists(name) == False :
                print("begin to download imgurl=",url)
                download("temp_image",url)
            else:
                print("exist file name="+name)
            gt_imdb.append(name)
        if len(gt_imdb)==0:
            return
        test_data = TestLoader(gt_imdb)
        start = int(round(time.time()*1000))
        all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
        end = int(round(time.time()*1000))
        print("cost time=",(end-start))
        count = 0
        for bboxes in all_boxes:
            if len(bboxes) ==0:
                faked_callback(urls[count],0)
            else:
                faked_callback(urls[count],1)
            count = count + 1
def run_http():
    urls = call_4_urls()
    gt_imdb = []
    for imgurl in urls:
        print("begin to download imgurl=",imgurl)
        name = download(imgurl)
        gt_imdb.append(name)
    test_data = TestLoader(gt_imdb)
    start = int(round(time.time()*1000))
    all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
    end = int(round(time.time()*1000))
    print("cost time=",(end-start))
    count = 0
    for bboxes in all_boxes:
        if len(bboxes) ==0:
            dectect_callback(urls[count],0)
        else:
            dectect_callback(urls[count],1)

if __name__ == "__main__":
    try:
        if run_mode == 0:
            run_file()
        else:
            while(1):
                run_http()
    except Exception as e:
        traceback.print_exc()



