import requests
import os
import hashlib

__all__ = ["download"]

def download(path, url):
    print("donwload url=",url)
    r=requests.get(url)
    r.raise_for_status()
    md5=hashlib.md5(url.encode('utf-8')).hexdigest()
    filename=path+"/"+md5+'.jpg';
    with open(filename,"wb") as f:
        f.write(r.content)
    print("done")
    return filename;

if __name__ == "__main__":    
    print(download("temp_image","http://img.mxtrip.cn/fadd1b80f8f62eb335cca0a1ffb777f1.jpeg"))
