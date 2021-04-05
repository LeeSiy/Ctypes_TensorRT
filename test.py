#from crnn import TensorRT, test_rgb_to_gray
import cv2
import matplotlib.pyplot as plt
import ctypes
import numpy as np
import os

#importing library----------------------------------------------
lib = ctypes.cdll.LoadLibrary('libcrnn_copy.so')

#setting for python object--------------------------------------
class TensorRT(object):
    def __init__(self):
        lib.newTensorRT.argtypes=[ctypes.c_void_p]
        lib.newTensorRT.restype=ctypes.c_void_p
              
        lib.TensorRT_TensorRT.argtypes=[ctypes.c_void_p]
        lib.TensorRT_TensorRT.restype=ctypes.c_void_p
    
        lib.TensorRT_run.argtypes=[ctypes.c_void_p]
        lib.TensorRT_run.restype=ctypes.c_int

        lib.TensorRT_memory_free=[ctypes.c_void_p]
        lib.TensorRT_memory_free=ctypes.c_void_p
        
        self.obj=lib.newTensorRT(None,0,None,0,0)
        
    def TensorRT(self):
        lib.TensorRT_TensorRT(self.obj)
    def run(self):
        lib.TensorRT_run(self.obj)
    def memory_free(self):
        lib.TensorRT_memory_free(self.obj)

#image dealing----------------------------------------------------
path_dir='/home/user/delete/tensorrtx/demo.png'
src = cv2.imread(path_dir)
cols = src.shape[1]
rows = src.shape[0]
channels = src.shape[2]

src = np.asarray(src, dtype=np.uint8)
src1 = src.ctypes.data_as(ctypes.c_char_p)
lib.main_mattostring.restype = ctypes.c_void_p
a = lib.main_mattostring(src1,rows,cols,channels)
b = ctypes.string_at(a,cols*rows*channels) #string_at(c_str_p) # Get content
nparr = np.frombuffer(b, np.uint8)
img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

lib.main_save_data.restype = ctypes.c_void_p
lib.main_save_data(src1,rows,cols,channels);

#running code with c++ function------------------------------------
rt = TensorRT()
for i in range(50):
    print("i=",i)
    lib.main_save_data(src1,rows,cols,channels);
    print(rt.run())
    os.remove('./result.png')
rt.memory_free

