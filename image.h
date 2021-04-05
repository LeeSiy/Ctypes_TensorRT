#ifndef __OPERATION_H
#define __OPERATION_H

#include <map>
#include <opencv2/opencv.hpp>
#include<iostream>
#include<vector>

cv::Mat readfrombuffer(uchar* ,int , int ,int );
char* mattostring(uchar* , int , int , int );
void save_data(uchar* , int , int , int );
#endif
