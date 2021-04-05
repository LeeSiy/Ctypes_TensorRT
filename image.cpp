#include <map>
#include <opencv2/opencv.hpp>
#include<iostream>
#include<vector>
#include "image.h"

cv::Mat readfrombuffer(uchar* frame_data,int height, int width,int channels){
    if(channels == 3){
        cv::Mat img(height, width, CV_8UC3);
        uchar* ptr =img.ptr<uchar>(0);
        int count = 0;
        for (int row = 0; row < height; row++){
             ptr = img.ptr<uchar>(row);
             for(int col = 0; col < width; col++){
	               for(int c = 0; c < channels; c++){
	                  ptr[col*channels+c] = frame_data[count];
	                  count++;
	                }
	         }
        }
	//delete frame_data;
        return img;
    }
}
char* mattostring(uchar* frame_data, int rows, int cols, int channels){
    cv::Mat mat = readfrombuffer(frame_data,rows,cols,channels);
    if (!mat.empty()) {

	std::vector<uchar> data_encode;
	std::vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);  // png select jpeg CV_IMWRITE_JPEG_QUALITY
        compression_params.push_back(1); // Fill in the picture quality you want in this 0-9

        imencode(".png", mat, data_encode);

        std::string str_encode(data_encode.begin(), data_encode.end());
        char* char_r = new char[str_encode.size() + 10];     
        memcpy(char_r, str_encode.data(), sizeof(char) * (str_encode.size()));
        return char_r;
    }
}
void save_data(uchar* frame_data, int rows, int cols, int channels){
        frame_data = frame_data;
        rows = rows;
        cols = cols;
        channels = channels;
        cv::Mat cv_ex = readfrombuffer(frame_data,rows,cols,channels);
        cv::imwrite("result.png", cv_ex);
};
extern "C"{
    char* main_mattostring(uchar* matrix, int rows, int cols, int channels)
    {
        return mattostring(matrix, rows, cols,  channels);
    }
    void main_save_data(uchar* frame_data, int rows, int cols, int channels)
    {
        save_data(frame_data, rows, cols, channels);
    }
}
