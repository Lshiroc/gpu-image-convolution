#ifndef FILTERS_H 
#define FILTERS_H 

#include "support.h"
#include "parser.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

void emboss_gpu(cv::Mat *img, Options opts);
void emboss_cpu(cv::Mat *img, Options opts);
void emboss_cpu_impl(Image img_in, Image *img_out, Matrix M_h, Options opts);

#endif

