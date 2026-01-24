#ifndef FILTER_H 
#define FILTER_H 

#include "support.h"
#include "parser.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

void emboss_gpu(cv::Mat *img, Options opts, Timer *timer);
void emboss_cpu(cv::Mat *img, Options opts, Timer *timer);
void emboss_cpu_impl(Image img_in, Image *img_out, Matrix M_h, Options opts, Timer *timer);

#endif

