#include "filter.h"
#include <iostream>

int main(int argc, char const *argv[]) {
	Timer timer;
	cv::Mat img;
	Options opts;


	opts = parser(argc, argv);
	if (opts.imgPath == "") {
		std::cerr << "Error: valid image path is required\n";
		exit(1);
	}
	img = cv::imread(opts.imgPath, cv::IMREAD_COLOR);
	if (!img.isContinuous()) {
		img = img.clone();
	}

	if (opts.isCPUEnabled) {
		emboss_cpu(&img, opts, &timer);
	} else {
		emboss_gpu(&img, opts, &timer);
	}

	// Displaying and saving result
	cv::imshow("Emboss filtered image", img);
	cv::waitKey(0);
	cv::imwrite("embossed.jpg", img);

	return 0;
}

