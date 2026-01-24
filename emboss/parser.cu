#include "parser.h"
#include "support.h"
#include <opencv2/core.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

void assignArgument(std::string argName, std::string argVal, Options *opts) {
	float parsedVal;
	try {
		parsedVal = std::stof(argVal);
	} catch(const std::invalid_argument&) {
		std::cerr << "Error: " << "invalid argument passed for " << argName << '\n';
		exit(1);
	}

	if (argName == "depth") {
		if (parsedVal >= DEPTH_MIN && parsedVal <= DEPTH_MAX) {
			opts->depth = parsedVal;
		} else {
			std::cerr << "Error: " << argName << " is out of range(" << DEPTH_MIN << "-" << DEPTH_MAX << ")\n";
			exit(1);
		}
	}

	if (argName == "angle") {
		if (parsedVal >= ANGLE_MIN && parsedVal <= ANGLE_MAX) {
			opts->angle = parsedVal;
		} else {
			std::cerr << "Error: " << argName << " is out of range(" << ANGLE_MIN << "-" << ANGLE_MAX << ")\n";
			exit(1);
		}
	}
}

Options parser(int argc, char const *argv[]) {
	Options opts;

	for (int i = 0; i < argc; i++) {
		std::string argClean = substr(argv[i], 2, len(argv[i]));
		if (i == 1 && argClean == "help") {
			help();
			exit(1);
		} else if (i == 1) {
			try {
				opts.imgPath = cv::samples::findFile(argv[1], true);
				std::cout << "adsdad: " << opts.imgPath << "\n";
				continue;
			} catch (cv::Exception exception) {
				std::cerr << "Error: valid image path is required\n";
				exit(1);
			}
		}


		if (argClean == "depth" || argClean == "angle") {
			if (i + 1 > argc) {
				std::cerr << argClean << " requires an argument of number\n";
				exit(1);
			}
			i++;
			assignArgument(argClean, argv[i], &opts);
		} else if (argClean == "grayscale") {
			opts.grayscale = true;
		} else if (argClean == "cpu") {
			opts.isCPUEnabled = true;
		}
	} 

	return opts;
}

