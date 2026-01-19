#include "parser.h"
#include "support.h"
#include <opencv2/core.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

void assignArgument(std::string argName, std::string argVal, Options *opts) {
	int parsedVal = parseInt(argVal);

	if (parsedVal < 0) {
		std::cerr << "Error: " << "invalid argument passed for " << argName << '\n';
	} 
	if (argName == "depth") {
		if (parsedVal >= DEPTH_MIN && parsedVal <= DEPTH_MAX) {
			opts->depth = parsedVal;
		} else {
			std::cerr << "Error: " << argName << " is out of range(" << DEPTH_MIN << "-" << DEPTH_MAX << ")\n";
		}
	}

	if (argName == "angle") {
		if (parsedVal >= ANGLE_MIN && parsedVal <= ANGLE_MAX) {
			opts->depth = parsedVal;
		} else {
			std::cerr << "Error: " << argName << " is out of range(" << ANGLE_MIN << "-" << ANGLE_MAX << ")\n";
		}
	}

	if (argName == "filtersize") {
		if (parsedVal >= FILTERSIZE_MIN && parsedVal <= FILTERSIZE_MAX) {
			opts->depth = parsedVal;
		} else {
			std::cerr << "Error: " << argName << " is out of range(" << FILTERSIZE_MIN << "-" << FILTERSIZE_MAX << ")\n";
		}
	}
}

Options parser(int argc, char const *argv[]) {
	Options opts;

	for (int i = 0; i < argc; i++) {
		if (i == 1) {
			try {
				opts.imgPath = cv::samples::findFile(argv[1], true);
				continue;
			} catch (cv::Exception exception) {
				std::cerr << "Error: valid image path is required\n";
			}
		}

		std::string argClean = substr(argv[i], 2, len(argv[i]));

		if (argClean == "depth" || argClean == "angle" || argClean == "filtersize") {
			if (i + 1 > argc) {
				std::cerr << argClean << " requires an argument of type int\n";
				exit(1);
			}
			i++;
			assignArgument(argClean, argv[i], &opts);
		} else if (argClean == "grayscale") {
			opts.grayscale = true;
		}
	} 

	return opts;
}

