#ifndef PARSER_H
#define PARSER_H

#include <string>

#define DEPTH_MIN	0
#define DEPTH_MAX	100
#define ANGLE_MIN	0
#define ANGLE_MAX	180
#define FILTERSIZE_MIN	3
#define FILTERSIZE_MAX	10

typedef struct {
	std::string imgPath;
	int depth{};
	int angle{ 90 };
	int filtersize{ 3 };
	bool grayscale{};
} Options;
Options parser(int argc, char const *argv[]);

#endif

