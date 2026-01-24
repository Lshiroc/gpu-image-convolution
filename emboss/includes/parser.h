#ifndef PARSER_H
#define PARSER_H

#include <string>

#define DEPTH_MIN	0
#define DEPTH_MAX	1
#define ANGLE_MIN	0
#define ANGLE_MAX	360

typedef struct {
	std::string imgPath;
	float depth{ 1.0f };
	float angle{ 45.0f };
	bool grayscale{};
	bool isCPUEnabled{};
} Options;
Options parser(int argc, char const *argv[]);

#endif

