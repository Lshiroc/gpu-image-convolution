#ifndef SUPPORT_H
#define SUPPORT_H

#include <cstddef>
#include <string>
#include <sys/time.h>

#define	FILTER_SIZE	3
#define TILE_SIZE	12
#define BLOCK_SIZE	(TILE_SIZE + FILTER_SIZE - 1)
typedef unsigned char uchar;

typedef struct {
	uchar *elements;
	int width;
	int height;
} Image;

typedef struct {
	float *elements;
	int width;
	int height;
} Matrix;

typedef struct {
	struct timeval startTime;
	struct timeval endTime;
} Timer;

Image allocateImageDevice(int width, int height);
Image allocateImage(int width, int height);
Matrix allocateMatrix(int width, int height);
void copyFromHostToDevice(Image dst, const uchar *src);
void copyFromDeviceToHost(uchar *dst, const Image src);
std::string substr(const char *str, size_t start, size_t end);
int parseInt(std::string str);
size_t len(const char *str);
Matrix initializeFilter(int width, int height, float angle);
void startTime(Timer *timer);
void stopTime(Timer *timer);
float elapsedTime(Timer timer);
void help();

#endif

