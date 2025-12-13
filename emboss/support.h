#ifndef SUPPORT_H
#define SUPPORT_H

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
	int *elements;
	int width;
	int height;
} Matrix;

Image allocateImageDevice(int width, int height);
Matrix allocateMatrix(int width, int height);
void copyFromHostToDevice(Image dst, const uchar *src);
void copyFromDeviceToHost(uchar *dst, const Image src);

#endif

