#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <dirent.h>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>  

#include "/usr/local/include/opencv/cv.h"
#include "/usr/local/include/opencv/cvaux.h"
#include "/usr/local/include/opencv/highgui.h"
#include "/usr/local/include/opencv2/video/video.hpp"

#ifndef __VIDEOS_H
#define __VIDEOS_H

#define _In_
#define _InOut_
#define _Out_

#define NUM_COEFFICIENTS 11
#define MAX_CONTOUR_NAME_LENGTH 100

#define INPUT_FILE_OPEN_ERROR -1
#define INCORRECT_INPUT_FILE_FORMAT_ERROR -2
#define INCORRECT_NUM_CAMERAS_ERROR -3
#define OUT_OF_MEMORY_ERROR -4
#define OUTPUT_FILE_OPEN_ERROR -5
#define NOT_ENOUGH_POINTS_ERROR -6
#define IMAGE_INFO_PROPERTIES_ERROR -7
#define IMAGE_INFO_DATA_ERROR -8
#define IMAGE_INFO_DIMENSIONS_ERROR -9
#define IMAGE_INFO_PADDED_WIDTH_ERROR -10
#define IMAGE_DEPTH_ERROR -11
#define INVALID_VIDEO_FILE_EXTENSION_ERROR -12
#define MODEL_NOT_FOUND_ERROR -13
#define INVALID_NUM_CONTOURS_ERROR -14
#define INVALID_FEATURE_DETECTOR_ERROR -15

#define GFTT_FEATURE_DETECTOR "GFTT"
#define FAST_FEATURE_DETECTOR "FAST"
#define SIFT_FEATURE_DETECTOR "SIFT"
#define SPEEDSIFT_FEATURE_DETECTOR "SPEEDSIFT"
#define SURF_FEATURE_DETECTOR "SURF"

#endif
