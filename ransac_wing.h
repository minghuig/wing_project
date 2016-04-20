#ifndef __RANSAC_H
#define __RANSAC_H

int ransacWing(_In_ CvPoint3D32f *pointCloud, _InOut_ char *validFeatureIndicator, _In_ int numPoints, _Out_ char *errorMessage);

#endif