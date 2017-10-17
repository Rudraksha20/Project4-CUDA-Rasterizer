/**
 * @file      rasterizeTools.h
 * @brief     Tools/utility functions for rasterization.
 * @authors   Yining Karl Li
 * @date      2012-2015
 * @copyright University of Pennsylvania
 */

#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <util/utilityCore.hpp>

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

struct LineSegment {
	glm::vec3 vertex1;
	glm::vec3 vertex2;

	float slope;

	float minY;
	float maxY;
};

/*
* Checks if two floats are within threshold of each other
* i.e. Nearly equal
*/
__host__ __device__ static
bool nearlyEqual(float f1, float f2) {
	if (f1 > (f2 - SMALL_EPSILON) && f1 < (f2 + SMALL_EPSILON)) {
		return true;
	}
	else {
		return false;
	}
}

/*
* Gets the slope of the line segment
*/
__host__ __device__ static
float getLineSegmentSlope(glm::vec3 point1, glm::vec3 point2) {
	// HORIZONTAL LINE
	if (nearlyEqual(point1[1], point2[1])) {
		return 0.0f;
	}
	// VERTICLE LINE
	else if (nearlyEqual(point1[0], point2[0])) {
		return FLT_MAX;
	}
	else {
		return ((point2[1] - point1[1]) / (point2[0] - point1[0]));
	}
}

/**
* Initializes a line segment
*/
__host__ __device__ static
LineSegment createLineSegment(glm::vec3 point1, glm::vec3 point2) {
	LineSegment LS;

	LS.slope = getLineSegmentSlope(point1, point2);
	
	LS.minY = glm::min(point1[1], point2[1]);
	LS.maxY = glm::max(point1[1], point2[1]);

	LS.vertex1 = point1;
	LS.vertex2 = point2;

	return LS;
}

/*
* Finds intersection with line segment
*/
__host__ __device__ static
bool intersectWithLineSegemnt(LineSegment LS, int Y, float& minX, float& maxX, AABB aabb) {
	if ((float)Y < LS.minY || (float)Y > LS.maxY) {
		return false;
	}
	
	if (LS.slope == 0) {
		// HORIZONTAL LINE
		minX = glm::min(minX, glm::min(LS.vertex1[0], LS.vertex2[0]));
		maxX = glm::max(maxX, glm::max(LS.vertex1[0], LS.vertex2[0]));
		return true;
	}
	else if (LS.slope == FLT_MAX) {
		// VERTICAL LINE
		minX = glm::min(minX, glm::min(LS.vertex1[0], LS.vertex2[0]));
		maxX = glm::max(maxX, glm::max(LS.vertex1[0], LS.vertex2[0]));
		return true;
	}
	else {
		// P(X,Y) -> Point of intersection
		float X = (Y - LS.vertex1[1]) / LS.slope + LS.vertex1[0];
		if (X < aabb.min[0] || X > aabb.max[0]) {
			return false;
		}

		minX = glm::min(minX, X);
		maxX = glm::max(maxX, X);
		return true;
	}
}

/**
 * Multiplies a glm::mat4 matrix and a vec4.
 */
__host__ __device__ static
glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Finds the axis aligned bounding box for a given triangle.
 */
__host__ __device__ static
AABB getAABBForTriangle(const glm::vec3 tri[3]) {
    AABB aabb;
    aabb.min = glm::vec3(
            min(min(tri[0].x, tri[1].x), tri[2].x),
            min(min(tri[0].y, tri[1].y), tri[2].y),
            min(min(tri[0].z, tri[1].z), tri[2].z));
    aabb.max = glm::vec3(
            max(max(tri[0].x, tri[1].x), tri[2].x),
            max(max(tri[0].y, tri[1].y), tri[2].y),
            max(max(tri[0].z, tri[1].z), tri[2].z));
    return aabb;
}

// CHECKITOUT
/**
 * Calculate the signed area of a given triangle.
 */
__host__ __device__ static
float calculateSignedArea(const glm::vec3 tri[3]) {
    return 0.5 * ((tri[2].x - tri[0].x) * (tri[1].y - tri[0].y) - (tri[1].x - tri[0].x) * (tri[2].y - tri[0].y));
}

// CHECKITOUT
/**
 * Helper function for calculating barycentric coordinates.
 */
__host__ __device__ static
float calculateBarycentricCoordinateValue(glm::vec2 a, glm::vec2 b, glm::vec2 c, const glm::vec3 tri[3]) {
    glm::vec3 baryTri[3];
    baryTri[0] = glm::vec3(a, 0);
    baryTri[1] = glm::vec3(b, 0);
    baryTri[2] = glm::vec3(c, 0);
    return calculateSignedArea(baryTri) / calculateSignedArea(tri);
}

// CHECKITOUT
/**
 * Calculate barycentric coordinates.
 */
__host__ __device__ static
glm::vec3 calculateBarycentricCoordinate(const glm::vec3 tri[3], glm::vec2 point) {
    float beta  = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), point, glm::vec2(tri[2].x, tri[2].y), tri);
    float gamma = calculateBarycentricCoordinateValue(glm::vec2(tri[0].x, tri[0].y), glm::vec2(tri[1].x, tri[1].y), point, tri);
    float alpha = 1.0 - beta - gamma;
    return glm::vec3(alpha, beta, gamma);
}

// CHECKITOUT
/**
 * Check if a barycentric coordinate is within the boundaries of a triangle.
 */
__host__ __device__ static
bool isBarycentricCoordInBounds(const glm::vec3 barycentricCoord) {
    return barycentricCoord.x >= 0.0 && barycentricCoord.x <= 1.0 &&
           barycentricCoord.y >= 0.0 && barycentricCoord.y <= 1.0 &&
           barycentricCoord.z >= 0.0 && barycentricCoord.z <= 1.0;
}

// CHECKITOUT
/**
 * For a given barycentric coordinate, compute the corresponding z position
 * (i.e. depth) on the triangle.
 */
__host__ __device__ static
float getZAtCoordinate(const glm::vec3 barycentricCoord, const glm::vec3 tri[3]) {
    return -(barycentricCoord.x * tri[0].z
           + barycentricCoord.y * tri[1].z
           + barycentricCoord.z * tri[2].z);
}

/**
*	Fills the depth buffer without race conditions or memory erite conflicts
*/
__device__
bool fillDepthBufferWithMinValue(int* mutex, int* dev_depth, int perspectiveCorrectZ) {
	// Loop-wait until this thread is able to execute its critical section.
	if (perspectiveCorrectZ > *dev_depth) {
		return false;
	}

	bool depthUpdated = false;

	bool isSet;
	do {
		isSet = (atomicCAS(mutex, 0, 1) == 0);
		if (isSet) {
			// Critical section goes here.
			// The critical section MUST be inside the wait loop;
			// if it is afterward, a deadlock will occur.
			if (perspectiveCorrectZ < *dev_depth) {
				*dev_depth = perspectiveCorrectZ;
				depthUpdated = true;
			}
		}
		if (isSet) {
			*mutex = 0;
		}
	} while (!isSet);

	return depthUpdated;
}