CUDA Rasterizer
===============

[CLICK ME FOR INSTRUCTION OF THIS PROJECT](./INSTRUCTION.md)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

* Rudraksha Shah
* Tested on: Windows 10, i7-7700HQ @ 2.80GHz 16GB, GTX 1050 4096MB (Personal Computer)

Overview
===========

In this projevcctct I have built a GPU Rasterizer using CUDA and C++.

The main features of the Rasterizer are as follows:

* Rasterization Methods:

    * Naive Barycentric Rasterization
    * Scan-Line Rasterization

* Rasterization Modes:

    * Points
    * Lines
    * Triangles    

* Texturing:

    * Perspective Correct Interpolation
    * Bi-Linear Filtering

* Shading:

    * Lambert

* Anti-Aliasing:

    * FXAA
    * SSAA

* Optimizations:

    * Back Face Culling

Rasterization is a way of rendering 3D graphics in which we project the geometry of the scene onto the screen.

Implementing the Rasterization on the CPU a year ago in CIS 460 was tricky but implementing the entire pipeline on the GPU was a challange in itself!

The basic process of Rasterization is to take 3d geometry in object space then take it from object space -> world space -> camera space -> un-homogenized projected space -> homogenized NDC space -> pixel space. Once the object is in 2D pixel space it is rendered onto the screen. For rendering I have implemented the following methods:

1. Barycentric Rasterization: In this method we iterate through each pizel in the bounding box surrounding the given triangle we are trying to render and check for each pixel if it lies inside the triangle or not using barycetric weights.

2. Scan-Line Rasterization: In this method we again start from the bounding box surrounding the triangle we are renderig but for each pixel row we find valid intersections with the triangle edges. Now we fill in those pixels from one point of intersection to the other. This way we do not have to spend time checking unnecessary pixels around the triangle.

Performance Analysis
=======================

* All performance analysis is done using the Duck gltf model and the Barycentric Rasterization process of rendering with solid color per vertex and lambert shading.

## No Anti Aliasing vs FXAA vs SSAA

| No Anti Aliasing | FXAA | SSAA |

| ---------------- | ---- | ---- |

| ![NO AA]() | ![FXAA]() | ![SSAA]() |

### Credits

* [Bresenham's Algo](https://www.cs.helsinki.fi/group/goa/mallinnus/lines/bresenh.html)
* [FXAA information](https://blog.codinghorror.com/fast-approximate-anti-aliasing-fxaa/)
* [FXAA original paper](http://developer.download.nvidia.com/assets/gamedev/files/sdk/11/FXAA_WhitePaper.pdf)
* [FXAA tutorial](http://blog.simonrodriguez.fr/articles/30-07-2016_implementing_fxaa.html)
* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
