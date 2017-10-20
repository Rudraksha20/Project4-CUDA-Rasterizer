/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

// Render Modes
#define POINTS 0
#define LINES 0
#define TRIANGLES 1

// Rasterization Methods (Renders Solid Triangles)
#define NAIVE_EDGEINTERSECTION_SCANLINE_TOGGLE 0	// 0 - Naive scanline & 1 - Edge intersection scanline

// Coloring (Either of the two should be on to have an output on the screen)
#define SOLIDCOLOR 0
#define TEXTURING 1
#define PERSPECTIVECORRECTTEXTURING 1
#define BILNEARFILTERING 0
// This is the color used for solid coloring
#define COLOR glm::vec3(0.98f, 0.98f, 0.98f)

//  Shading
#define LAMBERT 1

// Back Face Culling
#define BACKFACECULLING 0

// Anti-Aliasing
#define FXAA 0	// Fast Approximation AA (Post Processing)
#define SSAA 0	// SSAA toggle
#define SSAAMULTIPLYER 4 // 1x is no SSAA. Increase this value to increase the resolution and SSAA effect

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		 glm::vec3 color;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		 int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
		bool culled;	// Used for triangle culling
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		glm::vec3 eyePos;	// eye space position used for shading
		glm::vec3 eyeNor;
		VertexAttributeTexcoord texcoord0;
		TextureData* dev_diffuseTex;
		int textureWidth;
		int textureHeight;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};
}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

static int * dev_depth = NULL;	// you might need this buffer when doing depth test

// Newly created
static int * dev_mutex = NULL;	// used for depth test without conflicts
static glm::vec3 *dev_temp_framebuffer = NULL;
float* dev_quality = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

#if SSAA
	
	int downWidth = w / SSAAMULTIPLYER;
	int downHeight = h / SSAAMULTIPLYER;
    int index = x + (y * downWidth);

    if (x < downWidth && y < downHeight) {
        glm::vec3 color;

		// Down sampling the image to the original size
		for (int i = 0; i < SSAAMULTIPLYER; i++) {
			for (int j = 0; j < SSAAMULTIPLYER; j++) {
				int idx = (x * SSAAMULTIPLYER) + i + (y * SSAAMULTIPLYER + j) * w;
				color.x += glm::clamp(image[idx].x, 0.0f, 1.0f) * 255.0;
				color.y += glm::clamp(image[idx].y, 0.0f, 1.0f) * 255.0;
				color.z += glm::clamp(image[idx].z, 0.0f, 1.0f) * 255.0;
			}
		}
		color /= (SSAAMULTIPLYER*SSAAMULTIPLYER);

		// Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }

#else
	
	int index = x + (y * w);

	if (x < w && y < h) {
		glm::vec3 color;
		
		color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
		color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
		color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
		
		// Each thread writes one pixel location in the texture (textel)
		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}

#endif
}

/**
* Bilnear filtering
*/
__device__
glm::vec3 bilinearFiltering(Fragment& tempfragment, float uvXf, float uvYf) {
	int uvX0 = uvXf;
	int uvY0 = uvYf;
	int uvX1 = glm::clamp(uvX0 + 1, 0, tempfragment.textureWidth - 1);
	int uvY1 = glm::clamp(uvY0 + 1, 0, tempfragment.textureHeight - 1);
	glm::vec3 finalColor;
	int uvIndex;
	
	// Get the color of the four surrounding pixels
	uvIndex = (uvX0 + (uvY0 * tempfragment.textureWidth)) * 3;
	glm::vec3 colorX0Y0 = (glm::vec3(tempfragment.dev_diffuseTex[uvIndex], tempfragment.dev_diffuseTex[uvIndex + 1], tempfragment.dev_diffuseTex[uvIndex + 2])) / 255.0f;
	
	uvIndex = (uvX1 + (uvY0 * tempfragment.textureWidth)) * 3;
	glm::vec3 colorX1Y0 = (glm::vec3(tempfragment.dev_diffuseTex[uvIndex], tempfragment.dev_diffuseTex[uvIndex + 1], tempfragment.dev_diffuseTex[uvIndex + 2])) / 255.0f;
	
	uvIndex = (uvX0 + (uvY1 * tempfragment.textureWidth)) * 3;
	glm::vec3 colorX0Y1 = (glm::vec3(tempfragment.dev_diffuseTex[uvIndex], tempfragment.dev_diffuseTex[uvIndex + 1], tempfragment.dev_diffuseTex[uvIndex + 2])) / 255.0f;
	
	uvIndex = (uvX1 + (uvY1 * tempfragment.textureWidth)) * 3;
	glm::vec3 colorX1Y1 = (glm::vec3(tempfragment.dev_diffuseTex[uvIndex], tempfragment.dev_diffuseTex[uvIndex + 1], tempfragment.dev_diffuseTex[uvIndex + 2])) / 255.0f;

	// Bilinearly in terpolate between the colors based on the fractional part in the uvs
	float weightY = uvYf - uvY0;
	float weightX = uvXf - uvX0;
	glm::vec3 interpColorY0 = glm::mix(colorX0Y0, colorX1Y0, weightX);
	glm::vec3 interpColorY1 = glm::mix(colorX0Y1, colorX1Y1, weightX);
	finalColor = glm::mix(interpColorY0, interpColorY1, weightY);

	// Return the final color
	return finalColor;
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
		glm::vec3 finalColor;
		Fragment& thisFragment = fragmentBuffer[index];

#if (POINTS || LINES)

		finalColor = thisFragment.color;
		finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
		framebuffer[index] = finalColor;

#elif TEXTURING
		
		if (thisFragment.dev_diffuseTex != NULL) {

#if BILNEARFILTERING

			float uvXf = thisFragment.texcoord0.x * thisFragment.textureWidth;
			float uvYf = thisFragment.texcoord0.y * thisFragment.textureHeight;

			finalColor = bilinearFiltering(thisFragment, uvXf, uvYf);

#else

			// Get color form the texture and store it.
			int uvX = thisFragment.texcoord0.x * thisFragment.textureWidth;
			int uvY = thisFragment.texcoord0.y * thisFragment.textureHeight;

			int uvIndex = (uvX + (uvY * thisFragment.textureWidth)) * 3;
			finalColor = glm::vec3(thisFragment.dev_diffuseTex[uvIndex], thisFragment.dev_diffuseTex[uvIndex + 1], thisFragment.dev_diffuseTex[uvIndex + 2]);
			finalColor /= 255.0f;

#endif
		
		}

#if LAMBERT

		// Diffuse/Lambert shading
		glm::vec3 LightDirection = glm::normalize(thisFragment.eyePos - glm::vec3(100.0f));
		finalColor *= (glm::dot(-LightDirection, thisFragment.eyeNor));

#endif

		finalColor = glm::clamp(finalColor, 0.0f, 1.0f);
		framebuffer[index] = finalColor;

#elif SOLIDCOLOR

#if LAMBERT

		// Lambert Shading
		finalColor = thisFragment.color;
		glm::vec3 LightDirection = glm::normalize(thisFragment.eyePos - glm::vec3(100.0f));
		finalColor *= (glm::dot(-LightDirection, thisFragment.eyeNor));

#endif

		finalColor = glm::clamp(finalColor , 0.1f, 1.0f);
		framebuffer[index] = finalColor;

#endif
		// TODO: add your fragment shader code here
    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
#if SSAA

	width = w * SSAAMULTIPLYER;
    height = h * SSAAMULTIPLYER;

#else

	width = w;
	height = h;

#endif

	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	// Newly Created
	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));

	cudaFree(dev_temp_framebuffer);
	cudaMalloc(&dev_temp_framebuffer, width * height * sizeof(glm::vec3));
	cudaMemset(dev_temp_framebuffer, 0, width * height * sizeof(glm::vec3));

	cudaFree(dev_quality);
	cudaMalloc(&dev_quality, 12 * sizeof(float));
	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}

__global__
void initMutex(int w, int h, int * mutex) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		mutex[index] = 0;
	}
}

/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		VertexOut* tempPtrToPrimitiveOutVertex = &primitive.dev_verticesOut[vid];

		glm::vec4 objSpacePos = glm::vec4(primitive.dev_position[vid], 1.0f);
		glm::vec4 tempPos = objSpacePos;

		// Object space to un-homogenized coordinates
		tempPos = MVP * tempPos;
		
		// re-homogenizing the coordinates
		tempPos /= tempPos[3];

		// NDC -> Pixel space
		tempPos[0] = (1.0f - tempPos[0]) * width / 2.0f;
		tempPos[1] = (1.0f - tempPos[1]) * height / 2.0f;
		tempPos[2] = -tempPos[2];

		// Fill in the out variables
		tempPtrToPrimitiveOutVertex->pos = tempPos;
		tempPtrToPrimitiveOutVertex->eyePos = glm::vec3(MV * objSpacePos);
		tempPtrToPrimitiveOutVertex->eyeNor = glm::normalize(MV_normal * primitive.dev_normal[vid]);

#if TEXTURING
		tempPtrToPrimitiveOutVertex->texcoord0 = primitive.dev_texcoord0[vid];	// These are UV's
		tempPtrToPrimitiveOutVertex->dev_diffuseTex = primitive.dev_diffuseTex;
		tempPtrToPrimitiveOutVertex->texWidth = primitive.diffuseTexWidth;
		tempPtrToPrimitiveOutVertex->texHeight = primitive.diffuseTexHeight;
#endif

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		
	}
}


static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType] = primitive.dev_verticesOut[primitive.dev_indices[iid]];
		
#if BACKFACECULLING
			glm::vec3 cameraDirection(0.0f, 0.0f, 1.0f);
			if (glm::dot(cameraDirection, primitive.dev_verticesOut[primitive.dev_indices[iid]].eyeNor) <= 0.0f) {
				dev_primitives[pid + curPrimitiveBeginId].culled = true;
			}
			else {
				dev_primitives[pid + curPrimitiveBeginId].culled = false;
			}
#endif

		}

		// TODO: other primitive types (point, line)
	}
	
}

/**
*	Draws a line given a line segment
*/
__device__
void drawLine(LineSegment LS, Fragment* fragmentBuffer, int width, int height) {
		// Bresenham's Algo.
		// Refrence: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
		// Refrence: https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm

		int x0 = LS.vertex1.x;
		int y0 = LS.vertex1.y;
		int x1 = LS.vertex2.x;
		int y1 = LS.vertex2.y;

		bool steep = (glm::abs(y1 - y0) > glm::abs(x1 - x0));
		int temp;
		if (steep) {
			temp = x0;
			x0 = y0;
			y0 = temp;

			temp = x1;
			x1 = y1;
			y1 = temp;
		}

		if (x0 > x1) {
			temp = x0;
			x0 = x1;
			x1 = temp;

			temp = y0;
			y0 = y1;
			y1 = temp;
		}
	
		float dx = x1 - x0;
		float dy = glm::abs(y1 - y0);

		float error = dx / 2.0f;
		int ystep = (y0 < y1) ? 1 : -1;
		int y = (int)y0;

		int maxX = (int)x1;

		for (int x = (int)x0; x<maxX; x++)
		{
			if (steep)
			{
				if ((x <= width && x >= 0) && (y <= height && y >= 0)) {
					int pixelIndexP = y + (x * width);
					fragmentBuffer[pixelIndexP].color = COLOR;
				}
			}
			else
			{
				if ((x <= width && x >= 0) && (y <= height && y >= 0)) {
					int pixelIndexP = x + (y * width);
					fragmentBuffer[pixelIndexP].color = COLOR;
				}
			}

			error -= dy;
			if (error < 0)
			{
				y += ystep;
				error += dx;
			}
		}
}

__global__
void _rasterizeGeometry(int totalNumPrimitives, Primitive* dev_primitives, Fragment* dev_fragmentBuffer, int* dev_depth, int* dev_mutex , int width, int height) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= totalNumPrimitives) {
		return;
	}

	Primitive& tempPrimitive = dev_primitives[index];

#if BACKFACECULLING
	if (tempPrimitive.culled == true) {
		return;
	}
#endif

	// Create a trinagle
	glm::vec3 thisTriangle[3] = { glm::vec3(tempPrimitive.v[0].pos),
								  glm::vec3(tempPrimitive.v[1].pos),
								  glm::vec3(tempPrimitive.v[2].pos) };

	// Create a axis aligned bounding box
	AABB aabb = getAABBForTriangle(thisTriangle);

#if POINTS

	for (int i = 0; i < 3; i++) {
		int x = (int)thisTriangle[i].x;
		int y = (int)thisTriangle[i].y;
		if ((x <= width && x >= 0) && (y <= height && y >= 0)) {
			int pixelIndexP = x + (y * width);
			dev_fragmentBuffer[pixelIndexP].color = COLOR;
		}
	}

#elif LINES
		
	// Create LineSegments from vertices
	LineSegment LS1 = createLineSegment(thisTriangle[0], thisTriangle[1]);
	LineSegment LS2 = createLineSegment(thisTriangle[0], thisTriangle[2]);
	LineSegment LS3 = createLineSegment(thisTriangle[1], thisTriangle[2]);

	// Draw these lines
	drawLine(LS1, dev_fragmentBuffer, width, height);
	drawLine(LS2, dev_fragmentBuffer, width, height);
	drawLine(LS3, dev_fragmentBuffer, width, height);

#elif TRIANGLES

#if NAIVE_EDGEINTERSECTION_SCANLINE_TOGGLE	
	
	// Do ScanLine Edge Intersection Rasterization
	
	// Fill bounds and Clip them to screen size
	float maxY = glm::min(aabb.max[1], (float)height);	
	float minY = glm::max(aabb.min[1], 0.0f);

	// Create LineSegments from vertices
	LineSegment LS1 = createLineSegment(thisTriangle[0], thisTriangle[1]);
	LineSegment LS2 = createLineSegment(thisTriangle[0], thisTriangle[2]);
	LineSegment LS3 = createLineSegment(thisTriangle[1], thisTriangle[2]);

	for (int i = minY; i <= maxY; i++) {
		// Check for intersections and find the minX and maxX value for each pixel row
		float minX = FLT_MAX;
		float maxX = FLT_MIN;
		int intersectionCount = 0;
		
		if (intersectWithLineSegemnt(LS1, i, minX, maxX, aabb)) {
			intersectionCount++;
		}

		if (intersectWithLineSegemnt(LS2, i, minX, maxX, aabb)) {
			intersectionCount++;
		}
		
		if (intersectWithLineSegemnt(LS3, i, minX, maxX, aabb)) {
			intersectionCount++;
		}

		if (intersectionCount < 2) {
			continue;
		}
		
		// Clip them to the screen size
		minX = glm::max(minX, 0.0f);
		maxX = glm::min(maxX, (float)width);
			
		for (int j = minX; j <= maxX; j++) {
				
			// Get the baricentric coordinate for position x, y (j, i)
			glm::vec3 baryCentricCoordinate = calculateBarycentricCoordinate(thisTriangle, glm::vec2(j, i));
	
			if (!isBarycentricCoordInBounds(baryCentricCoordinate)) {
				continue;
			}

			int perspectiveCorrectZ = getZAtCoordinate(baryCentricCoordinate, thisTriangle) * 10000;

			int pixelIndex = j + (i * width);

			bool depthUpdated = fillDepthBufferWithMinValue(&dev_mutex[pixelIndex], &dev_depth[pixelIndex], perspectiveCorrectZ);

			if (depthUpdated) {
				// Interpolating the eye normals and the positions used for shading later
				dev_fragmentBuffer[pixelIndex].eyePos = tempPrimitive.v[0].eyePos * baryCentricCoordinate.x
					+ tempPrimitive.v[1].eyePos * baryCentricCoordinate.y
					+ tempPrimitive.v[2].eyePos * baryCentricCoordinate.z;
				dev_fragmentBuffer[pixelIndex].eyeNor = tempPrimitive.v[0].eyeNor * baryCentricCoordinate.x
					+ tempPrimitive.v[1].eyeNor * baryCentricCoordinate.y
					+ tempPrimitive.v[2].eyeNor * baryCentricCoordinate.z;

#if SOLIDCOLOR
				dev_fragmentBuffer[pixelIndex].color = COLOR;
#elif TEXTURING

#if PERSPECTIVECORRECTTEXTURING
				float z0 = tempPrimitive.v[0].eyePos.z;
				float z1 = tempPrimitive.v[1].eyePos.z;
				float z2 = tempPrimitive.v[2].eyePos.z;

				// Correctly interpolated z value
				float z = baryCentricCoordinate.x / z0 + baryCentricCoordinate.y / z1 + baryCentricCoordinate.z / z2;

				// Perspective corrected texture coordinates
				dev_fragmentBuffer[pixelIndex].texcoord0 = (tempPrimitive.v[0].texcoord0 / z0 * baryCentricCoordinate.x
															+ tempPrimitive.v[1].texcoord0 / z1 * baryCentricCoordinate.y
															+ tempPrimitive.v[2].texcoord0 / z2 * baryCentricCoordinate.z) / z;
#else
				dev_fragmentBuffer[pixelIndex].texcoord0 = tempPrimitive.v[0].texcoord0 * baryCentricCoordinate.x
															+ tempPrimitive.v[1].texcoord0 * baryCentricCoordinate.y
															+ tempPrimitive.v[2].texcoord0 * baryCentricCoordinate.z;
#endif
				dev_fragmentBuffer[pixelIndex].textureWidth = tempPrimitive.v[0].texWidth;
				dev_fragmentBuffer[pixelIndex].textureHeight = tempPrimitive.v[0].texHeight;
				dev_fragmentBuffer[pixelIndex].dev_diffuseTex = tempPrimitive.v[0].dev_diffuseTex;

#endif
			}

		}
	
	}
#else	
	
	// Do Scanline Naive Rasterization

	float minX = glm::max(aabb.min[0], 0.0f);
	float maxX = glm::min(aabb.max[0], (float)(width - 1));
	float minY = glm::max(aabb.min[1], 0.0f);
	float maxY = glm::min(aabb.max[1], (float)(height - 1));

	for (int y = minY; y <= maxY; y++) {
		for (int x = minX; x <= maxX; x++) {

			// Get the baricentric coordinate for position x, y on screen
			glm::vec3 baryCentricCoordinate = calculateBarycentricCoordinate(thisTriangle, glm::vec2(x, y));

			if (!isBarycentricCoordInBounds(baryCentricCoordinate)) {
				continue;
			}

			int perspectiveCorrectZ = getZAtCoordinate(baryCentricCoordinate, thisTriangle) * 10000;
	
			int pixelIndex = x + (y * width);

			bool depthUpdated = fillDepthBufferWithMinValue(&dev_mutex[pixelIndex], &dev_depth[pixelIndex], perspectiveCorrectZ);

			if (depthUpdated) {
				// Interpolating the eye normals and the positions used for shading later
				dev_fragmentBuffer[pixelIndex].eyePos = tempPrimitive.v[0].eyePos * baryCentricCoordinate.x
					+ tempPrimitive.v[1].eyePos * baryCentricCoordinate.y
					+ tempPrimitive.v[2].eyePos * baryCentricCoordinate.z;
				dev_fragmentBuffer[pixelIndex].eyeNor = tempPrimitive.v[0].eyeNor * baryCentricCoordinate.x
					+ tempPrimitive.v[1].eyeNor * baryCentricCoordinate.y
					+ tempPrimitive.v[2].eyeNor * baryCentricCoordinate.z;

#if SOLIDCOLOR
				dev_fragmentBuffer[pixelIndex].color = COLOR;
#elif TEXTURING

#if PERSPECTIVECORRECTTEXTURING
				float z0 = tempPrimitive.v[0].eyePos.z;
				float z1 = tempPrimitive.v[1].eyePos.z;
				float z2 = tempPrimitive.v[2].eyePos.z;

				// Correctly interpolated z value
				float z = baryCentricCoordinate.x / z0 + baryCentricCoordinate.y / z1 + baryCentricCoordinate.z / z2;

				// Perspective corrected texture coordinates
				dev_fragmentBuffer[pixelIndex].texcoord0 = (tempPrimitive.v[0].texcoord0 / z0 * baryCentricCoordinate.x 
															+ tempPrimitive.v[1].texcoord0 / z1 * baryCentricCoordinate.y 
															+ tempPrimitive.v[2].texcoord0 / z2 * baryCentricCoordinate.z) / z;
#else
				dev_fragmentBuffer[pixelIndex].texcoord0 = tempPrimitive.v[0].texcoord0 * baryCentricCoordinate.x 
															+ tempPrimitive.v[1].texcoord0 * baryCentricCoordinate.y 
															+ tempPrimitive.v[2].texcoord0 * baryCentricCoordinate.z;
#endif
				dev_fragmentBuffer[pixelIndex].textureWidth = tempPrimitive.v[0].texWidth;
				dev_fragmentBuffer[pixelIndex].textureHeight = tempPrimitive.v[0].texHeight;
				dev_fragmentBuffer[pixelIndex].dev_diffuseTex = tempPrimitive.v[0].dev_diffuseTex;

#endif
			}
	
		}
	}
#endif

#endif
}


/**
*	Perform FXAA
*	Reference: https://blog.codinghorror.com/fast-approximate-anti-aliasing-fxaa/
*/
__global__
void FXAAkern(int width, int height, glm::vec3* dev_framebuffer, glm::vec3* dev_temp_framebuffer, float FXAA_SPAN_MAX, float FXAA_EDGE_THRESHOLD_MAX, float FXAA_EDGE_THRESHOLD_MIN, float* dev_quality) {
	int x0 = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y0 = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x0 + (y0 * width);

	if (x0 < width && y0 < height) {
		int x1 = glm::clamp(x0 + 1, 0, width - 1);
		int y1 = glm::clamp(y0 + 1, 0, height - 1);
		int xm1 = glm::clamp(x0 - 1, 0, width - 1);
		int ym1 = glm::clamp(y0 - 1, 0, height - 1);

		// uv Index of the four pixels on the sides of a given pixel
		int uvIndexUp = x0 + (y1 * width);
		int uvIndexDown = x0 + (ym1 * width);
		int uvIndexLeft = xm1 + (y0 * width);
		int uvIndexRight = x1 + (y0 * width);

		// uv index of the four pixels in the corner around the given pixel
		int uvIndexUpLeft = xm1 + (y1 * width);
		int uvIndexDownLeft = xm1 + (ym1 * width);
		int uvIndexUpRight = x1 + (y1 * width);
		int uvIndexDownRight = x1 + (ym1 * width);

		// Standard luminosity values of RGB based on the percieption of individual colors by humans
		glm::vec3 luma(0.299, 0.587, 0.114);

		// Luminosity at the given pixel index
		float lumaCenter = glm::dot(dev_framebuffer[index] , luma);

		// Find the luminosity of the texture in the surrounding four pixels
		float lumaUp = glm::dot(dev_framebuffer[uvIndexUp], luma);
		float lumaDown = glm::dot(dev_framebuffer[uvIndexDown], luma);
		float lumaRight = glm::dot(dev_framebuffer[uvIndexRight], luma);
		float lumaLeft = glm::dot(dev_framebuffer[uvIndexLeft], luma);
	
		// Find the luminosity of the four corners around a given pixel
		// These four values combined with the above values will be used to determine if an edge is horizontal or vertical
		float lumaUpLeft = glm::dot(dev_framebuffer[uvIndexUpLeft], luma);
		float lumaUpRight = glm::dot(dev_framebuffer[uvIndexUpRight], luma);
		float lumaDownLeft = glm::dot(dev_framebuffer[uvIndexDownLeft], luma);
		float lumaDownRight = glm::dot(dev_framebuffer[uvIndexDownRight], luma);

		// Check if we are in a region which needs to be AA'ed
		
		// find the min and max luminosity around a given fragmnet
		float lumaMin = glm::min(lumaCenter, glm::min(glm::min(lumaUp, lumaDown), glm::min(lumaRight, lumaLeft)));
		float lumaMax = glm::max(lumaCenter, glm::max(glm::max(lumaUp, lumaDown), glm::max(lumaRight, lumaLeft)));

		// Find the deviation (DELTA) of the luminosity for deciding if there is a significant edge to perform AA around the given pixel index
		float delta = lumaMax - lumaMin;

		// If the deviation is not significant enough don't bother doing AA
		if (delta < glm::max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD_MAX)) {
			dev_temp_framebuffer[index] = dev_framebuffer[index];
			return;
		}

		// Combine the lumas
		// Edge
		float lumaDownUp = lumaDown + lumaUp;
		float lumaLeftRight = lumaLeft + lumaRight;
		// Corners
		float lumaLeftCorners = lumaDownLeft + lumaUpLeft;
		float lumaDownCorners = lumaDownLeft + lumaDownRight;
		float lumaRightCorners = lumaDownRight + lumaUpRight;
		float lumaUpCorners = lumaUpRight + lumaUpLeft;

		// Compute an estimation of the gradient along the horizontal and vertical axis.
		float edgeHorizontal = glm::abs(-2.0 * lumaLeft + lumaLeftCorners) + glm::abs(-2.0 * lumaCenter + lumaDownUp) * 2.0 + glm::abs(-2.0 * lumaRight + lumaRightCorners);
		float edgeVertical = glm::abs(-2.0 * lumaUp + lumaUpCorners) + glm::abs(-2.0 * lumaCenter + lumaLeftRight) * 2.0 + glm::abs(-2.0 * lumaDown + lumaDownCorners);

		// Is the local edge horizontal or vertical ?
		bool isHorizontal = (edgeHorizontal >= edgeVertical);

		// Select the two neighboring texels lumas in the opposite direction to the local edge.
		float luma1 = isHorizontal ? lumaDown : lumaLeft;
		float luma2 = isHorizontal ? lumaUp : lumaRight;
		// Compute gradients in this direction.
		float gradient1 = luma1 - lumaCenter;
		float gradient2 = luma2 - lumaCenter;

		// Which direction is the steepest ?
		bool is1Steepest = glm::abs(gradient1) >= glm::abs(gradient2);

		// Gradient in the corresponding direction, normalized.
		float gradientScaled = 0.25*glm::max(abs(gradient1), abs(gradient2));


		// Choose the step size (one pixel) according to the edge direction.
		float stepLength = isHorizontal ? (1.0f/height) : (1.0f/width);

		// Average luma in the correct direction.
		float lumaLocalAverage = 0.0;

		if (is1Steepest) {
			// Switch the direction
			stepLength = -stepLength;
			lumaLocalAverage = 0.5*(luma1 + lumaCenter);
		}
		else {
			lumaLocalAverage = 0.5*(luma2 + lumaCenter);
		}

		// Shift UV in the correct direction by half a pixel.
		glm::vec2 currentUV = glm::vec2(x0, y0);
		if (isHorizontal) {
			currentUV.y += stepLength * 0.5;
		}
		else {
			currentUV.x += stepLength * 0.5;
		}

		// Exploer the edge on both sides and find the endpoint
		// Do the first iteration and you are done if you find the luminosity gradient is significant
		// Compute offset (for each iteration step) in the correct direction.
		glm::vec2 offset = isHorizontal ? glm::vec2((1.0/width), 0.0) : glm::vec2(0.0, (1.0f/height));
		// Compute UVs to explore on each side of the edge, orthogonally. 
		// The QUALITY allows us to step faster.
		glm::vec2 uv1 = currentUV - offset;
		glm::vec2 uv2 = currentUV + offset;

		// Read the lumas at both current extremities of the exploration segment, and compute the delta wrt to the local average luma.
		float lumaEnd1 = glm::dot(dev_framebuffer[(int)uv1.x + ((int)uv1.y * width)], luma);//rgb2luma(texture(screenTexture, uv1).rgb);
		float lumaEnd2 = glm::dot(dev_framebuffer[(int)uv2.x + ((int)uv2.y * width)], luma);//rgb2luma(texture(screenTexture, uv2).rgb);
		lumaEnd1 -= lumaLocalAverage;
		lumaEnd2 -= lumaLocalAverage;

		// If the luma deltas at the current extremities are larger than the local gradient, we have reached the side of the edge.
		bool reached2 = glm::abs(lumaEnd2) >= gradientScaled;
		bool reached1 = glm::abs(lumaEnd1) >= gradientScaled;
		bool reachedBoth = reached1 && reached2;

		// If the side is not reached, we continue to explore in this direction.
		if (!reached1) {
			uv1 -= offset;
		}
		if (!reached2) {
			uv2 += offset;
		}

		// Itereating
		if(!reachedBoth) {
			for (int i = 1; i < FXAA_SPAN_MAX; i++) {
			
				// If needed, read luma in 1st direction, compute delta.
				if (!reached1) {
					lumaEnd1 = glm::dot(dev_framebuffer[(int)uv1.x + ((int)uv1.y * width)], luma);// rgb2luma(texture(screenTexture, uv1).rgb);
					lumaEnd1 = lumaEnd1 - lumaLocalAverage;
				}
				// If needed, read luma in opposite direction, compute delta.
				if (!reached2) {
					lumaEnd2 = glm::dot(dev_framebuffer[(int)uv2.x + ((int)uv2.y * width) ], luma);// rgb2luma(texture(screenTexture, uv2).rgb);
					lumaEnd2 = lumaEnd2 - lumaLocalAverage;
				}
				// If the luma deltas at the current extremities is larger than the local gradient, we have reached the side of the edge.
				reached1 = abs(lumaEnd1) >= gradientScaled;
				reached2 = abs(lumaEnd2) >= gradientScaled;
				reachedBoth = reached1 && reached2;

				// If the side is not reached, we continue to explore in this direction, with a variable quality.
				if (!reached1) {
					uv1 -= offset * dev_quality[i];
				}
				if (!reached2) {
					uv2 += offset * dev_quality[i];
				}

				if (reachedBoth) {
					break;
				}
			}
		}
		// Done iterating

		// Now we estimate the offset if we are at the center of the edge or near the far sides.
		// The closer we are to the far sides the more blurring will need to be done to make the edge look smooth

		// Compute the distances to each extremity of the edge.
		float distance1 = isHorizontal ? (x0 - uv1.x) : (y0 - uv1.y);
		float distance2 = isHorizontal ? (uv2.x - x0) : (uv2.y - y0);

		// In which direction is the extremity of the edge closer ?
		bool isDirection1 = distance1 < distance2;
		float distanceFinal = glm::min(distance1, distance2);

		// Length of the edge.
		float edgeThickness = (distance1 + distance2);

		// UV offset: read in the direction of the closest side of the edge.
		float pixelOffset = -distanceFinal / edgeThickness + 0.5;

		// Now check if the luminosity of the center pixe; corrosponds to that on the edges detected
		// If not than we may have stepped too far

		// Is the luma at center smaller than the local average ?
		bool isLumaCenterSmaller = lumaCenter < lumaLocalAverage;

		// If the luma at center is smaller than at its neighbour, the delta luma at each end should be positive (same variation).
		// (in the direction of the closer side of the edge.)
		bool correctVariation = ((isDirection1 ? lumaEnd1 : lumaEnd2) < 0.0) != isLumaCenterSmaller;

		// If the luma variation is incorrect, do not offset.
		float finalOffset = correctVariation ? pixelOffset : 0.0;

		// sub-pixel AA
		float lumaAverage = (1.0 / 12.0) * (2.0 * (lumaDownUp + lumaLeftRight) + lumaLeftCorners + lumaRightCorners);
		// Ratio of the delta between the global average and the center luma, over the luma range in the 3x3 neighborhood.
		float subPixelOffset1 = glm::clamp(glm::abs(lumaAverage - lumaCenter) / delta, 0.0f, 1.0f);
		float subPixelOffset2 = (-2.0 * subPixelOffset1 + 3.0) * subPixelOffset1 * subPixelOffset1;
		// Compute a sub-pixel offset based on this delta.
		float subPixelOffsetFinal = subPixelOffset2 * subPixelOffset2 * 0.75;// SUBPIXEL_QUALITY;

		// Pick the biggest of the two offsets.
		finalOffset = glm::max(finalOffset, subPixelOffsetFinal);

		// Compute the final UV coordinates.
		glm::vec2 finalUv = glm::vec2(x0, y0);
		if (isHorizontal) {
			finalUv.y += finalOffset * stepLength;
		}
		else {
			finalUv.x += finalOffset * stepLength;
		}

		// Read the color at the new UV coordinates, and use it.
		dev_temp_framebuffer[index] = dev_framebuffer[(int)finalUv.x + ((int)finalUv.y * width)];
	}
}

/**
*	Copy data from dev_temp_framebuffer to dev_framebuffer
*/
__global__
void copyKern(int width, int height, glm::vec3* dev_framebuffer, glm::vec3* dev_temp_framebuffer) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int index = x + (y * width);

	if (x < width && y < height) {
		dev_framebuffer[index] = dev_temp_framebuffer[index];
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);
	initMutex << <blockCount2d, blockSize2d >> >(width, height, dev_mutex);

	// TODO: rasterize
	dim3 blockSize(128);
	dim3 numBlocksPerTriangle((totalNumPrimitives + blockSize.x - 1) / blockSize.x);
	_rasterizeGeometry<<<numBlocksPerTriangle, blockSize>>>(totalNumPrimitives, dev_primitives, dev_fragmentBuffer, dev_depth, dev_mutex,  width, height);
	checkCUDAError("rasterize geometry");

    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");

// Anti-Aliasing Effect using post processing Fast Approximation AA
#if FXAA

	// Do FXAA
	// Fillup the quality array used for iterating over the pixel edge
	float quality[12] = {1.5, 2.0, 2.0, 2.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0};
	cudaMemcpy(dev_quality, &quality, 12 * sizeof(float), cudaMemcpyHostToDevice);

	float FXAA_SPAN_MAX = 12.0;	// This is the number of steps about a given pixel we will take at a given time 
	float FXAA_EDGE_THRESHOLD_MAX = 1.0 / 8.0;
	float FXAA_EDGE_THRESHOLD_MIN = 0.0312;
	FXAAkern << <blockCount2d, blockSize2d>> > (width, height, dev_framebuffer, dev_temp_framebuffer, FXAA_SPAN_MAX, FXAA_EDGE_THRESHOLD_MAX, FXAA_EDGE_THRESHOLD_MIN, dev_quality);
	checkCUDAError("Apply FXAA");

	// Copy dev_temp_Framebuffer to dev_framebuffer
	copyKern << <blockCount2d, blockSize2d>> > (width, height, dev_framebuffer, dev_temp_framebuffer);
	checkCUDAError("copy dev_temp_framebuffer to dev_fraebuffer");

#endif

    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

	// Newly Added
	cudaFree(dev_mutex);
	dev_mutex = NULL;

	cudaFree(dev_temp_framebuffer);
	dev_temp_framebuffer = NULL;

	cudaFree(dev_quality);
	dev_quality = NULL;

    checkCUDAError("rasterize Free");
}
