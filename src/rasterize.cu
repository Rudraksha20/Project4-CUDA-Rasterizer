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
#define BILNEARFILTERING 1

// This is the color used for solid coloring
#define COLOR glm::vec3(0.98f, 0.98f, 0.98f)

//  Shading
#define LAMBERT 1

// Back Face Culling
#define BACKFACECULLING 1

// Anti-Aliasing
#define FXAA 1

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
static int * dev_mutex = NULL;	// used for depth test without conflicts

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
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
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(dev_mutex);
	cudaMalloc(&dev_mutex, width * height * sizeof(int));
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

	cudaFree(dev_mutex);
	dev_mutex = NULL;

    checkCUDAError("rasterize Free");
}
