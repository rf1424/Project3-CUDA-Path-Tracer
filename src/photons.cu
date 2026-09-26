#include "photons.h"
#include "intersections.h" // utilhash
#include "utilities.h"      // PI, TWO_PI

#include <thrust/random.h>
#include <cstdio>

// temp
__host__ __device__ inline thrust::default_random_engine makePhotonRandomEngine(int iter, int index)
{
    const unsigned int kPhotonStream = 9999u;
    unsigned int h = utilhash((1u << 31) | (kPhotonStream << 22) | static_cast<unsigned int>(iter))
        ^ utilhash(static_cast<unsigned int>(index));
    return thrust::default_random_engine(h);
}

__global__ void kernGeneratePhotonDirections(
    int numPhotons,
    int iter,
    glm::vec3 lightPos,
    glm::vec3 photonPower,
    Photon* photons)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPhotons)
    {
        return;
    }

    thrust::default_random_engine rng = makePhotonRandomEngine(iter, idx);
    thrust::uniform_real_distribution<float> u01(0, 1);

    // Uniform point on the unit sphere temp
    float z = 1.0f - 2.0f * u01(rng);
    float r = sqrtf(glm::max(0.0f, 1.0f - z * z));
    float angle = TWO_PI * u01(rng);
    glm::vec3 direction(r * cosf(angle), r * sinf(angle), z);

    Photon photon;
    photon.origin = lightPos;
    photon.direction = direction;
    photon.power = photonPower / static_cast<float>(numPhotons);

    photons[idx] = photon;
}


void photonMap(Scene* scene, int numPhotons, int iter) {
    // temporarily set a point light 
    glm::vec3 pos = glm::vec3(0.0f, 5.0f, 0.0f);
    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
    float power = 5.0f;

    float phi = 4.0f * PI * power;// FLUX

    // kernel to determine photon directions 
    Photon* dev_photons;
    cudaMalloc(&dev_photons, numPhotons * sizeof(Photon));

    const int blockSize = 128;
    const int numBlocks = (numPhotons + blockSize - 1) / blockSize;
    kernGeneratePhotonDirections<<<numBlocks, blockSize>>>(numPhotons, iter, pos, color * phi, dev_photons);
    checkCUDAError("kernGeneratePhotonDirections");

    cudaFree(dev_photons);
}

