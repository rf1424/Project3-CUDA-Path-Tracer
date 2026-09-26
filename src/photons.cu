#include "photons.h"
#include "intersections.h" // utilhash
#include "utilities.h"      // PI, TWO_PI

#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
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
    int traceDepth,
    glm::vec3 lightPos,
    glm::vec3 photonPower,
    Photon* photons
    )
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
	photon.passedGlass = false;
	photon.remainingBounces = traceDepth;

    photons[idx] = photon;
}

__global__ void computePhotonIntersections(
    int numPhotons,
    Photon* photons,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numPhotons)
    {
        Photon photon = photons[idx];

        Ray r;
        r.origin = photon.origin;
        r.direction = photon.direction;

        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min;
        int hit_geom_index;

        findClosestIntersection(r, geoms, geoms_size, t_min, intersect_point, normal, hit_geom_index);

        if (hit_geom_index == -1)
        {
            intersections[idx].t = -1.0f;
        }
        else
        {
            intersections[idx].t = t_min;
            intersections[idx].materialId = geoms[hit_geom_index].materialid;
            intersections[idx].surfaceNormal = normal;
        }
    }
}

__device__ glm::vec2 projectToScreen(
    const Camera& cam,
    const glm::vec3& pos)
{
    glm::vec3 viewHat = glm::normalize(cam.view);
    glm::vec3 Q = pos - cam.position;
    float z = glm::dot(Q, viewHat);

    if (z > 0.0f) {
        float qx = glm::dot(Q, cam.right);
        float qy = glm::dot(Q, cam.up);
        float viewLength = glm::length(cam.view);

        float px = cam.resolution.x * 0.5f - qx * viewLength / (z * cam.pixelLength.x);
        float py = cam.resolution.y * 0.5f - qy * viewLength / (z * cam.pixelLength.y);

        int x = static_cast<int>(px);
        int y = static_cast<int>(py);
        return glm::vec2(x, y);
    } else {
		return glm::vec2(-1.0f, -1.0f);
    }
}


__global__ void evalPhoton(Photon* photons, ShadeableIntersection* intersections, int numPhotons, Material* materials, Camera cam, glm::vec3* image, int iter) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPhotons) {
        return;
    }
    Photon photon = photons[idx];
    ShadeableIntersection intersection = intersections[idx];

	Material material = materials[intersection.materialId];

    if (intersection.t > 0.0f && material.hasRefractive) {
        // Handle photon interaction with the surface
        // todo 
        //test code
        // project into screen space 
        // color pixel PINK vec3(1, 0, 1)
        glm::vec3 hitPoint = getPointOnRay(Ray{ photon.origin, photon.direction }, intersection.t);
		glm::vec2 screenPos = projectToScreen(cam, hitPoint);

        if (screenPos.x >= 0 && screenPos.x < cam.resolution.x && screenPos.y >= 0 && screenPos.y < cam.resolution.y) {
            int pixelIndex = screenPos.y * cam.resolution.x + screenPos.x;
            image[pixelIndex] = glm::vec3(1.0f, 0.0f, 1.0f) * static_cast<float>(iter);
        }
        
        photons[idx].remainingBounces = 0;

    } else {
        // Photon did not hit any surface
		photons[idx].remainingBounces = 0; // terminated
    }
}

// for stream compaction 
// functor to check if a photon is terminated
struct isTerminatedPhoton {
    __host__ __device__ bool operator()(const Photon& p) const {
        return (p.remainingBounces <= 0);
    }
};


void photonMap(Scene* scene, int numPhotons, int iter, 
    Geom* geoms, int geomsSize, 
    Material* materials, int materialsSize,
    Camera cam, glm::vec3* image) {
    // temporarily set a point light 
    glm::vec3 pos = glm::vec3(0.0f, 9.0f, 0.0f);
    glm::vec3 color = glm::vec3(1.0f, 1.0f, 1.0f);
    float power = 5.0f;

    float phi = 4.0f * PI * power;// FLUX

    // kernel to determine photon directions 
    Photon* dev_photons;
    cudaMalloc(&dev_photons, numPhotons * sizeof(Photon));
	ShadeableIntersection* dev_photon_intersections;
	cudaMalloc(&dev_photon_intersections, numPhotons * sizeof(ShadeableIntersection));


	// initialize photon directions
    int traceDepth = 5;
    const int blockSize = 128;
    const int numBlocks = (numPhotons + blockSize - 1) / blockSize;
    kernGeneratePhotonDirections<<<numBlocks, blockSize>>>(numPhotons, iter, traceDepth, pos, color * phi, dev_photons);
    checkCUDAError("kernGeneratePhotonDirections");

    // photon tracing loop
    int depth = 0;
	bool iterationComplete = false;
	Photon* dev_photons_end = dev_photons + numPhotons;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_photon_intersections, 0, numPhotons * sizeof(ShadeableIntersection));

        dim3 numblocksPathSegmentTracing = (numPhotons + blockSize - 1) / blockSize;

        // 0. compute intersections -> dev_intersections
        computePhotonIntersections << <numblocksPathSegmentTracing, blockSize >> > (
            numPhotons,
            dev_photons,
            geoms,
            geomsSize,
            dev_photon_intersections
            );
        checkCUDAError("compute intersections");
        evalPhoton<<<numblocksPathSegmentTracing, blockSize>>> (
            dev_photons,
            dev_photon_intersections,
            numPhotons,
            materials,
            cam,
            image,
            iter
			);

        checkCUDAError("evalPhoton");
        cudaDeviceSynchronize();

        // stream compaction to remove terminated paths
        dev_photons_end = thrust::remove_if(thrust::device, dev_photons, dev_photons_end, isTerminatedPhoton());
		numPhotons = dev_photons_end - dev_photons;

        depth++;
        iterationComplete = (depth >= traceDepth) || (numPhotons <= 0);
    } // end of the bounce loop



    cudaFree(dev_photons);
	cudaFree(dev_photon_intersections);
}

