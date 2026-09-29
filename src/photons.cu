#include "photons.h"
#include "intersections.h" 
#include "utilities.h"    
#include "interactions.h"

#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <cstdio>
#include <cfloat>


// temp
__host__ __device__ inline thrust::default_random_engine makePhotonRandomEngine(int iter, int index, int depth)
{
    const unsigned int kPhotonStream = 9999u;
    unsigned int h = utilhash((1u << 31) | (kPhotonStream << 22) | static_cast<unsigned int>(iter))
        ^ utilhash(static_cast<unsigned int>(index))
        ^ utilhash(static_cast<unsigned int>(depth));
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

    thrust::default_random_engine rng = makePhotonRandomEngine(iter, idx, traceDepth);
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

    if (intersection.t > 0.0f) {
        Ray photonRay{ photon.origin, photon.direction };
        glm::vec3 hitPoint = getPointOnRay(photonRay, intersection.t);

        if (material.hasReflective || material.hasRefractive) { // glass, mirror
			photon.passedGlass = true;

            thrust::default_random_engine rng = makePhotonRandomEngine(iter, idx, photon.remainingBounces);
            float pdf;
            glm::vec3 bsdf = scatterRay(photonRay, hitPoint, intersection.surfaceNormal, material, pdf, rng);

            float cosTheta = glm::abs(glm::dot(intersection.surfaceNormal, photonRay.direction));
            photon.power *= bsdf * cosTheta / pdf; // throughput 

            photon.origin = photonRay.origin;
            photon.direction = photonRay.direction;
            photon.remainingBounces -= 1;

            photons[idx] = photon;
        }
        else { // diffuse 
            if (photon.passedGlass) {
                // todo test code 
                // project into screen space 
                // color pixel PINK vec3(1, 0, 1)
                glm::vec2 screenPos = projectToScreen(cam, hitPoint);

                if (screenPos.x >= 0 && screenPos.x < cam.resolution.x && screenPos.y >= 0 && screenPos.y < cam.resolution.y) {
                    int pixelIndex = (int)screenPos.y * cam.resolution.x + (int)screenPos.x;

                    //image[pixelIndex] = glm::vec3(1.0f, 0.0f, 1.0f) * static_cast<float>(iter);
                    //image[pixelIndex] += photon.power * static_cast<float>(iter);
                    glm::vec3 contribution = photon.power * (material.color / PI)*4000.0f;
                    atomicAdd(&image[pixelIndex].x, contribution.x);
                    atomicAdd(&image[pixelIndex].y, contribution.y);
                    atomicAdd(&image[pixelIndex].z, contribution.z);
                }
            }
            // todo terminate on first diffuse hit for now 
            photons[idx].remainingBounces = 0;
        }
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

    std::vector<PhotonLight> lights = scene->photonLights;
    if (lights.empty()) {
        // fallback
        printf("no photonmap light source\n");
        return;
    }

    float totalPower = 0.0f;
    for (const PhotonLight& light : lights) { totalPower += light.power; }

    // kernel to determine photon directions 
    Photon* dev_photons;
    cudaMalloc(&dev_photons, numPhotons * sizeof(Photon));
	ShadeableIntersection* dev_photon_intersections;
	cudaMalloc(&dev_photon_intersections, numPhotons * sizeof(ShadeableIntersection));


	// initialize photon directions
    int traceDepth = scene->state.traceDepth;
    const int blockSize = 128;

    // todo 
    int photonsAssigned = 0;
    for (size_t i = 0; i < lights.size(); i++)
    {
        const PhotonLight& light = lights[i];
        int count = (i + 1 < lights.size()) // last?
            ? static_cast<int>(numPhotons * (light.power / totalPower))
            : (numPhotons - photonsAssigned); // remainder
        if (count <= 0) { continue; }

        float phi = 4.0f * PI * light.power; // FLUX
        const int numBlocks = (count + blockSize - 1) / blockSize;
        kernGeneratePhotonDirections<<<numBlocks, blockSize>>>(
            count, iter, traceDepth, light.position, light.color * phi,
            dev_photons + photonsAssigned);
        checkCUDAError("kernGeneratePhotonDirections");

        photonsAssigned += count;
    }

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

