
#include "sdf.h"
#include "utilities.h"
#include <glm/gtc/matrix_transform.hpp>


// sdfs from iq
static __host__ __device__ float boxSDF(glm::vec3 p, glm::vec3 b)
{
	glm::vec3 q = glm::abs(p) - b;
	return glm::length(glm::max(q, glm::vec3(0.0f))) + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

static __host__ __device__ float sphereSDF(glm::vec3 p)
{
	return glm::length(p) - 1.8f;
}

static __host__ __device__ float octahedronSDF(glm::vec3 p, float s)
{
	p = glm::abs(p);
	float m = p.x + p.y + p.z - s;
	glm::vec3 q;
	if (3.0f * p.x < m) { q = glm::vec3(p.x, p.y, p.z); }
	else if (3.0f * p.y < m) { q = glm::vec3(p.y, p.z, p.x); }
	else if (3.0f * p.z < m) { q = glm::vec3(p.z, p.x, p.y); }
	else { return m * 0.57735027f; }

	float k = glm::clamp(0.5f * (q.z - q.y + s), 0.0f, s);
	return glm::length(glm::vec3(q.x, q.y - s + k, q.z - k));
}

static __host__ __device__ float hexPrismSDF(glm::vec3 p, glm::vec2 h)
{
	const glm::vec3 k = glm::vec3(-0.8660254f, 0.5f, 0.57735f);
	p = glm::abs(p);

	float kxyDotPxy = k.x * p.x + k.y * p.y;
	float t = 2.0f * glm::min(kxyDotPxy, 0.0f);
	p.x -= t * k.x;
	p.y -= t * k.y;

	glm::vec2 pxy = glm::vec2(p.x, p.y);
	glm::vec2 clampedPoint = glm::vec2(glm::clamp(p.x, -k.z * h.x, k.z * h.x), h.x);
	float dx = glm::length(pxy - clampedPoint) * glm::sign(p.y - h.x);
	float dy = p.z - h.y;
	glm::vec2 d = glm::vec2(dx, dy);

	return glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, glm::vec2(0.0f)));
}

static __host__ __device__ float cutHollowSphereSDF(glm::vec3 p, float r, float h, float t)
{
	float w = sqrt(r * r - h * h);
	glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y);

	return ((h * q.x < w * q.y) ? glm::length(q - glm::vec2(w, h))
	                             : glm::abs(glm::length(q) - r)) - t;
}

static __host__ __device__ float triacontahedronSDF(glm::vec3 p)
{
	float size = 2.0f;
	p /= size;
	float c = cos(PI / 5.);
	float s = sqrt(0.75 - c * c);

	glm::vec3 n = glm::vec3(-0.5, -c, s); 

	p = glm::abs(p);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	p.x = glm::abs(p.x);
	p.y = glm::abs(p.y);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	p.x = glm::abs(p.x);
	p.y = glm::abs(p.y);
	p -= glm::vec3(2.0f * glm::min(glm::dot(n, p), 0.0f)) * n;

	float d = p.z-1.0f;

	return d*size;
}

__host__ __device__ float sceneSDF(glm::vec3 p) {

	p -= glm::vec3(0.0f, 2.f, 0.0f);
	const float scale = 0.6;
	
	//return boxSDF(p, glm::vec3(1.0f, 1.0f, 1.0f)-glm::vec3(0.2)) * scale-0.2;
	float sph = sphereSDF(p - glm::vec3(0.0f)) * scale;
	float cupp = cutHollowSphereSDF(p, 1.8f, 0.5f, 0.2f) * scale;
	float triacon = triacontahedronSDF(p) * scale;
	float oct = octahedronSDF(p, 2.0f) * scale;

	return oct;
	
}

static __host__ __device__ glm::vec3 getSDFNormal(glm::vec3 p)
{
	const float eps = 0.0001f;
	glm::vec3 n;
	n.x = sceneSDF(p + glm::vec3(eps, 0.0f, 0.0f)) - sceneSDF(p - glm::vec3(eps, 0.0f, 0.0f));
	n.y = sceneSDF(p + glm::vec3(0.0f, eps, 0.0f)) - sceneSDF(p - glm::vec3(0.0f, eps, 0.0f));
	n.z = sceneSDF(p + glm::vec3(0.0f, 0.0f, eps)) - sceneSDF(p - glm::vec3(0.0f, 0.0f, eps));
	return glm::normalize(n);
}



__host__ __device__ float sdfIntersectionTest(Ray r, glm::vec3& intersectionPoint, glm::vec3& normal) {
	glm::vec3 rd = glm::normalize(r.direction);

	const int maxSteps = 128;
	const float epsilon = 0.0001f;
	const float maxDist = 100.0f;

	float t = 0.0f;
	for (int i = 0; i < maxSteps; i++)
	{
		glm::vec3 p = r.origin + t * rd;
		float dist = sceneSDF(p);

		if (glm::abs(dist) < epsilon)
		{
			intersectionPoint = p;
			normal = getSDFNormal(p);
			return t;
		}

		t += glm::abs(dist); // glass too

		if (t > maxDist)
		{
			break;
		}
	}

	return -1;
}
