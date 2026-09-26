#pragma once

#include <glm/glm.hpp>
#include "scene.h"

struct Photon
{
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 power;
};

void photonMap(Scene* scene, int numPhotons, int iter);


void generatePhotonDirections(
    int numPhotons,
    int iter,
    glm::vec3 lightPos,
    glm::vec3 lightColor,
    float totalFlux,
    Photon* dev_photons);
