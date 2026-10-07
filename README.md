**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* (TODO) YOUR NAME HERE
* Tested on: (TODO) Windows 22, i7-2222 @ 2.22GHz 22GB, GTX 222 222MB (Moore 2222 Lab)

CUDA Path Tracer
================

![](renders/cornell.2026-10-04_03-47-11z.2032samp.png)

### With caustics vs. without caustics

With caustics vs. Without Caustics 

![alt text](renders/overRing.png)
![alt text](renders/overRing_noPhotons.png)


### More Renders

<!-- TODO: add renders -->

## The Caustics Problem in Path Tracing

A caustic is a bright pattern formed when light is refracted or reflected by a specular surface, such as glass, and gets concentrated onto a diffuse surface.

Caustics are difficult to render with a traditional path tracer that starts from the camera. The ray has to bounce off a diffuse surface, find the narrow path through a dielectric, and then land on the light. Sharp, well-defined caustic patterns need a small light source. But because glass is a delta BSDF and scatters light in only one direction, the ray has a very small chance of hitting the light's small solid angle.

To solve this, I implemented photon splatting, a variation of photon mapping. Instead of tracing from the camera towards the light, it traces light paths from the light source into the scene and splats the energy back onto the camera's image.

## Table of Contents

0. [Dielectrics](#0-dielectrics)
1. [MIS / NEE](#1-mis--nee)
2. [Caustics & Photon Splatting](#2-caustics--photon-splatting)
    - [Photon Pass](#photon-pass)
    - [Removing Double Counting](#removing-double-counting)
    - [Normalization](#normalization)
    - [Depth Test](#depth-test)
    - [Dispersion](#dispersion)
3. [Procedural SDF Shapes for Glass](#3-procedural-sdf-shapes-for-glass)
4. [Performance](#4-performance)
5. [Bloopers](#5-bloopers)

## Photon Splatting Path Tracer Pipeline

### 0. Dielectrics

In a dielectric material, light either reflects or refracts based on the Fresnel reflectance. In my BSDF implementation (based on PBRT), I compute the dielectric Fresnel term and use it to randomly choose between reflection and refraction. At grazing angles Fresnel is high, so reflection is more likely. Otherwise, the ray refracts using Snell's law with the material's index of refraction. Total internal reflection always reflects the ray back inside the glass, which is important for the complex light paths that create caustics.

![](renders/naive.png)

### 1. MIS / NEE

The basic path tracer shoots rays from the camera and samples new directions from the surface BSDF. If the path never hits a light, it contributes nothing to the final image.

| Naive with large light, 5000 samples | Naive with small light, 5000 samples |
|:---:|:---:|
| ![](renders/naive.png) | ![](renders/0naive_smallL.png) |

As seen above, this works fine for large lights. But to render sharp, defined caustics through glass, I need a small light, and the naive path tracer is bad at finding small lights.

To fix this, I implemented NEE (next event estimation) with MIS (multiple importance sampling). At every diffuse hit, I sample both a point on the light and a BSDF direction, and combine them with the power heuristic. This way, light is found directly through light sampling, while BSDF sampling still covers the cases where light sampling is weak.

| Naive, 5000 samples | MIS / NEE, 500 samples | MIS / NEE, 5000 samples |
|:---:|:---:|:---:|
| ![](renders/0naive_smallL.png) | ![](renders/1mis500.png) | ![](renders/1mis5000.png) |

The image is now much less noisy, especially on diffuse surfaces and in glass, and it converges much faster. However, the caustics are still barely visible, look discontinuous, and have fireflies.

Looking at the path a caustic needs again, from the camera's side:

Diffuse -> specular bounces through glass -> Light

On diffuse surfaces, NEE treats the glass as an occluder, because a straight shadow ray to the light can't pass through a refractive surface. And during specular bounces the BSDF is a delta (pdf = 1), so there is no light sampling to do. The only way to get the caustic is for the glass BSDF sample to exit and hit the light by chance. With a small light that is very unlikely, and when it does happen the sample is very bright, creating fireflies.

### 2. Caustics & Photon Splatting

To solve this, I added a separate photon pass that traces light paths from the light source into the scene. I used photon splatting, a variant of photon mapping where each photon is traced independently and splatted directly onto the screen. Unlike traditional photon mapping, it doesn't need a spatial acceleration structure (like a kd-tree), which makes it easy to parallelize on the GPU. The downside is that photons have to be shot again every frame, but this fits well with how the path tracer already accumulates samples.

**Camera pass + Photon pass = Final image** below. 

| Camera pass | Photon pass | Final |
|:---:|:---:|:---:|
| ![](renders/pass0.png) | ![](renders/pass1.png) | ![](renders/combined.png) |

#### Photon Pass

My photon pass works as follows:
My implementation of Photon Pass as follows:
- Choose a point on a light and shoot a photon in a cosine-weighted direction from it. Total flux is Phi = PI * A * Le, and each photon carries Phi / N.
- If it hits a diffuse surface on the first bounce: discarded since it is direct lighting. 
- If it hits a specular surface: keep bouncing, using the same BSDF logic as the camera pass.
- If it hits a diffuse surface after a specular surface: project photon at that location onto the camera image. 

Photon visualization

| 500 samples | 5000 samples |
|:---:|:---:|
| ![](renders/vis00.png) | ![](renders/vis11.png) |

This shows where the splatted photons land: on diffuse surfaces, after going through a dielectric.

Each photon contributes its power times the diffuse BSDF, photon power * (albedo / PI):

![](renders/00.png)

#### Removing Double Counting

It looks pretty good! But there are still fireflies. This is because the caustic path is counted twice: once by the photons, and once by camera paths that find the light through glass by luck. So in the camera pass, I don't count the camera -> diffuse -> dielectric -> light path. 
And we can get rid of fireflies: 

![](renders/01.png)

#### Normalization

Photons are currently splatted onto a pixel without considering the area of the surface that pixel covers. This gives inaccurate radiance, and a photon spreading over a large area (at grazing angle) looks as bright as one concentrated on a small area. 
We need to project the pixel to the surface to compute this footprint area, then divide the photon flux by it to get the correct radiance.

#### Depth Test

Photons are projected onto the screen without knowing what the camera actually sees there, so photons landing on hidden surfaces leak through objects in front of them. I store the camera's first-hit depth for each pixel during the camera pass, and a photon is only splatted if it isn't behind that surface.

| Before normalization and depth test | After |
|:---:|:---:|
| ![](renders/01.png) | ![](renders/comparison.png) |

#### Dispersion

Different colors refract at different angles in glass, causing dispersion. I approximate this without full spectral rendering by randomly selecting one RGB channel's IOR at the first specular bounce. Accumulated over samples, this creates a rainbow X) 🌈 

![](renders/combined.png)

### 3. Procedural SDF Shapes for Glass

I used signed distance functions (SDFs) to create procedural glass objects in the scene

inner sdfs must also be accurate (or underestimated). I refered to this iq's article regarding this.

- smooth minimum of many donut shapes

- using 3D voronoi noise to add roughness/texture to objects. More faces in glass causes more complex refractions.

## 4. Performance

Photon splatting pass on vs. off, rendered for the same amount of time (5s, 30s, 2min).

#### Cornell box

| | 5s | 30s | 120s |
|:---:|:---:|:---:|:---:|
| **No photon pass**<br>71.9 fps | ![](renders/Performance/cornellGlass00_21-05_5s.png) | ![](renders/Performance/cornellGlass00_21-05_30s.png) | ![](renders/Performance/cornellGlass00_21-05_120s.png) |
| **Photon pass**<br>58.7 fps | ![](renders/Performance/cornellGlass00_21-02_5s.png) | ![](renders/Performance/cornellGlass00_21-02_30s.png) | ![](renders/Performance/cornellGlass00_21-02_120s.png) |

#### Glass spherical Object

| | 5s | 30s | 120s |
|:---:|:---:|:---:|:---:|
| **No photon pass**<br>25.5 fps | ![](renders/Performance/thickRings7_21-21_5s.png) | ![](renders/Performance/thickRings7_21-21_30s.png) | ![](renders/Performance/thickRings7_21-21_120s.png) |
| **Photon pass**<br>32.4 fps | ![](renders/Performance/thickRings7_21-18_5s.png) | ![](renders/Performance/thickRings7_21-18_30s.png) | ![](renders/Performance/thickRings7_21-18_120s.png) |

#### Three Glass Objects

| | 5s | 30s | 120s |
|:---:|:---:|:---:|:---:|
| **No photon pass**<br>35.0 fps | ![](renders/Performance/ACuteBall6_21-54_5s.png) | ![](renders/Performance/ACuteBall6_21-54_30s.png) | ![](renders/Performance/ACuteBall6_21-54_120s.png) |
| **Photon pass**<br>46.9 fps | ![](renders/Performance/ACuteBall6_21-50_5s.png) | ![](renders/Performance/ACuteBall6_21-50_30s.png) | ![](renders/Performance/ACuteBall6_21-50_120s.png) |

Without photons, the caustics in the spherical object scene and 3 glasses scenes are still grainy even after 2 minutes, while with photons they are already smooth. In the Cornell box, the large light lets the camera pass find caustics on its own, so the difference is smaller.

## 5. Bloopers

