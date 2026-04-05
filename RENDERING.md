# Spark Engine Rendering Technology Stack

This document details the advanced rendering techniques implemented in the Spark Engine.

## 1. Hybrid Rendering Architecture
Spark uses a **Hybrid Rendering Pipeline** that combines the efficiency of rasterization with the accuracy of hardware-accelerated ray tracing.

### Core Stages:
- **Hi-Z Generation**: Builds a hierarchical depth buffer for occlusion culling and SSR.
- **Clustered Shading**: Divides the frustum into 16x9x24 clusters to efficiently process hundreds of local light sources.
- **Occlusion Culling**: Compute-based culling using the Hi-Z pyramid to skip rendering hidden objects.
- **G-Buffer Pass**: Captures Albedo, Normals, PBR properties (Metallic, Roughness), and Velocity.
- **Deferred Lighting**: Performs PBR lighting calculation using the Clustered data and IBL.
- **Hybrid Ray Tracing**: A specialized pass that reconstructs world position from the G-Buffer and traces rays for high-quality Reflections, Shadows, AO, and Global Illumination.

## 2. Physically Based Rendering (PBR)
Spark implements the **Cook-Torrance BRDF** model:
- **D (NDF)**: Trowbridge-Reitz GGX for microfacet distribution.
- **G (Geometry)**: Smith model with Schlick-GGX approximation.
- **F (Fresnel)**: Schlick's approximation for view-dependent reflectivity.

### Image-Based Lighting (IBL):
- **Irradiance Maps**: Prefiltered cubemaps for diffuse ambient lighting.
- **Specular Maps**: Importance-sampled prefiltered environment maps with varying LODs for roughness.
- **BRDF LUT**: A look-up table for integrating the specular split-sum approximation.

## 3. Advanced Effects
### Hardware Ray Tracing (Vulkan RT)
Leverages `VK_KHR_acceleration_structure` and `VK_KHR_ray_tracing_pipeline`.
- **Hybrid Optimization**: Primary rays are skipped (using G-Buffer instead), focusing GPU budget on secondary effects.
- **Dynamic TLAS**: Acceleration structures are rebuilt/updated every frame to support moving objects.

### Post-Processing:
- **Jimenez 2014 Bloom**: High-quality bloom with multiple downsampling/upsampling stages.
- **Temporal Anti-Aliasing (TAA)**: Sub-pixel jittering and temporal accumulation to eliminate aliasing and shimmering.
- **ACES Tonemapping**: Industry-standard filmic tone mapping curve.
- **SSAO & SSGI**: Screen-space techniques for ambient occlusion and global illumination, providing contact shadows and bounced light.

### Environmental Effects:
- **Cascaded Shadow Maps (CSM)**: 4 cascades with stable snapping to texel size for crisp shadows at any distance.
- **Volumetric Fog**: Exponential height fog integrated with shadows for god-ray effects.
- **Depth of Field (DoF)**: Compute-based bokeh blur with depth-aware sampling.

## 4. Performance & Efficiency
- **Vulkan Synchronization2**: Automated barrier injection in the `RenderGraph` minimizes pipeline stalls.
- **Bindless Textures**: Uses `descriptorIndexing` to allow shaders to access any texture without re-binding, reducing CPU overhead.
- **Parallel Dispatch**: Command recording and GPU data preparation are fully parallelized via Rayon.
