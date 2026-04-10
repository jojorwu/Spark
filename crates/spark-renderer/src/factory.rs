use crate::error::RendererError;
use crate::Renderer;
use ash::vk;

pub struct PassShaders {
    pub culling: Vec<u32>,
    pub hiz: Vec<u32>,
    pub cluster_build: Vec<u32>,
    pub cluster_cull: Vec<u32>,
    pub shadow_vert: Vec<u32>,
    pub shadow_frag: Vec<u32>,
    pub gbuffer_vert: Vec<u32>,
    pub gbuffer_frag: Vec<u32>,
    pub ssao_vert: Vec<u32>,
    pub ssao_frag: Vec<u32>,
    pub ssao_blur_frag: Vec<u32>,
    pub deferred_vert: Vec<u32>,
    pub deferred_frag: Vec<u32>,
    pub grid_vert: Vec<u32>,
    pub grid_frag: Vec<u32>,
    pub volumetric: Vec<u32>,
    pub taa_vert: Vec<u32>,
    pub taa_frag: Vec<u32>,
    pub fullscreen_vert: Vec<u32>,
    pub tonemap_frag: Vec<u32>,
    pub bloom_filter: Vec<u32>,
    pub bloom_downsample: Vec<u32>,
    pub bloom_upsample: Vec<u32>,
    pub forward_vert: Vec<u32>,
    pub forward_frag: Vec<u32>,
    pub particle_comp: Vec<u32>,
    pub particle_vert: Vec<u32>,
    pub particle_frag: Vec<u32>,
    pub ssr_comp: Vec<u32>,
    pub sprite_vert: Vec<u32>,
    pub sprite_frag: Vec<u32>,
    pub luminance: Vec<u32>,
    pub dof: Vec<u32>,
    pub skinning: Vec<u32>,
    pub point_shadow_vert: Vec<u32>,
    pub point_shadow_frag: Vec<u32>,
    pub ssgi: Vec<u32>,
    pub rgen: Vec<u32>,
    pub rmiss: Vec<u32>,
    pub rchit: Vec<u32>,
}

impl Renderer {
    /// Orchestrates the initialization and registration of all default rendering passes.
    pub fn setup_default_passes(&mut self, shaders: PassShaders) -> Result<(), RendererError> {
        let extent = self.get_extent();
        let cache = self.pipeline_cache;
        let pool = self.gpu_resource_manager.descriptor_pool;
        let layout = self.global_descriptor_set_layout;
        let msaa = self.get_msaa_samples();
        let bindless_layout = self.gpu_resource_manager.bindless.layout;

        // 1. Initialize Base Infrastructure
        let hiz_pass = self.init_hiz_pass(&shaders)?;
        let clustered_pass = self.init_clustered_pass(&shaders)?;
        let culling_pass = crate::passes::culling::CullingPass::new(
            &self.device.device,
            pool,
            &shaders.culling,
            layout,
        )?;
        let shadow_pass = self.init_shadow_pass(&shaders, cache)?;
        let gbuffer_pass = crate::passes::gbuffer::GBufferPass::new();
        let as_build_pass = crate::passes::as_build::AccelerationStructurePass::new();

        // 2. Initialize Lighting and Global Effects
        let ssao_pass = self.init_ssao_pass(&shaders, cache, extent, pool)?;
        let lighting_pass =
            self.init_lighting_pass(&shaders, cache, extent, pool, layout, bindless_layout, msaa)?;
        let grid_pass = crate::passes::grid::GridPass::new(
            &self.device.device,
            cache,
            &shaders.grid_vert,
            &shaders.grid_frag,
            layout,
            vk::Format::R16G16B16A16_SFLOAT,
        )?;
        let volumetric_pass =
            crate::passes::volumetric::VolumetricPass::new(self, &shaders.volumetric)?;
        let ssr_pass = crate::passes::ssr::SSRPass::new(self, &shaders.ssr_comp)?;
        let luminance_pass =
            crate::passes::luminance::LuminancePass::new(self, &shaders.luminance)?;
        let dof_pass = crate::passes::dof::DoFPass::new(self, &shaders.dof)?;
        let point_shadow_pass = crate::passes::point_shadow::PointShadowPass::new(
            self,
            &shaders.point_shadow_vert,
            &shaders.point_shadow_frag,
        )?;
        let ssgi_pass = crate::passes::ssgi::SSGIPass::new(self, &shaders.ssgi)?;
        let rt_pass = crate::passes::rt::RayTracingPass::new(
            self,
            &shaders.rgen,
            &shaders.rmiss,
            &shaders.rchit,
        )?;
        let forward_pass = crate::passes::forward::ForwardPass::new(
            self,
            &shaders.forward_vert,
            &shaders.forward_frag,
        )?;
        let particle_pass = crate::passes::particle::ParticlePass::new(
            self,
            &shaders.particle_comp,
            &shaders.particle_vert,
            &shaders.particle_frag,
        )?;
        let sprite_pass = crate::passes::sprite::SpritePass::new(
            self,
            &shaders.sprite_vert,
            &shaders.sprite_frag,
        )?;

        // 3. Initialize Post-Processing
        let taa_pass = crate::passes::taa::TAAPass::new(self, &shaders.taa_frag, &shaders.taa_vert)
            .map_err(|_| RendererError::NoSuitableDevice)?;
        let post_process_pass = self.init_post_process_pass(&shaders, cache, extent)?;

        // 4. Register all passes to the RenderGraph
        self.add_render_pass(hiz_pass, &[], &["HiZ"]);
        self.add_render_pass(clustered_pass, &[], &["ClusteredData"]);
        self.add_render_pass(culling_pass, &["HiZ"], &["CullingData"]);
        self.add_render_pass(shadow_pass, &[], &["ShadowMap"]);
        self.add_render_pass(gbuffer_pass, &[], &["GBuffer"]);
        self.add_render_pass(ssao_pass, &["GBuffer"], &["SSAO"]);
        self.add_render_pass(
            lighting_pass,
            &["GBuffer", "ShadowMap", "ClusteredData", "SSAO"],
            &["HDRColor"],
        );
        self.add_render_pass(grid_pass, &["GBuffer"], &["GridColor"]);
        self.add_render_pass(
            volumetric_pass,
            &["ShadowMap", "ClusteredData"],
            &["VolumetricColor"],
        );
        self.add_render_pass(forward_pass, &["GBuffer"], &["ForwardColor"]);
        self.add_render_pass(particle_pass, &["GBuffer"], &["ParticleColor"]);
        self.add_render_pass(
            ssr_pass,
            &[
                "GBufferAlbedo",
                "GBufferNormal",
                "GBufferDepth",
                "GBufferPBR",
                "HDRColor",
                "HiZ",
            ],
            &["SSR"],
        );
        self.add_render_pass(luminance_pass, &["HDRColor"], &["Luminance"]);
        self.add_render_pass(taa_pass, &["HDRColor", "GBufferDepth"], &["TAAColor"]);
        self.add_render_pass(dof_pass, &["HDRColor", "GBufferDepth"], &["DoF"]);
        self.add_render_pass(point_shadow_pass, &[], &["PointShadowMap"]);
        self.add_render_pass(
            ssgi_pass,
            &["GBufferAlbedo", "GBufferNormal", "GBufferDepth", "HDRColor"],
            &["SSGI"],
        );
        self.add_render_pass(as_build_pass, &[], &["SceneTLAS"]);
        self.add_render_pass(
            rt_pass,
            &[
                "GBufferDepth",
                "GBufferNormal",
                "GBufferPBR",
                "HiZ",
                "SceneTLAS",
            ],
            &["RTOutput"],
        );
        self.add_render_pass(sprite_pass, &["GBufferDepth"], &["SpriteColor"]);
        self.add_render_pass(
            post_process_pass,
            &[
                "TAAColor",
                "SpriteColor",
                "Luminance",
                "DoF",
                "SSGI",
                "RTOutput",
                "GBufferHDR",
                "GBufferVelocity",
            ],
            &["FinalColor"],
        );

        self.compile_render_graph();
        self.update_all_descriptor_sets();

        // Initialize default pipeline for the renderer
        self.init_main_pipeline(
            extent,
            msaa,
            layout,
            bindless_layout,
            cache,
            &shaders.gbuffer_vert,
            &shaders.gbuffer_frag,
        );

        Ok(())
    }

    fn init_hiz_pass(
        &mut self,
        shaders: &PassShaders,
    ) -> Result<crate::passes::hiz::HiZPass, RendererError> {
        let extent = self.get_extent();
        let pass = crate::passes::hiz::HiZPass::new(
            &self.device,
            self.gpu_resource_manager.descriptor_pool,
            &shaders.hiz,
            extent.width,
            extent.height,
        )?;
        self.set_hiz_view(pass.pyramid_view);
        Ok(pass)
    }

    fn init_clustered_pass(
        &self,
        shaders: &PassShaders,
    ) -> Result<crate::passes::clustered::ClusteredPass, RendererError> {
        crate::passes::clustered::ClusteredPass::new(
            self,
            &shaders.cluster_build,
            &shaders.cluster_cull,
        )
        .map_err(|_| RendererError::NoSuitableDevice)
    }

    fn init_shadow_pass(
        &mut self,
        shaders: &PassShaders,
        cache: vk::PipelineCache,
    ) -> Result<crate::passes::shadow::ShadowPass, RendererError> {
        let mut pass = crate::passes::shadow::ShadowPass::new(&self.device)
            .map_err(|_| RendererError::NoSuitableDevice)?;
        pass.create_pipeline(
            &self.device.device,
            cache,
            &shaders.shadow_vert,
            &shaders.shadow_frag,
        );
        self.set_common_shadow_view(pass.view);
        Ok(pass)
    }

    fn init_ssao_pass(
        &mut self,
        shaders: &PassShaders,
        cache: vk::PipelineCache,
        extent: vk::Extent2D,
        pool: vk::DescriptorPool,
    ) -> Result<crate::passes::ssao::SSAOPass, RendererError> {
        let mut pass = crate::passes::ssao::SSAOPass::new(self, pool)
            .map_err(|_| RendererError::NoSuitableDevice)?;
        pass.create_pipelines(crate::passes::ssao::SSAOPipelineParams {
            device: &self.device.device,
            pipeline_cache: cache,
            extent,
            vert_shader: &shaders.ssao_vert,
            ssao_shader: &shaders.ssao_frag,
            blur_shader: &shaders.ssao_blur_frag,
        });
        Ok(pass)
    }

    #[allow(clippy::too_many_arguments)]
    fn init_lighting_pass(
        &self,
        shaders: &PassShaders,
        cache: vk::PipelineCache,
        extent: vk::Extent2D,
        pool: vk::DescriptorPool,
        layout: vk::DescriptorSetLayout,
        bindless_layout: vk::DescriptorSetLayout,
        msaa: vk::SampleCountFlags,
    ) -> Result<crate::passes::lighting::LightingPass, RendererError> {
        let mut pass =
            crate::passes::lighting::LightingPass::new(&self.device.device, pool, layout)
                .map_err(|_| RendererError::NoSuitableDevice)?;
        pass.create_pipeline(crate::passes::lighting::LightingPipelineParams {
            device: &self.device.device,
            pipeline_cache: cache,
            extent,
            vert_spirv: &shaders.deferred_vert,
            frag_spirv: &shaders.deferred_frag,
            msaa_samples: msaa,
            global_ds_layout: layout,
            bindless_ds_layout: bindless_layout,
        });
        Ok(pass)
    }

    fn init_post_process_pass(
        &self,
        shaders: &PassShaders,
        cache: vk::PipelineCache,
        extent: vk::Extent2D,
    ) -> Result<crate::passes::post_process::PostProcessPass, RendererError> {
        let mut pass =
            crate::passes::post_process::PostProcessPass::new(self, self.swapchain.format, extent)
                .map_err(|_| RendererError::NoSuitableDevice)?;
        pass.create_pipelines(
            crate::passes::post_process::PostProcessPipelineParams {
                device: &self.device.device,
                pipeline_cache: cache,
                extent,
                vert_spirv: &shaders.fullscreen_vert,
                frag_spirv: &shaders.tonemap_frag,
                downsample_spirv: &shaders.bloom_downsample,
                upsample_spirv: &shaders.bloom_upsample,
            },
            &shaders.bloom_filter,
        );
        Ok(pass)
    }

    #[allow(clippy::too_many_arguments)]
    fn init_main_pipeline(
        &mut self,
        extent: vk::Extent2D,
        msaa: vk::SampleCountFlags,
        layout: vk::DescriptorSetLayout,
        bindless_layout: vk::DescriptorSetLayout,
        cache: vk::PipelineCache,
        vert: &[u32],
        frag: &[u32],
    ) {
        let pipeline = crate::pipeline::Pipeline::new(
            self.get_device(),
            &crate::pipeline::PipelineCreateParams {
                extent,
                vert_shader_code: vert,
                frag_shader_code: frag,
                msaa_samples: msaa,
                is_deferred_lighting: false,
                input_attachments_count: 0,
                pipeline_cache: cache,
                global_ds_layout: layout,
                bindless_ds_layout: bindless_layout,
            },
        );
        self.set_pipeline(pipeline);
    }
}
