use crate::pipeline::Pipeline;
use crate::resource::Attachment;
use crate::MAX_FRAMES_IN_FLIGHT;
use ash::vk;

use super::{RenderContext, RenderPass};
use crate::Renderer;

pub struct PostProcessPipelineParams<'a> {
    pub device: &'a ash::Device,
    pub pipeline_cache: vk::PipelineCache,
    pub extent: vk::Extent2D,
    pub vert_spirv: &'a [u32],
    pub frag_spirv: &'a [u32],
    pub downsample_spirv: &'a [u32],
    pub upsample_spirv: &'a [u32],
}

pub struct PostProcessPass {
    pub pipeline: Option<vk::Pipeline>,
    pub downsample_pipeline: Option<vk::Pipeline>,
    pub upsample_pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,

    // Bloom chain resources
    pub bloom_mips: Vec<Attachment>,
    pub bloom_descriptor_sets: Vec<vk::DescriptorSet>,

    pub swapchain_format: vk::Format,
}

impl RenderPass for PostProcessPass {
    fn name(&self) -> &str {
        "PostProcessPass"
    }

    fn inputs(&self) -> Vec<&'static str> {
        vec![
            "GBufferHDR",
            "GBufferVelocity",
            "VolumetricOutput",
            "SpriteColor",
            "DoFOutput",
            "RTOutput",
        ]
    }

    fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    fn set_descriptor_sets(&mut self, sets: Vec<vk::DescriptorSet>) {
        self.descriptor_sets = sets;
    }

    fn bindings(&self) -> Vec<super::ResourceBinding> {
        vec![
            super::ResourceBinding::SampledImage(0, "GBufferHDR".to_string()),
            super::ResourceBinding::SampledImage(1, "BloomOutput".to_string()),
            super::ResourceBinding::SampledImage(2, "VolumetricOutput".to_string()),
            super::ResourceBinding::SampledImage(3, "SpriteColor".to_string()),
            super::ResourceBinding::SampledImage(4, "GBufferVelocity".to_string()),
            super::ResourceBinding::StorageBuffer(5, "Luminance".to_string()),
            super::ResourceBinding::SampledImage(6, "DoFOutput".to_string()),
            super::ResourceBinding::SampledImage(7, "RTOutput".to_string()),
        ]
    }

    fn dependencies(&self) -> Vec<&'static str> {
        vec![
            "LightingPass",
            "TAAPass",
            "VolumetricPass",
            "SpritePass",
            "RayTracingPass",
        ]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let target_view = renderer.viewport_attachment.as_ref().map(|a| a.view);

        let taa_view = renderer.get_pass_resource_view("TAAPass", "history", 0);
        let input_view = taa_view.unwrap_or(
            renderer
                .get_pass_resource_view("", "GBufferHDR", 0)
                .unwrap_or(renderer.common_shadow_view),
        );

        unsafe {
            // 1. Bloom Downsampling Chain
            let mut current_src_view = input_view;
            let mut current_src_res = vk::Extent2D {
                width: renderer.get_extent().width,
                height: renderer.get_extent().height,
            };

            for i in 0..self.bloom_mips.len() {
                let mip = &self.bloom_mips[i];
                let ds = self.bloom_descriptor_sets[ctx.current_frame * self.bloom_mips.len() + i];

                // Update descriptor with current source
                let img_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(current_src_view)
                    .sampler(renderer.common_sampler)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_info)];
                renderer.device.device.update_descriptor_sets(&write, &[]);

                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(mip.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: mip.extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                renderer
                    .device
                    .device
                    .cmd_begin_rendering(ctx.command_buffer, &rendering_info);
                renderer.device.device.cmd_bind_pipeline(
                    ctx.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.downsample_pipeline.expect("Downsample pipeline missing in PostProcessPass"),
                );
                renderer.device.device.cmd_bind_descriptor_sets(
                    ctx.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[ds],
                    &[],
                );

                let res = [current_src_res.width as f32, current_src_res.height as f32];
                let pc_bytes = std::slice::from_raw_parts(res.as_ptr() as *const u8, 8);
                renderer.device.device.cmd_push_constants(
                    ctx.command_buffer,
                    self.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );

                let viewport = vk::Viewport::default()
                    .width(mip.extent.width as f32)
                    .height(mip.extent.height as f32)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(mip.extent);
                renderer
                    .device
                    .device
                    .cmd_set_viewport(ctx.command_buffer, 0, &[viewport]);
                renderer
                    .device
                    .device
                    .cmd_set_scissor(ctx.command_buffer, 0, &[scissor]);

                renderer
                    .device
                    .device
                    .cmd_draw(ctx.command_buffer, 3, 1, 0, 0);
                renderer.device.device.cmd_end_rendering(ctx.command_buffer);

                // Barrier to read from this mip in next stage
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(mip.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    })
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                renderer.device.device.cmd_pipeline_barrier(
                    ctx.command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );

                current_src_view = mip.view;
                current_src_res = mip.extent;
            }

            // 2. Bloom Upsampling Chain (Additive)
            for i in (0..self.bloom_mips.len() - 1).rev() {
                let dst_mip = &self.bloom_mips[i];
                let src_mip = &self.bloom_mips[i + 1];
                let ds =
                    self.bloom_descriptor_sets[ctx.current_frame * self.bloom_mips.len() + i + 1]; // Reuse DS for upsampling

                let img_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(src_mip.view)
                    .sampler(renderer.common_sampler)];
                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_info)];
                renderer.device.device.update_descriptor_sets(&write, &[]);

                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(dst_mip.view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD) // Additive blending
                    .store_op(vk::AttachmentStoreOp::STORE);

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: dst_mip.extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                renderer
                    .device
                    .device
                    .cmd_begin_rendering(ctx.command_buffer, &rendering_info);
                renderer.device.device.cmd_bind_pipeline(
                    ctx.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.upsample_pipeline.expect("Upsample pipeline missing in PostProcessPass"),
                );
                renderer.device.device.cmd_bind_descriptor_sets(
                    ctx.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[ds],
                    &[],
                );

                let filter_radius = 0.005f32; // Adjustable
                let pc_bytes =
                    std::slice::from_raw_parts(&filter_radius as *const f32 as *const u8, 4);
                renderer.device.device.cmd_push_constants(
                    ctx.command_buffer,
                    self.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );

                let viewport = vk::Viewport::default()
                    .width(dst_mip.extent.width as f32)
                    .height(dst_mip.extent.height as f32)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(dst_mip.extent);
                renderer
                    .device
                    .device
                    .cmd_set_viewport(ctx.command_buffer, 0, &[viewport]);
                renderer
                    .device
                    .device
                    .cmd_set_scissor(ctx.command_buffer, 0, &[scissor]);

                renderer
                    .device
                    .device
                    .cmd_draw(ctx.command_buffer, 3, 1, 0, 0);
                renderer.device.device.cmd_end_rendering(ctx.command_buffer);

                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(dst_mip.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        level_count: 1,
                        layer_count: 1,
                        ..Default::default()
                    })
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);
                renderer.device.device.cmd_pipeline_barrier(
                    ctx.command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            // 3. Final Tonemapping and UI Swapchain Write
            let swapchain_image = renderer.swapchain.images[ctx.image_index as usize];
            let extent = renderer.swapchain.extent;

            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                });
            renderer.device.device.cmd_pipeline_barrier(
                ctx.command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let view = target_view.unwrap_or(renderer.swapchain.views[ctx.image_index as usize]);
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            renderer
                .device
                .device
                .cmd_begin_rendering(ctx.command_buffer, &rendering_info);
            renderer.device.device.cmd_bind_pipeline(
                ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.expect("Final tonemapping pipeline missing in PostProcessPass"),
            );
            renderer.device.device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[
                    renderer.gpu_resource_manager.bindless.set,
                    self.descriptor_sets[0],
                ],
                &[],
            );

            #[repr(C)]
            struct PostProcessPC {
                exposure: f32,
                gamma: f32,
                bloom_enabled: f32,
                vignette_intensity: f32,
                vignette_smoothness: f32,
                chromatic_aberration: f32,
                film_grain: f32,
                motion_blur_strength: f32,
                auto_exposure_enabled: f32,
                dof_enabled: f32,
                lut_index: f32,
                time: f32,
                rt_reflections: f32,
                rt_shadows: f32,
                rt_ao: f32,
                rt_gi: f32,
            }
            let pc = PostProcessPC {
                exposure: renderer.settings.exposure,
                gamma: renderer.settings.gamma,
                bloom_enabled: if renderer.settings.enable_bloom {
                    1.0
                } else {
                    0.0
                },
                vignette_intensity: renderer.settings.vignette_intensity,
                vignette_smoothness: renderer.settings.vignette_smoothness,
                chromatic_aberration: renderer.settings.chromatic_aberration,
                film_grain: renderer.settings.film_grain,
                motion_blur_strength: if renderer.settings.enable_motion_blur {
                    renderer.settings.motion_blur_strength
                } else {
                    0.0
                },
                auto_exposure_enabled: if renderer.settings.enable_auto_exposure {
                    1.0
                } else {
                    0.0
                },
                dof_enabled: if renderer.settings.enable_dof {
                    1.0
                } else {
                    0.0
                },
                lut_index: if renderer.settings.enable_color_grading {
                    renderer.settings.lut_index as f32
                } else {
                    -1.0
                },
                time: (renderer.frame_manager.frame_index as f32) * 0.016,
                rt_reflections: if renderer.settings.enable_rt_reflections {
                    1.0
                } else {
                    0.0
                },
                rt_shadows: if renderer.settings.enable_rt_shadows {
                    1.0
                } else {
                    0.0
                },
                rt_ao: if renderer.settings.enable_rt_ao {
                    1.0
                } else {
                    0.0
                },
                rt_gi: if renderer.settings.enable_rt_gi {
                    1.0
                } else {
                    0.0
                },
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<PostProcessPC>(),
            );
            renderer.device.device.cmd_push_constants(
                ctx.command_buffer,
                self.layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_bytes,
            );

            let viewport = vk::Viewport::default()
                .width(extent.width as f32)
                .height(extent.height as f32)
                .max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(extent);
            renderer
                .device
                .device
                .cmd_set_viewport(ctx.command_buffer, 0, &[viewport]);
            renderer
                .device
                .device
                .cmd_set_scissor(ctx.command_buffer, 0, &[scissor]);

            renderer
                .device
                .device
                .cmd_draw(ctx.command_buffer, 3, 1, 0, 0);
            renderer.device.device.cmd_end_rendering(ctx.command_buffer);
        }
    }

    fn on_resize(&mut self, renderer: &Renderer, new_extent: vk::Extent2D) {
        let device = &renderer.device;
        for mip in self.bloom_mips.drain(..) {
            mip.destroy(&device.device, &device.allocator);
        }

        let num_bloom_mips = 6;
        for i in 1..=num_bloom_mips {
            let att = Attachment::create_image_resource(
                device,
                (new_extent.width >> i).max(1),
                (new_extent.height >> i).max(1),
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )
            .expect("Failed to create bloom mip attachment during resize");
            self.bloom_mips.push(att);
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            let device = &renderer.device.device;
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.downsample_pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.upsample_pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for mip in self.bloom_mips.drain(..) {
                mip.destroy(device, &renderer.device.allocator);
            }
        }
    }
}

impl PostProcessPass {
    pub fn new(
        renderer: &crate::Renderer,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let ds_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[renderer.gpu_resource_manager.bindless.layout, ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                        offset: 0,
                        size: 64, // Increased size for new PC
                    }]),
                None,
            )?
        };

        let num_bloom_mips = 6;
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(
                    (MAX_FRAMES_IN_FLIGHT as u32 * 6)
                        + (num_bloom_mips as u32 * MAX_FRAMES_IN_FLIGHT as u32),
                ),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(
                        MAX_FRAMES_IN_FLIGHT as u32
                            + (num_bloom_mips as u32 * MAX_FRAMES_IN_FLIGHT as u32),
                    ),
                None,
            )?
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let mut bloom_layouts = Vec::new();
        for _ in 0..(6 * MAX_FRAMES_IN_FLIGHT) {
            bloom_layouts.push(ds_layout);
        }
        let bloom_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&bloom_layouts),
            )?
        };

        let mut bloom_mips = Vec::new();
        for i in 1..=num_bloom_mips {
            let att = Attachment::create_image_resource(
                &renderer.device,
                (extent.width >> i).max(1),
                (extent.height >> i).max(1),
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )?;
            bloom_mips.push(att);
        }

        Ok(Self {
            pipeline: None,
            downsample_pipeline: None,
            upsample_pipeline: None,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_pool,
            descriptor_sets,
            bloom_mips,
            bloom_descriptor_sets,
            swapchain_format: format,
        })
    }

    pub fn create_pipelines(&mut self, params: PostProcessPipelineParams) {
        let device = params.device;
        let extent = params.extent;
        let pipeline_cache = params.pipeline_cache;

        let vert_module = Pipeline::create_shader_module(device, params.vert_spirv);
        let frag_module = Pipeline::create_shader_module(device, params.frag_spirv);
        let downsample_module = Pipeline::create_shader_module(device, params.downsample_spirv);
        let upsample_module = Pipeline::create_shader_module(device, params.upsample_spirv);
        let entry_point = std::ffi::CString::new("main").expect("Failed to create entry point name");

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let color_formats = [self.swapchain_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        self.pipeline = Some(unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[info], None)
                .expect("Failed to create final PostProcess pipeline")[0]
        });

        // Downsample Pipeline
        let ds_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(downsample_module)
                .name(&entry_point),
        ];
        let bloom_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let mut bloom_rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&bloom_formats);

        let ds_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&ds_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.layout)
            .push_next(&mut bloom_rendering);
        self.downsample_pipeline = Some(unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[ds_info], None)
                .expect("Failed to create bloom downsample pipeline")[0]
        });

        // Upsample Pipeline (Additive)
        let us_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(upsample_module)
                .name(&entry_point),
        ];
        let us_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE)
            .alpha_blend_op(vk::BlendOp::ADD);
        let us_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&us_blend_attachment));

        let us_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&us_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&us_blend)
            .layout(self.layout)
            .push_next(&mut bloom_rendering);
        self.upsample_pipeline = Some(unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[us_info], None)
                .expect("Failed to create bloom upsample pipeline")[0]
        });

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
            device.destroy_shader_module(downsample_module, None);
            device.destroy_shader_module(upsample_module, None);
        }
    }
}
