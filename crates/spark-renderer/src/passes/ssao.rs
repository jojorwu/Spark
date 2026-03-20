use ash::vk;
use crate::resource::{Attachment, Buffer};
use crate::Renderer;
use crate::MAX_FRAMES_IN_FLIGHT;
use spark_math::{Vec3, Vec4, Mat4};
use rand::Rng;

pub struct SSAORecordParams<'a> {
    pub renderer: &'a Renderer,
    pub command_buffer: vk::CommandBuffer,
    pub extent: vk::Extent2D,
    pub current_frame: usize,
    pub ssao_target_view: vk::ImageView,
    pub ssao_target_image: vk::Image,
    pub blur_target_view: vk::ImageView,
    pub blur_target_image: vk::Image,
    pub projection: Mat4,
    pub view: Mat4,
}

pub struct SSAOPipelineParams<'a> {
    pub device: &'a ash::Device,
    pub pipeline_cache: vk::PipelineCache,
    pub extent: vk::Extent2D,
    pub vert_shader: &'a [u32],
    pub ssao_shader: &'a [u32],
    pub blur_shader: &'a [u32],
}

pub struct SSAOPass {
    pub ssao_pipeline: vk::Pipeline,
    pub blur_pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub blur_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub blur_descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub blur_descriptor_sets: Vec<vk::DescriptorSet>,
    pub noise_texture: crate::vulkan::texture::Texture,
    pub ssao_params_buffer: Vec<Buffer>,
    pub kernel_samples: [Vec4; 64],
    pub ssao_images: Vec<Attachment>,
    pub ssao_blur_images: Vec<Attachment>,
}

use super::{RenderPass, RenderContext};

impl RenderPass for SSAOPass {
    fn name(&self) -> &str { "SSAOPass" }
    fn is_enabled(&self, renderer: &Renderer) -> bool { renderer.enable_ssao }
    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let extent = renderer.get_extent();
        let view = renderer.scene_view_matrix_for_pos;
        let projection = spark_math::Mat4::perspective_rh(
            45.0f32.to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
            100.0,
        );
        let params = SSAOParamsStruct {
            samples: self.kernel_samples,
            projection,
            view,
            screen_size: [extent.width as f32, extent.height as f32],
        };
        renderer.upload_to_buffer(&self.ssao_params_buffer[current_frame], &[params]);
    }

    fn update_descriptor_sets(&self, renderer: &Renderer) {
        self.update_descriptor_sets_impl(
            &renderer.device.device,
            &renderer.gbuffer.normal,
            &renderer.gbuffer.depth,
            &self.ssao_images,
            renderer.common_sampler,
        );
    }

    fn needs_descriptor_update(&self, _renderer: &Renderer, _frame_index: usize) -> bool {
        // Example: logic to check if versions changed
        // In a full implementation, we'd compare renderer.gbuffer.normal[frame_index].version
        // against a version stored in the pass.
        true
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let extent = renderer.swapchain.extent;
        let view = renderer.scene_view_matrix_for_pos;
        let projection = spark_math::Mat4::perspective_rh(
            45.0f32.to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
            100.0,
        );
        let params = SSAORecordParams {
            renderer,
            command_buffer: ctx.command_buffer,
            extent,
            current_frame: ctx.current_frame,
            ssao_target_view: self.ssao_images[ctx.current_frame].view,
            ssao_target_image: self.ssao_images[ctx.current_frame].image,
            blur_target_view: self.ssao_blur_images[ctx.current_frame].view,
            blur_target_image: self.ssao_blur_images[ctx.current_frame].image,
            projection,
            view,
        };
        self.record_commands_impl(&params);
    }

    fn get_resource_view(&self, name: &str, frame_index: usize) -> Option<vk::ImageView> {
        if name == "ssao" {
            Some(self.ssao_blur_images[frame_index].view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.ssao_pipeline, None);
            device.destroy_pipeline(self.blur_pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_pipeline_layout(self.blur_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_set_layout(self.blur_descriptor_set_layout, None);
            for buffer in self.ssao_params_buffer.drain(..) {
                renderer.destroy_buffer(buffer);
            }
            for img in self.ssao_images.drain(..) {
                img.destroy(device, &renderer.device.allocator);
            }
            for img in self.ssao_blur_images.drain(..) {
                img.destroy(device, &renderer.device.allocator);
            }
            renderer.device.device.destroy_sampler(self.noise_texture.sampler, None);
            renderer.device.device.destroy_image_view(self.noise_texture.view, None);
            renderer.device.device.destroy_image(self.noise_texture.image, None);
            if let Some(alloc) = self.noise_texture.allocation.take() {
                renderer.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }
}

impl SSAOPass {
    pub fn new(
        renderer: &Renderer,
        descriptor_pool: vk::DescriptorPool,
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;

        // 1. Generate Kernel Samples
        let mut rng = rand::thread_rng();
        let mut kernel_samples = [Vec4::ZERO; 64];
        for (i, sample_vec) in kernel_samples.iter_mut().enumerate() {
            let mut sample = Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(0.0..1.0),
            ).normalize();
            sample *= rng.gen_range(0.0..1.0);

            let mut scale = i as f32 / 64.0;
            scale = 0.1 + scale * scale * (1.0 - 0.1); // Lerp
            *sample_vec = Vec4::new(sample.x * scale, sample.y * scale, sample.z * scale, 0.0);
        }

        // 2. Generate Noise Texture
        let mut ssao_noise = Vec::new();
        for _ in 0..16 {
            ssao_noise.push(rng.gen_range(-1.0..1.0));
            ssao_noise.push(rng.gen_range(-1.0..1.0));
            ssao_noise.push(0.0f32);
            ssao_noise.push(1.0f32);
        }

        let noise_img = image::DynamicImage::ImageRgba32F(
            image::Rgba32FImage::from_raw(4, 4, bytemuck::cast_slice(&ssao_noise).to_vec()).unwrap()
        );
        let noise_texture = renderer.create_texture_from_image(&noise_img);

        // 3. Create Buffers
        let mut ssao_params_buffer = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = renderer.create_buffer(
                (64 * 16 + 64 * 2 + 8) as u64, // samples + proj + view + screen
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            ssao_params_buffer.push(buffer);
        }

        // 4. Descriptor Set Layouts
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };

        let blur_bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];
        let blur_ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&blur_bindings), None)? };

        // 5. Pipelines
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[ds_layout]), None)? };
        let blur_layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[blur_ds_layout]), None)? };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let blur_descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[blur_ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let mut ssao_params_buffer = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let buffer = renderer.create_buffer(
                std::mem::size_of::<SSAOParamsStruct>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            ssao_params_buffer.push(buffer);
        }

        let mut ssao_images = Vec::new();
        let mut ssao_blur_images = Vec::new();
        let extent = renderer.get_extent();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
             ssao_images.push(Attachment::create_image_resource(
                &renderer.device, extent.width, extent.height,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )?);
            ssao_blur_images.push(Attachment::create_image_resource(
                &renderer.device, extent.width, extent.height,
                vk::Format::R8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )?);
        }

        Ok(Self {
            ssao_pipeline: vk::Pipeline::null(),
            blur_pipeline: vk::Pipeline::null(),
            layout,
            blur_layout,
            descriptor_set_layout: ds_layout,
            blur_descriptor_set_layout: blur_ds_layout,
            descriptor_sets,
            blur_descriptor_sets,
            noise_texture,
            ssao_params_buffer,
            kernel_samples,
            ssao_images,
            ssao_blur_images,
        })
    }

    pub fn create_pipelines(
        &mut self,
        params: SSAOPipelineParams,
    ) {
        let device = params.device;
        let extent = params.extent;
        let pipeline_cache = params.pipeline_cache;

        let vert_module = crate::pipeline::Pipeline::create_shader_module(device, params.vert_shader);
        let ssao_module = crate::pipeline::Pipeline::create_shader_module(device, params.ssao_shader);
        let blur_module = crate::pipeline::Pipeline::create_shader_module(device, params.blur_shader);

        let entry_point = std::ffi::CString::new("main").unwrap();

        // SSAO Pipeline
        let ssao_stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(ssao_module).name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::Viewport::default().width(extent.width as f32).height(extent.height as f32).max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::BACK).front_face(vk::FrontFace::CLOCKWISE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let color_formats = [vk::Format::R8_UNORM];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let ssao_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&ssao_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        self.ssao_pipeline = unsafe { device.create_graphics_pipelines(pipeline_cache, &[ssao_info], None).unwrap()[0] };

        // Blur Pipeline
        let blur_stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(blur_module).name(&entry_point),
        ];

        let blur_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&blur_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.blur_layout)
            .push_next(&mut rendering_info);

        self.blur_pipeline = unsafe { device.create_graphics_pipelines(pipeline_cache, &[blur_info], None).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(ssao_module, None);
            device.destroy_shader_module(blur_module, None);
        }
    }

    pub fn record_commands_impl(
        &self,
        params: &SSAORecordParams,
    ) {
        let renderer = params.renderer;
        let command_buffer = params.command_buffer;
        let extent = params.extent;
        let current_frame = params.current_frame;
        let ssao_target_view = params.ssao_target_view;
        let ssao_target_image = params.ssao_target_image;
        let blur_target_view = params.blur_target_view;
        let blur_target_image = params.blur_target_image;
        let device = &renderer.device.device;
        unsafe {
            // 1. SSAO Pass
            let ssao_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(ssao_target_image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[ssao_barrier]);

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(ssao_target_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] } });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            device.cmd_begin_rendering(command_buffer, &rendering_info);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.ssao_pipeline);
            device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.descriptor_sets[current_frame]], &[]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);

            // 2. Blur Pass
            let ssao_to_shader_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(ssao_target_image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            let blur_init_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(blur_target_image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[ssao_to_shader_barrier, blur_init_barrier]);

            let blur_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(blur_target_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [1.0, 1.0, 1.0, 1.0] } });

            let blur_rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&blur_attachment));

            device.cmd_begin_rendering(command_buffer, &blur_rendering_info);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.blur_pipeline);
            device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.blur_layout, 0, &[self.blur_descriptor_sets[current_frame]], &[]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);

            let blur_to_shader_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(blur_target_image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[blur_to_shader_barrier]);
        }
    }

    pub fn update_descriptor_sets_impl(
        &self,
        device: &ash::Device,
        normal_attachments: &[Attachment],
        depth_attachments: &[Attachment],
        ssao_attachments: &[Attachment],
        sampler: vk::Sampler,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let norm_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(normal_attachments[i].view)
                .sampler(sampler)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_attachments[i].view)
                .sampler(sampler)];
            let noise_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.noise_texture.view)
                .sampler(sampler)];
            let param_info = [vk::DescriptorBufferInfo::default()
                .buffer(self.ssao_params_buffer[i].handle)
                .range(self.ssao_params_buffer[i].size)];

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&norm_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&noise_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&param_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]); }

            let ssao_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(ssao_attachments[i].view)
                .sampler(sampler)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth_attachments[i].view)
                .sampler(sampler)];
            let blur_writes = [
                vk::WriteDescriptorSet::default().dst_set(self.blur_descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&ssao_info),
                vk::WriteDescriptorSet::default().dst_set(self.blur_descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
            ];
            unsafe { device.update_descriptor_sets(&blur_writes, &[]); }
        }
    }

}

#[repr(C)]
#[derive(Clone, Copy)]
struct SSAOParamsStruct {
    samples: [Vec4; 64],
    projection: Mat4,
    view: Mat4,
    screen_size: [f32; 2],
}
