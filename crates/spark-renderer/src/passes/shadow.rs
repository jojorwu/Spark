use ash::vk;
use std::sync::{Arc, Mutex};
use crate::Renderer;
use crate::pipeline::Pipeline;

pub const SHADOW_CASCADE_COUNT: usize = 4;

pub struct ShadowPass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub image: vk::Image,
    pub allocation: Arc<Mutex<Option<gpu_allocator::vulkan::Allocation>>>,
    pub view: vk::ImageView, // View into the entire array
    pub cascade_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    pub sampler: vk::Sampler,
}

use super::{RenderPass, RenderContext};

impl RenderPass for ShadowPass {
    fn name(&self) -> &str { "ShadowPass" }
    fn is_enabled(&self, renderer: &Renderer) -> bool { renderer.enable_shadows }

    fn record_secondary_commands(&self, ctx: &RenderContext) -> Vec<vk::CommandBuffer> {
        let renderer = ctx.renderer;
        let device = &renderer.device.device;
        let lvp = renderer.main_light_view_proj;

        let mut buffers = Vec::new();
        for cascade_idx in 0..SHADOW_CASCADE_COUNT {
            let cb = renderer.allocate_secondary_command_buffer();

            let mut rendering_info = vk::CommandBufferInheritanceRenderingInfo::default().depth_attachment_format(vk::Format::D32_SFLOAT);
            let inheritance = vk::CommandBufferInheritanceInfo::default()
                .push_next(&mut rendering_info);
            let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE).inheritance_info(&inheritance);

            unsafe {
                device.begin_command_buffer(cb, &begin).unwrap();
                self.record_cascade_commands(device, cb, lvp, renderer, cascade_idx, true);
                device.end_command_buffer(cb).unwrap();
            }
            buffers.push(cb);
        }
        buffers
    }


    fn record_commands(&self, ctx: &RenderContext) {
        let lvp = ctx.renderer.main_light_view_proj;
        self.record_commands_impl(
            &ctx.renderer.device.device,
            ctx.command_buffer,
            &[lvp; 4],
            ctx.renderer,
            ctx.renderer.last_object_count,
            false // Not secondary if called this way
        );
    }

    fn get_resource_view(&self, name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        if name == "shadow_map" {
            Some(self.view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &Renderer) {
        unsafe {
            let device = &renderer.device.device;
            for i in 0..SHADOW_CASCADE_COUNT {
                device.destroy_image_view(self.cascade_views[i], None);
            }
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            if let Some(alloc) = self.allocation.lock().unwrap().take() {
                renderer.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }
}

impl ShadowPass {
    pub fn record_commands_internal(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        light_view_projs: &[spark_math::Mat4; SHADOW_CASCADE_COUNT],
        renderer: &Renderer,
        object_count: u32,
        is_secondary: bool,
    ) {
        self.record_commands_impl(device, command_buffer, light_view_projs, renderer, object_count, is_secondary);
    }

    pub fn new(
        device_wrapper: &crate::vulkan::device::VulkanDevice,
    ) -> Result<Self, crate::error::RendererError> {
        let device = &device_wrapper.device;
        let extent = vk::Extent3D {
            width: Renderer::SHADOW_MAP_CASCADE_SIZE,
            height: Renderer::SHADOW_MAP_CASCADE_SIZE,
            depth: 1,
        };

        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(1)
            .array_layers(SHADOW_CASCADE_COUNT as u32)
            .format(vk::Format::D32_SFLOAT)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe { device.create_image(&image_info, None)? };
        let reqs = unsafe { device.get_image_memory_requirements(image) };

        let allocation = device_wrapper.allocator.lock().unwrap().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Shadow Map",
            requirements: reqs,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }).map_err(|_| crate::error::RendererError::NoSuitableDevice)?;

        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::D32_SFLOAT)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: SHADOW_CASCADE_COUNT as u32,
            });
        let view = unsafe { device.create_image_view(&view_info, None)? };

        let mut cascade_views = [vk::ImageView::null(); SHADOW_CASCADE_COUNT];
        for (i, view) in cascade_views.iter_mut().enumerate() {
             let v_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: i as u32,
                    layer_count: 1,
                });
            *view = unsafe { device.create_image_view(&v_info, None)? };
        }

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(128),
                ]),
                None,
            )?
        };

        Ok(Self {
            pipeline: None,
            layout,
            image,
            allocation: Arc::new(Mutex::new(Some(allocation))),
            view,
            cascade_views,
            sampler,
        })
    }


    pub fn create_pipeline(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
    ) {
        let vert_module = Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = Pipeline::create_shader_module(device, frag_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();
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

        let viewport = vk::Viewport::default()
            .width(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
            .height(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: Renderer::SHADOW_MAP_CASCADE_SIZE,
            height: Renderer::SHADOW_MAP_CASCADE_SIZE,
        });
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(1.25)
            .depth_bias_slope_factor(1.75);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        self.pipeline = Some(unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[info], None)
                .unwrap()[0]
        });

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }
    }

    fn record_cascade_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        lvp: spark_math::Mat4,
        renderer: &Renderer,
        cascade_idx: usize,
        is_secondary: bool,
    ) {
        let pipeline = match self.pipeline {
            Some(p) => p,
            None => return,
        };

        let clear = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        unsafe {
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.cascade_views[cascade_idx])
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear[0]);

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: Renderer::SHADOW_MAP_CASCADE_SIZE,
                        height: Renderer::SHADOW_MAP_CASCADE_SIZE,
                    },
                })
                .layer_count(1)
                .depth_attachment(&depth_attachment);

            if is_secondary {
                // Secondary buffers used in dynamic rendering don't use this flag in begin_rendering,
                // but the inheritance info must match.
            }

            device.cmd_begin_rendering(command_buffer, &rendering_info);

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

            let shadow_viewport = vk::Viewport::default()
                .width(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
                .height(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            let shadow_scissor = vk::Rect2D::default().extent(vk::Extent2D {
                width: Renderer::SHADOW_MAP_CASCADE_SIZE,
                height: Renderer::SHADOW_MAP_CASCADE_SIZE,
            });
            device.cmd_set_viewport(command_buffer, 0, &[shadow_viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[shadow_scissor]);

            #[repr(C)]
            struct PC {
                lvp: spark_math::Mat4,
                address: u64,
                vertex_address: u64,
            }
            let frame = &renderer.frames[renderer.current_frame];
            let pc = PC {
                lvp,
                address: frame.object_data_buffer.as_ref().map_or(0, |b| b.address),
                vertex_address: renderer.global_vertex_buffer.as_ref().map_or(0, |b| b.address),
            };
            let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());

            device.cmd_push_constants(
                command_buffer,
                self.layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                pc_bytes,
            );

            if let Some(ref indirect_buffer) = frame.indirect_commands_buffer {
                if let Some(ib) = renderer.global_index_buffer.as_ref() {
                    device.cmd_bind_index_buffer(command_buffer, ib.handle, 0, vk::IndexType::UINT32);

                    if let Some(ref count_buffer) = frame.draw_count_buffer {
                        device.cmd_draw_indexed_indirect_count(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            count_buffer.handle,
                            0,
                            renderer.last_object_count,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    } else {
                        device.cmd_draw_indexed_indirect(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            renderer.last_object_count,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    }
                }
            }

            device.cmd_end_rendering(command_buffer);
        }
    }

    fn record_commands_impl(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        light_view_projs: &[spark_math::Mat4; SHADOW_CASCADE_COUNT],
        renderer: &Renderer,
        _object_count: u32,
        is_secondary: bool,
    ) {
        for (cascade_idx, &lvp) in light_view_projs.iter().enumerate() {
            self.record_cascade_commands(device, command_buffer, lvp, renderer, cascade_idx, is_secondary);
        }
    }

}
