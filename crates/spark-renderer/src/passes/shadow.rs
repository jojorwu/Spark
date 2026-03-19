use ash::vk;
use crate::resource::Attachment;
use crate::Renderer;
use crate::vertex::Vertex;
use crate::pipeline::Pipeline;

pub const SHADOW_CASCADE_COUNT: usize = 4;

pub struct ShadowPass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView, // View into the entire array
    pub cascade_views: [vk::ImageView; SHADOW_CASCADE_COUNT],
    pub sampler: vk::Sampler,
}

use super::RenderPass;

impl RenderPass for ShadowPass {
    fn update_descriptor_sets(&self, _renderer: &Renderer) {}
    fn record_commands(&self, renderer: &Renderer, command_buffer: vk::CommandBuffer, _current_frame: usize) {
        // Need to calculate light_view_projs here or pass it through GlobalUBO
        // For simplicity, we'll keep the specialized record_commands and call it from the trait if possible,
        // or refactor the specialized one to take the needed data.
    }
}

impl ShadowPass {
    pub fn new(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
    ) -> Result<Self, crate::error::RendererError> {
        let props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
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
        let mut type_idx = 0;
        for i in 0..props.memory_type_count {
            if (reqs.memory_type_bits & (1 << i)) != 0
                && (props.memory_types[i as usize].property_flags & vk::MemoryPropertyFlags::DEVICE_LOCAL) == vk::MemoryPropertyFlags::DEVICE_LOCAL
            {
                type_idx = i;
                break;
            }
        }

        let memory = unsafe {
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_idx),
                None,
            )?
        };
        unsafe { device.bind_image_memory(image, memory, 0)? };

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
        for i in 0..SHADOW_CASCADE_COUNT {
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
            cascade_views[i] = unsafe { device.create_image_view(&v_info, None)? };
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
            memory,
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

        let binding_descriptions = [
            Vertex::get_binding_description(),
        ];
        let attribute_descriptions = Vertex::get_attribute_descriptions();

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

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

    pub fn record_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        light_view_projs: &[spark_math::Mat4; SHADOW_CASCADE_COUNT],
        renderer: &Renderer,
        object_count: u32,
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
            let mut inheritance_info = vk::CommandBufferInheritanceRenderingInfo::default()
                .depth_attachment_format(vk::Format::D32_SFLOAT);
            let inherit = vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance_info);
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(if is_secondary {
                    vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE | vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                } else {
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                })
                .inheritance_info(&inherit);

            device.begin_command_buffer(command_buffer, &begin_info).unwrap();

            for cascade_idx in 0..SHADOW_CASCADE_COUNT {
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
                    padding: u32,
                    address: u64,
                }
                let frame = &renderer.frames[renderer.current_frame];
                let pc = PC {
                    lvp: light_view_projs[cascade_idx],
                    padding: 0,
                    address: frame.object_data_buffer.map_or(0, |b| b.address),
                };
                let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());

                device.cmd_push_constants(
                    command_buffer,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    pc_bytes,
                );

                if let Some(indirect_buffer) = frame.indirect_commands_buffer {
                    if let (Some(vb), Some(ib)) = (renderer.global_vertex_buffer, renderer.global_index_buffer) {
                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle], &[0]);
                        device.cmd_bind_index_buffer(command_buffer, ib.handle, 0, vk::IndexType::UINT32);

                        if let Some(count_buffer) = frame.draw_count_buffer {
                            device.cmd_draw_indexed_indirect_count(
                                command_buffer,
                                indirect_buffer.handle,
                                0,
                                count_buffer.handle,
                                0,
                                object_count,
                                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                            );
                        } else {
                            device.cmd_draw_indexed_indirect(
                                command_buffer,
                                indirect_buffer.handle,
                                0,
                                object_count,
                                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                            );
                        }
                    }
                }

                device.cmd_end_rendering(command_buffer);
            }
            device.end_command_buffer(command_buffer).unwrap();
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
