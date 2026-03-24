use ash::vk;
use ash::vk::Handle;

pub struct EguiRenderer {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub textures: std::collections::HashMap<egui::TextureId, crate::vulkan::texture::Texture>,
    pub texture_descriptor_sets: std::collections::HashMap<egui::TextureId, vk::DescriptorSet>,
    pub vertex_buffer: Option<crate::Buffer>,
    pub index_buffer: Option<crate::Buffer>,
    pub max_vertices: u64,
    pub max_indices: u64,
}

impl EguiRenderer {
    pub fn new(
        device: &ash::Device,
        vert_shader_code: &[u32],
        frag_shader_code: &[u32],
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> Self {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .unwrap()
        };

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8)]; // vec2 screen_size

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(100)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(100);
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None).unwrap() };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info).unwrap()[0] };

        let pipeline = Self::create_pipeline(
            device,
            pipeline_layout,
            vert_shader_code,
            frag_shader_code,
            extent,
            msaa_samples,
        );

        Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            textures: std::collections::HashMap::new(),
            texture_descriptor_sets: std::collections::HashMap::new(),
            vertex_buffer: None,
            index_buffer: None,
            max_vertices: 0,
            max_indices: 0,
        }
    }

    pub fn draw(
        &mut self,
        renderer: &mut crate::Renderer,
        command_buffer: vk::CommandBuffer,
        full_output: egui::FullOutput,
        screen_size: [f32; 2],
        ctx: &egui::Context,
    ) {
        self.update_textures(renderer, &full_output.textures_delta);
        if self.pipeline == vk::Pipeline::null() {
            return;
        }

        let clipped_primitives = ctx.tessellate(full_output.shapes, full_output.pixels_per_point);

        for primitive in &clipped_primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &primitive.primitive {
                self.update_buffers(
                    renderer,
                    mesh.vertices.len() as u64,
                    mesh.indices.len() as u64,
                );

                if let (Some(vb), Some(ib)) = (&self.vertex_buffer, &self.index_buffer) {
                    renderer.upload_to_buffer(vb, &mesh.vertices);
                    renderer.upload_to_buffer(ib, &mesh.indices);

                    unsafe {
                        let device = renderer.get_device();

                        let color_attachment = vk::RenderingAttachmentInfo::default()
                            .image_view(renderer.get_pass_resource_view("", "GBufferHDR", renderer.current_frame).unwrap())
                            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                            .load_op(vk::AttachmentLoadOp::LOAD)
                            .store_op(vk::AttachmentStoreOp::STORE);

                        let rendering_info = vk::RenderingInfo::default()
                            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: renderer.get_extent() })
                            .layer_count(1)
                            .color_attachments(std::slice::from_ref(&color_attachment));

                        device.cmd_begin_rendering(command_buffer, &rendering_info);

                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline,
                        );
                        let viewport = vk::Viewport::default()
                            .x(0.0)
                            .y(0.0)
                            .width(screen_size[0])
                            .height(screen_size[1])
                            .min_depth(0.0)
                            .max_depth(1.0);
                        let scissor = vk::Rect2D::default().extent(vk::Extent2D {
                            width: screen_size[0] as u32,
                            height: screen_size[1] as u32,
                        });
                        device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                        device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                        let texture_id = mesh.texture_id;
                        let ds = self
                            .texture_descriptor_sets
                            .get(&texture_id)
                            .unwrap_or(&self.descriptor_set);

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            self.pipeline_layout,
                            0,
                            &[*ds],
                            &[],
                        );

                        let bytes =
                            std::slice::from_raw_parts(screen_size.as_ptr() as *const u8, 8);
                        device.cmd_push_constants(
                            command_buffer,
                            self.pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            bytes,
                        );

                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle], &[0]);
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            ib.handle,
                            0,
                            vk::IndexType::UINT32,
                        );
                        device.cmd_draw_indexed(
                            command_buffer,
                            mesh.indices.len() as u32,
                            1,
                            0,
                            0,
                            0,
                        );

                        device.cmd_end_rendering(command_buffer);
                    }
                }
            }
        }
    }

    fn update_buffers(&mut self, renderer: &mut crate::Renderer, vertex_count: u64, index_count: u64) {
        if self.vertex_buffer.is_none() || self.max_vertices < vertex_count {
            if let Some(vb) = self.vertex_buffer.take() {
                renderer.destroy_buffer(vb);
            }
            self.max_vertices = vertex_count.next_power_of_two().max(1024);
            self.vertex_buffer = Some(renderer.create_buffer(
                self.max_vertices * std::mem::size_of::<egui::epaint::Vertex>() as u64,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        }

        if self.index_buffer.is_none() || self.max_indices < index_count {
            if let Some(ib) = self.index_buffer.take() {
                renderer.destroy_buffer(ib);
            }
            self.max_indices = index_count.next_power_of_two().max(1024);
            self.index_buffer = Some(renderer.create_buffer(
                self.max_indices * 4,
                vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        }
    }

    fn create_pipeline(
        device: &ash::Device,
        layout: vk::PipelineLayout,
        vert_code: &[u32],
        frag_code: &[u32],
        _extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
    ) -> vk::Pipeline {
        let vert_module = {
            let info = vk::ShaderModuleCreateInfo::default().code(vert_code);
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let frag_module = {
            let info = vk::ShaderModuleCreateInfo::default().code(frag_code);
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };

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

        let vertex_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(20) // pos: 8, tc: 8, color: 4
            .input_rate(vk::VertexInputRate::VERTEX);

        let vertex_attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding))
            .vertex_attribute_descriptions(&vertex_attrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);

        let multisample =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(msaa_samples);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let color_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        pipeline
    }

    pub fn register_native_texture(&mut self, renderer: &mut crate::Renderer, view: vk::ImageView, sampler: vk::Sampler) -> egui::TextureId {
        let layout = [self.descriptor_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(&layout);

        let ds = unsafe {
            renderer.get_device().allocate_descriptor_sets(&alloc_info).unwrap()[0]
        };

        let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)];

        let write = [vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];

        unsafe {
            renderer.get_device().update_descriptor_sets(&write, &[]);
        }

        let id = egui::TextureId::User(ds.as_raw());
        self.texture_descriptor_sets.insert(id, ds);
        id
    }

    fn update_textures(&mut self, renderer: &mut crate::Renderer, delta: &egui::TexturesDelta) {
        for (id, delta) in &delta.set {
            let (pixels, size) = match &delta.image {
                egui::ImageData::Color(image) => {
                    let pixels: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                    (pixels, [image.size[0] as u32, image.size[1] as u32])
                }
                egui::ImageData::Font(image) => {
                    let pixels: Vec<u8> = image
                        .pixels
                        .iter()
                        .flat_map(|&p| {
                            let v = (p * 255.0) as u8;
                            [v, v, v, v]
                        })
                        .collect();
                    (pixels, [image.size[0] as u32, image.size[1] as u32])
                }
            };

            {
                let staging = renderer.create_buffer(
                    pixels.len() as u64,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                renderer.upload_to_buffer(&staging, &pixels);

                let (image, _old_alloc) = renderer.create_image_basic(
                    &crate::vulkan::device::ImageCreateParams {
                        width: size[0],
                        height: size[1],
                        mip_levels: 1,
                        format: vk::Format::R8G8B8A8_UNORM,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        samples: vk::SampleCountFlags::TYPE_1,
                    }
                );

                renderer.transition_image_layout_basic(
                    image,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    1,
                );
                renderer.copy_buffer_to_image_basic(staging.handle, image, size[0], size[1]);
                renderer.transition_image_layout_basic(
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    1,
                );

                let view = renderer.create_image_view_basic(image, vk::Format::R8G8B8A8_UNORM, 1);
                let sampler = renderer.create_texture_sampler(1);

                renderer.destroy_buffer(staging);

                let allocation = _old_alloc;

                let texture = crate::vulkan::texture::Texture {
                    image,
                    allocation: Some(allocation),
                    view,
                    sampler,
                    mip_levels: 1,
                    bindless_index: 0, // egui textures are handled separately in its own DS for now
                };

                let layouts = [self.descriptor_set_layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&layouts);

                let ds = unsafe {
                    renderer
                        .get_device()
                        .allocate_descriptor_sets(&alloc_info)
                        .unwrap()[0]
                };

                let image_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(view)
                    .sampler(sampler)];

                let write = [vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info)];

                unsafe {
                    renderer.get_device().update_descriptor_sets(&write, &[]);
                }

                self.textures.insert(*id, texture);
                self.texture_descriptor_sets.insert(*id, ds);
            }
        }
        for id in &delta.free {
            if let Some(tex) = self.textures.remove(id) {
                renderer.destroy_texture(tex);
            }
            self.texture_descriptor_sets.remove(id);
        }
    }

    pub fn destroy(&mut self, renderer: &mut crate::Renderer) {
        unsafe {
            let textures = std::mem::take(&mut self.textures);
            for (_, tex) in textures {
                renderer.destroy_texture(tex);
            }
            let device = renderer.get_device();
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }

            if let Some(vb) = self.vertex_buffer.take() {
                renderer.destroy_buffer(vb);
            }
            if let Some(ib) = self.index_buffer.take() {
                renderer.destroy_buffer(ib);
            }
        }
    }
}
