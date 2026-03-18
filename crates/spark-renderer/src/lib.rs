pub mod error;
pub mod pipeline;
pub mod passes;
pub mod resource;
pub mod ui;
pub mod vertex;
pub mod vulkan;

use crate::error::RendererError;
use crate::passes::deferred::DeferredPass;
use crate::passes::post_process::PostProcessPass;
use crate::passes::shadow::ShadowPass;
use crate::pipeline::Pipeline;
pub use crate::resource::{Attachment, Buffer, RenderFrame, MAX_FRAMES_IN_FLIGHT};
use crate::ui::EguiRenderer;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::gbuffer::GBuffer;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
pub use ash;
use ash::vk;
use winit::window::Window;

/// The main renderer of the Spark Engine.
pub struct Renderer {
    context: VulkanContext,
    pub device: VulkanDevice,
    swapchain: VulkanSwapchain,
    pub gbuffer: GBuffer,
    pub global_descriptor_set_layout: vk::DescriptorSetLayout,
    pub light_count: u32,
    pub shadow_pass: ShadowPass,
    pub deferred_pass: DeferredPass,
    pub post_process_pass: PostProcessPass,
    pub frames: [RenderFrame; MAX_FRAMES_IN_FLIGHT],
    pub pipeline_cache: vk::PipelineCache,
    current_frame: usize,
    pub pipeline: Option<Pipeline>,
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_descriptor_sets: std::collections::HashMap<vk::ImageView, vk::DescriptorSet>,
    pub default_texture: Option<Texture>,
    pub default_descriptor_set: vk::DescriptorSet,
    pub scene_view_matrix_for_pos: spark_math::Mat4,
    egui_renderer: Option<EguiRenderer>,
}

impl Renderer {
    /// Cascaded shadow map texture size.
    pub const SHADOW_MAP_CASCADE_SIZE: u32 = 2048;

    /// Creates a new Renderer instance.
    pub fn new(
        window: &Window,
        ui_shaders: Option<(&[u32], &[u32])>,
    ) -> Result<Self, RendererError> {
        let context = VulkanContext::new(window)?;
        let device =
            VulkanDevice::new(&context.instance, &context.surface_loader, context.surface)?;
        let swapchain = VulkanSwapchain::new(
            &context.instance,
            &device.device,
            device.pdevice,
            &context.surface_loader,
            context.surface,
            window.inner_size().width,
            window.inner_size().height,
        )?;

        let shadow_pass = ShadowPass::new(&device.device, device.pdevice, &context.instance)?;

        let gbuffer = GBuffer::new(
            &device.device,
            device.pdevice,
            &context.instance,
            swapchain.extent,
            device.msaa_samples,
            device.depth_format,
        );

        let mut post_process_pass = PostProcessPass::new(
            &device.device,
            device.pdevice,
            &context.instance,
            swapchain.format,
            swapchain.extent,
        )?;
        post_process_pass.create_framebuffers(&device.device, &swapchain.views, swapchain.extent);

        let (av, fi, in_f) = Self::create_sync_objects_impl(&device.device);
        let pipeline_cache = unsafe {
            device.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
        };

        let global_descriptor_set_layout = unsafe {
            device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
                ]),
                None,
            )?
        };

        let descriptor_pool = Self::create_descriptor_pool_impl(&device.device);

        let global_descriptor_sets = unsafe {
            device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[global_descriptor_set_layout, global_descriptor_set_layout]),
            )?
        };

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(device.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cb = unsafe { device.device.allocate_command_buffers(&alloc_info).unwrap()[0] };

                RenderFrame {
                    command_buffer: cb,
                    image_available: av[i],
                    render_finished: fi[i],
                    in_flight: in_f[i],
                    global_buffer: None,
                    light_buffer: None,
                    global_descriptor_set: global_descriptor_sets[i],
                    instance_buffers: Vec::new(),
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let deferred_pass =
            DeferredPass::new(&device.device, descriptor_pool, global_descriptor_set_layout)?;

        let egui_renderer = ui_shaders.map(|(v, f)| {
            EguiRenderer::new(
                &device.device,
                gbuffer.render_pass,
                v,
                f,
                swapchain.extent,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        Ok(Self {
            context,
            device,
            swapchain,
            gbuffer,
            global_descriptor_set_layout,
            light_count: 0,
            shadow_pass,
            deferred_pass,
            post_process_pass,
            frames,
            pipeline_cache,
            current_frame: 0,
            pipeline: None,
            vertex_buffers: Vec::new(),
            index_buffer: None,
            descriptor_pool,
            texture_descriptor_sets: std::collections::HashMap::new(),
            default_texture: None,
            default_descriptor_set: vk::DescriptorSet::null(),
            scene_view_matrix_for_pos: spark_math::Mat4::IDENTITY,
            egui_renderer,
        })
    }

    pub fn set_deferred_pipeline(&mut self, pipeline: vk::Pipeline) {
        self.deferred_pass.pipeline = Some(pipeline);
        self.update_deferred_descriptor_sets();
    }

    fn update_deferred_descriptor_sets(&self) {
        let light_buffers: Vec<Buffer> = self.frames.iter().filter_map(|f| f.light_buffer).collect();
        self.deferred_pass.update_descriptor_sets(
            &self.device.device,
            &self.gbuffer.albedo,
            &self.gbuffer.normal,
            &self.gbuffer.pbr,
            &self.gbuffer.depth,
            self.shadow_pass.view,
            self.shadow_pass.sampler,
            &light_buffers,
        );
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        if let Some(old) = &self.pipeline {
            if old.descriptor_set_layout != pipeline.descriptor_set_layout {
                self.texture_descriptor_sets.clear();
            }
        }
        self.pipeline = Some(pipeline);
        self.init_default_resources();
    }

    fn init_default_resources(&mut self) {
        if self.default_texture.is_none() {
            let white_pixel = [255u8, 255, 255, 255];
            let img = image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(1, 1, white_pixel.to_vec()).unwrap(),
            );
            let tex = self.create_texture_from_image(&img);
            self.default_texture = Some(tex);
        }

        let view = self.default_texture.as_ref().unwrap().view;
        let sampler = self.default_texture.as_ref().unwrap().sampler;

        let pipeline_layout = if let Some(pipeline) = &self.pipeline {
            pipeline.descriptor_set_layout
        } else {
            return;
        };

        let device = &self.device.device;
        let ds = *self
            .texture_descriptor_sets
            .entry(view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&[pipeline_layout]),
                    )
                    .expect("Failed to allocate texture descriptor set")[0]
            });

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        self.default_descriptor_set = ds;
    }

    pub fn create_shadow_pipeline(&mut self, vert_spirv: &[u32], frag_spirv: &[u32]) {
        self.shadow_pass.create_pipeline(
            &self.device.device,
            self.pipeline_cache,
            vert_spirv,
            frag_spirv,
        );
    }

    pub fn create_post_process_pipeline(&mut self, vert_spirv: &[u32], frag_spirv: &[u32], bloom_frag_spirv: &[u32]) {
        self.post_process_pass.create_pipelines(
            &self.device.device,
            self.pipeline_cache,
            self.swapchain.extent,
            vert_spirv,
            frag_spirv,
            bloom_frag_spirv,
        );
        self.update_post_process_descriptor_sets();
    }

    fn update_post_process_descriptor_sets(&self) {
        self.post_process_pass.update_descriptor_sets(
            &self.device.device,
            &self.gbuffer.hdr,
            self.shadow_pass.sampler,
        );
    }

    pub fn ensure_global_descriptor_set(&mut self) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let frame = &self.frames[i];
            if let Some(global_buffer) = frame.global_buffer {
                let ds = frame.global_descriptor_set;
                let buf_info = [vk::DescriptorBufferInfo::default()
                    .buffer(global_buffer.handle)
                    .offset(0)
                    .range(global_buffer.size)];
                let writes = [vk::WriteDescriptorSet::default()
                    .dst_set(ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buf_info)];
                unsafe {
                    self.device.device.update_descriptor_sets(&writes, &[]);
                }
            }
        }
    }

    pub fn ensure_texture_descriptor(&mut self, texture: &Texture) {
        let pipeline_layout = if let Some(pipeline) = &self.pipeline {
            pipeline.descriptor_set_layout
        } else {
            return;
        };

        let device = &self.device.device;
        let ds = *self
            .texture_descriptor_sets
            .entry(texture.view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&[pipeline_layout]),
                    )
                    .expect("Failed to allocate texture descriptor set")[0]
            });

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.view)
            .sampler(texture.sampler)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn update_lights(&mut self, lights: &[(spark_math::Vec3, spark_math::Vec3, f32)]) {
        let count = lights.len();
        if count == 0 {
            return;
        }
        self.light_count = count as u32;
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LD {
            pos: [f32; 4],
            col: [f32; 4],
        }
        let ld: Vec<LD> = lights
            .iter()
            .map(|(p, c, i)| LD {
                pos: [p.x, p.y, p.z, 1.0],
                col: [c.x, c.y, c.z, *i],
            })
            .collect();
        let sz = (ld.len() * std::mem::size_of::<LD>()) as u64;

        let mut needs_reupdate = false;
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let needs_new_buffer = {
                let frame = &self.frames[i];
                frame.light_buffer.is_none() || frame.light_buffer.unwrap().size < sz
            };

            if needs_new_buffer {
                let new_buffer = self.create_buffer(
                    sz,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );

                let frame = &mut self.frames[i];
                if let Some(old) = frame.light_buffer {
                    self.device.destroy_buffer(old);
                }
                frame.light_buffer = Some(new_buffer);
                needs_reupdate = true;
            }
        }

        if needs_reupdate {
            self.update_deferred_descriptor_sets();
        }

        let lb = self.frames[self.current_frame].light_buffer.unwrap();
        self.upload_to_buffer(&lb, &ld);
    }

    pub fn add_vertex_buffer(&mut self, buffer: Buffer) -> u32 {
        self.vertex_buffers.push(buffer);
        (self.vertex_buffers.len() - 1) as u32
    }

    pub fn add_instance_buffer(&mut self, buffer: Buffer) -> u32 {
        let frame = &mut self.frames[self.current_frame];
        frame.instance_buffers.push(buffer);
        (frame.instance_buffers.len() - 1) as u32
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_frame(
        &mut self,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        window: &Window,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        let (image_available, in_flight, command_buffer, render_finished) = {
            let frame = &self.frames[self.current_frame];
            (frame.image_available, frame.in_flight, frame.command_buffer, frame.render_finished)
        };

        unsafe {
            self.device
                .device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("Failed to wait for fence");
            let result = self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            );
            let image_index = match result {
                Ok((index, _)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    let _ = self.recreate_swapchain(window);
                    return;
                }
                Err(e) => panic!("Failed to acquire swapchain image: {:?}", e),
            };
            self.device
                .device
                .reset_fences(&[in_flight])
                .expect("Failed to reset fence");
            self.device
                .device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");
            self.record_command_buffer(
                image_index,
                renderables,
                instanced_renderables,
                view_proj,
                light_view_proj,
                egui_output,
            );
            let s_available = [image_available];
            let s_finished = [render_finished];
            let w_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let c_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&s_available)
                .wait_dst_stage_mask(&w_stages)
                .command_buffers(&c_buffers)
                .signal_semaphores(&s_finished);
            self.device
                .device
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    in_flight,
                )
                .unwrap();
            let result = self.swapchain.loader.queue_present(
                self.device.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&s_finished)
                    .swapchains(&[self.swapchain.handle])
                    .image_indices(&[image_index]),
            );
            match result {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let _ = self.recreate_swapchain(window);
                }
                Err(e) => panic!("Failed to present swapchain image: {:?}", e),
            }
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }

    fn record_command_buffer(
        &mut self,
        image_index: u32,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        let frame = &self.frames[self.current_frame];
        let command_buffer = frame.command_buffer;
        unsafe {
            self.device
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            // 1. Shadow Pass
            self.shadow_pass.record_commands(
                &self.device.device,
                command_buffer,
                light_view_proj,
                renderables,
                instanced_renderables,
                self,
            );

            // 2. Main Pass (Geometry + Lighting)
            let clear = [
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } },
                vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.1, 1.0] } },
                vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } },
            ];

            self.device.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.gbuffer.render_pass)
                    .framebuffer(self.gbuffer.framebuffers[self.current_frame])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );

            #[repr(C)]
            #[derive(Copy, Clone)]
            struct GlobalUBO {
                vp: spark_math::Mat4,
                lvp: spark_math::Mat4,
                inv_vp: spark_math::Mat4,
                camera_pos: [f32; 4],
            }

            let inv_v = self.scene_view_matrix_for_pos.inverse();
            let camera_pos = [inv_v.w_axis.x, inv_v.w_axis.y, inv_v.w_axis.z, 1.0];
            let ubo = GlobalUBO {
                vp: view_proj,
                lvp: light_view_proj,
                inv_vp: view_proj.inverse(),
                camera_pos,
            };

            {
                let needs_init = self.frames[0].global_buffer.is_none();
                if needs_init {
                    for i in 0..MAX_FRAMES_IN_FLIGHT {
                        let new_buffer = self.create_buffer(
                            std::mem::size_of::<GlobalUBO>() as u64,
                            vk::BufferUsageFlags::UNIFORM_BUFFER,
                            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                        );
                        self.frames[i].global_buffer = Some(new_buffer);
                    }
                    self.ensure_global_descriptor_set();
                }
                let gb = self.frames[self.current_frame].global_buffer.unwrap();
                self.upload_to_buffer(&gb, &[ubo]);
            }

            #[repr(C)]
            struct PC {
                count: u32,
                metallic: f32,
                roughness: f32,
                width: f32,
                height: f32,
            }
            let pc = PC {
                count: self.light_count,
                metallic: 0.5,
                roughness: 0.5,
                width: self.swapchain.extent.width as f32,
                height: self.swapchain.extent.height as f32,
            };
            let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());

            self.deferred_pass.record_commands(
                self,
                command_buffer,
                renderables,
                instanced_renderables,
                pc_bytes,
                self.current_frame,
            );

            if let Some((output, ctx)) = egui_output {
                let ext = self.swapchain.extent;
                if let Some(mut egui) = self.egui_renderer.take() {
                    egui.draw(
                        self,
                        command_buffer,
                        output,
                        [ext.width as f32, ext.height as f32],
                        &ctx,
                    );
                    self.egui_renderer = Some(egui);
                }
            }

            self.device.device.cmd_end_render_pass(command_buffer);

            self.post_process_pass.record_commands(
                &self.device.device,
                command_buffer,
                image_index,
                self.current_frame,
                self.swapchain.images[image_index as usize],
                self.swapchain.extent,
            );

            self.device
                .device
                .end_command_buffer(command_buffer)
                .unwrap();
        }
    }

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), RendererError> {
        unsafe {
            self.device.device.device_wait_idle()?;
            self.cleanup_swapchain();
            self.swapchain = VulkanSwapchain::new(
                &self.context.instance,
                &self.device.device,
                self.device.pdevice,
                &self.context.surface_loader,
                self.context.surface,
                window.inner_size().width,
                window.inner_size().height,
            )?;
            self.gbuffer.recreate(
                &self.device.device,
                self.device.pdevice,
                &self.context.instance,
                self.swapchain.extent,
                self.device.msaa_samples,
                self.device.depth_format,
            );
            self.post_process_pass.create_framebuffers(
                &self.device.device,
                &self.swapchain.views,
                self.swapchain.extent,
            );
            self.update_deferred_descriptor_sets();
            self.update_post_process_descriptor_sets();
        }
        Ok(())
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.gbuffer.destroy(&self.device.device);
            for &f in &self.post_process_pass.framebuffers {
                self.device.device.destroy_framebuffer(f, None);
            }
            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.handle, None);
        }
    }

    pub fn create_texture_from_image(&self, img: &image::DynamicImage) -> Texture {
        let (w, h) = (img.width(), img.height());
        let mip = (((w.max(h) as f32).log2().floor()) as u32) + 1;
        let rgba = img.to_rgba8();
        let pix = rgba.as_raw();
        let sz = pix.len() as u64;
        let st = self.create_buffer(
            sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        self.upload_to_buffer(&st, pix);
        let (i, m) = self.create_image_basic(
            w,
            h,
            mip,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.transition_image_layout_basic(
            i,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip,
        );
        let cb = self.begin_single_time_commands();
        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                cb,
                st.handle,
                i,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: w,
                        height: h,
                        depth: 1,
                    })],
            );
        }
        self.end_single_time_commands(cb);
        self.generate_mipmaps(i, vk::Format::R8G8B8A8_SRGB, w, h, mip);
        let v = self.create_image_view_basic(i, vk::Format::R8G8B8A8_SRGB, mip);
        let s = self.create_texture_sampler(mip);
        self.destroy_buffer(st);

        Texture {
            image: i,
            memory: m,
            view: v,
            sampler: s,
            mip_levels: mip,
        }
    }

    pub fn destroy_texture(&mut self, t: Texture) {
        self.texture_descriptor_sets.remove(&t.view);
        unsafe {
            self.device.device.destroy_sampler(t.sampler, None);
            self.device.device.destroy_image_view(t.view, None);
            self.device.device.destroy_image(t.image, None);
            self.device.device.free_memory(t.memory, None);
        }
    }

    pub fn create_image_basic(
        &self,
        w: u32,
        h: u32,
        mip: u32,
        f: vk::Format,
        t: vk::ImageTiling,
        u: vk::ImageUsageFlags,
        p: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        self.device.create_image(w, h, mip, f, t, u, p)
    }

    pub fn create_texture_sampler(&self, mip: u32) -> vk::Sampler {
        let props = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.device.pdevice)
        };
        unsafe {
            self.device
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .anisotropy_enable(true)
                        .max_anisotropy(props.limits.max_sampler_anisotropy)
                        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .min_lod(0.0)
                        .max_lod(mip as f32),
                    None,
                )
                .unwrap()
        }
    }

    fn generate_mipmaps(&self, i: vk::Image, _f: vk::Format, w: u32, h: u32, mip: u32) {
        let props = unsafe {
            self.context
                .instance
                .get_physical_device_format_properties(self.device.pdevice, _f)
        };
        if !props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear filtering not supported");
        }
        let cb = self.begin_single_time_commands();
        let mut bar = vk::ImageMemoryBarrier::default()
            .image(i)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });
        let (mut mw, mut mh) = (w as i32, h as i32);
        for j in 1..mip {
            bar.subresource_range.base_mip_level = j - 1;
            bar.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            bar.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            bar.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            bar.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[bar],
                );
            }
            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: mw, y: mh, z: 1 },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: j - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: (mw / 2).max(1),
                        y: (mh / 2).max(1),
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: j,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                self.device.device.cmd_blit_image(
                    cb,
                    i,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    i,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }
            bar.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            bar.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            bar.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            bar.dst_access_mask = vk::AccessFlags::SHADER_READ;
            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[bar],
                );
            }
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
        }
        bar.subresource_range.base_mip_level = mip - 1;
        bar.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        bar.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        bar.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        bar.dst_access_mask = vk::AccessFlags::SHADER_READ;
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[bar],
            );
        }
        self.end_single_time_commands(cb);
    }


    pub fn get_buffer(&self, id: u32) -> Option<&Buffer> {
        self.vertex_buffers.get(id as usize)
    }
    pub fn get_instance_buffer(&self, id: u32) -> Option<&Buffer> {
        self.frames[self.current_frame].instance_buffers.get(id as usize)
    }
    /// Clears instance buffers for the current frame to prevent leaks.
    pub fn clear_instance_buffers(&mut self) {
        let buffers = std::mem::take(&mut self.frames[self.current_frame].instance_buffers);
        for b in buffers {
            self.device.destroy_buffer(b);
        }
    }
    /// Returns the graphics queue handle.
    pub fn get_graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }
    /// Returns the configured MSAA sample count.
    pub fn get_msaa_samples(&self) -> vk::SampleCountFlags {
        self.device.msaa_samples
    }
    /// Returns the current swapchain extent.
    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }
    /// Returns the raw ash::Device.
    pub fn get_device(&self) -> &ash::Device {
        &self.device.device
    }

    /// Uploads data to a GPU buffer.
    ///
    /// # Safety
    /// The user must ensure the buffer is not being used by the GPU when calling this,
    /// or that the buffer is host-coherent and visible.
    pub fn upload_to_buffer<T: Copy>(&self, b: &Buffer, data: &[T]) {
        self.device.upload_to_buffer(b, data);
    }
    pub fn create_buffer(
        &self,
        sz: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        self.device.create_buffer(sz, usage, properties)
    }
    pub fn destroy_buffer(&self, buffer: Buffer) {
        self.device.destroy_buffer(buffer);
    }
    pub fn create_image_view_basic(&self, i: vk::Image, f: vk::Format, mip: u32) -> vk::ImageView {
        self.device.create_image_view(i, f, mip)
    }
    pub fn transition_image_layout_basic(
        &self,
        i: vk::Image,
        ol: vk::ImageLayout,
        nl: vk::ImageLayout,
        mip: u32,
    ) {
        self.device.transition_image_layout(i, ol, nl, mip)
    }
    pub fn copy_buffer_to_image_basic(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        self.end_single_time_commands(command_buffer);
    }
    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        self.device.begin_single_time_commands()
    }
    pub fn end_single_time_commands(&self, cb: vk::CommandBuffer) {
        self.device.end_single_time_commands(cb);
    }

    fn create_descriptor_pool_impl(device: &ash::Device) -> vk::DescriptorPool {
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1000),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(100),
        ];
        unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .max_sets(1000),
                    None,
                )
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_sync_objects_impl(
        device: &ash::Device,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut av = Vec::new();
        let mut fi = Vec::new();
        let mut in_f = Vec::new();
        let s_info = vk::SemaphoreCreateInfo::default();
        let f_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                av.push(device.create_semaphore(&s_info, None).unwrap());
                fi.push(device.create_semaphore(&s_info, None).unwrap());
                in_f.push(device.create_fence(&f_info, None).unwrap());
            }
        }
        (av, fi, in_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_struct() {
        let buffer = Buffer {
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
        };
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.handle, vk::Buffer::null());
    }

    #[test]
    fn test_light_data_layout() {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LD {
            pos: [f32; 4],
            col: [f32; 4],
        }
        assert_eq!(std::mem::size_of::<LD>(), 32);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().ok();
            self.cleanup_swapchain();

            for frame in &mut self.frames {
                if let Some(lb) = frame.light_buffer.take() {
                    self.device.destroy_buffer(lb);
                }
                if let Some(gb) = frame.global_buffer.take() {
                    self.device.destroy_buffer(gb);
                }
                for ib in frame.instance_buffers.drain(..) {
                    self.device.destroy_buffer(ib);
                }
                self.device.device.destroy_semaphore(frame.image_available, None);
                self.device.device.destroy_semaphore(frame.render_finished, None);
                self.device.device.destroy_fence(frame.in_flight, None);
            }

            self.shadow_pass.destroy(&self.device.device);
            self.deferred_pass.destroy(&self.device.device);
            self.post_process_pass.destroy(&self.device.device);

            if let Some(p) = self.pipeline.take() {
                self.device
                    .device
                    .destroy_pipeline(p.graphics_pipeline, None);
                self.device.device.destroy_pipeline_layout(p.layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(p.descriptor_set_layout, None);
            }
            self.texture_descriptor_sets.clear();
            self.device
                .device
                .destroy_descriptor_set_layout(self.global_descriptor_set_layout, None);
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            if let Some(mut e) = self.egui_renderer.take() {
                e.destroy(self);
            }
            if let Some(t) = self.default_texture.take() {
                self.destroy_texture(t);
            }
            let vbs = std::mem::take(&mut self.vertex_buffers);
            for vb in vbs {
                self.device.destroy_buffer(vb);
            }
            if let Some(ib) = self.index_buffer.take() {
                self.device.destroy_buffer(ib);
            }
        }
    }
}
