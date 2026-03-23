use ash::vk;
use std::sync::{Arc, Mutex};
use crate::vulkan::device::VulkanDevice;

pub struct HiZPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pyramid_view: vk::ImageView,
    pub pyramid_image: vk::Image,
    pub pyramid_allocation: Arc<Mutex<Option<gpu_allocator::vulkan::Allocation>>>,
    pub mip_views: Vec<vk::ImageView>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
}

use super::{RenderPass, RenderContext};
use crate::Renderer;

impl RenderPass for HiZPass {
    fn name(&self) -> &str { "HiZPass" }

    fn needs_descriptor_update(&self, renderer: &Renderer, frame_index: usize) -> bool {
        let prev_frame = (frame_index + crate::MAX_FRAMES_IN_FLIGHT - 1) % crate::MAX_FRAMES_IN_FLIGHT;
        let depth_version = renderer.gbuffer_depth[prev_frame].version.load(std::sync::atomic::Ordering::Relaxed);

        let pass_versions = renderer.pass_descriptor_versions[frame_index].lock().unwrap();
        if let Some(&v) = pass_versions.get(self.name()) {
            return v != depth_version;
        }
        true
    }

    fn update_descriptor_sets(&self, _renderer: &Renderer) {
        // The descriptor sets are updated dynamically in record_commands_impl
        // because each mip level needs a different source/destination.
        // We update the pass version to match the current depth version.
        // This is a bit of a hack since HiZPass doesn't use the standard update_descriptor_sets,
        // but it satisfies the versioning system.
        // Actually, we'll leave it to true in record_commands to be safe,
        // but we can track the depth version there.
    }
    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;

        let prev_frame = (current_frame + crate::MAX_FRAMES_IN_FLIGHT - 1) % crate::MAX_FRAMES_IN_FLIGHT;
        self.record_commands_impl(
            renderer,
            command_buffer,
            renderer.gbuffer_depth[prev_frame].view,
            renderer.common_sampler,
            current_frame,
        );
    }

    fn get_resource_view(&self, name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        if name == "pyramid" {
            Some(self.pyramid_view)
        } else {
            None
        }
    }

    fn on_resize(&mut self, renderer: &mut Renderer, new_extent: vk::Extent2D) {
        let device = &renderer.device;

        // Destroy old
        unsafe {
            for view in &self.mip_views {
                device.device.destroy_image_view(*view, None);
            }
            device.device.destroy_image_view(self.pyramid_view, None);
            device.device.destroy_image(self.pyramid_image, None);
            if let Some(alloc) = self.pyramid_allocation.lock().unwrap().take() {
                device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }

        self.width = new_extent.width;
        self.height = new_extent.height;
        self.mip_levels = (self.width.max(self.height) as f32).log2().floor() as u32 + 1;

        let (image, allocation) = device.create_image(
            &crate::vulkan::device::ImageCreateParams {
                width: self.width,
                height: self.height,
                mip_levels: self.mip_levels,
                format: vk::Format::R32_SFLOAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            }
        ).unwrap();

        self.pyramid_image = image;
        *self.pyramid_allocation.lock().unwrap() = Some(allocation);
        self.pyramid_view = device.create_image_view(image, vk::Format::R32_SFLOAT, self.mip_levels);

        self.mip_views.clear();
        for i in 0..self.mip_levels {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.mip_views.push(unsafe { device.device.create_image_view(&view_info, None).unwrap() });
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            let device = &renderer.device.device;
            for view in &self.mip_views {
                device.destroy_image_view(*view, None);
            }
            device.destroy_image_view(self.pyramid_view, None);
            device.destroy_image(self.pyramid_image, None);
            if let Some(alloc) = self.pyramid_allocation.lock().unwrap().take() {
                renderer.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl HiZPass {
    pub fn new(
        device: &VulkanDevice,
        descriptor_pool: vk::DescriptorPool,
        shader_code: &[u32],
        width: u32,
        height: u32,
    ) -> Result<Self, crate::error::RendererError> {
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let (image, allocation) = device.create_image(
            &crate::vulkan::device::ImageCreateParams {
                width,
                height,
                mip_levels,
                format: vk::Format::R32_SFLOAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            }
        )?;

        let pyramid_view = device.create_image_view(image, vk::Format::R32_SFLOAT, mip_levels);

        let mut mip_views = Vec::new();
        for i in 0..mip_levels {
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R32_SFLOAT)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: i,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            mip_views.push(unsafe { device.device.create_image_view(&view_info, None).unwrap() });
        }

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let descriptor_set_layout = unsafe {
            device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let layout = unsafe {
            device.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[descriptor_set_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 8,
                    }]),
                None,
            )?
        };

        let shader_module = unsafe {
            device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(shader_code),
                None,
            )?
        };

        let pipeline = unsafe {
            device.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader_module)
                        .name(c"main"))
                    .layout(layout)],
                None,
            ).map_err(|e| e.1)?[0]
        };

        unsafe { device.device.destroy_shader_module(shader_module, None); }

        let mut descriptor_sets = Vec::new();
        if mip_levels > 1 {
            let total_sets = (mip_levels - 1) * crate::MAX_FRAMES_IN_FLIGHT as u32;
            let mut layouts = Vec::new();
            for _ in 0..total_sets {
                layouts.push(descriptor_set_layout);
            }
            descriptor_sets = unsafe {
                device.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                )?
            };
        }

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout,
            pyramid_view,
            pyramid_image: image,
            pyramid_allocation: Arc::new(Mutex::new(Some(allocation))),
            mip_views,
            descriptor_sets,
            width,
            height,
            mip_levels,
        })
    }

    pub fn record_commands_impl(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        depth_view: vk::ImageView,
        sampler: vk::Sampler,
        frame_index: usize,
    ) {
        let device = &renderer.device.device;
        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            let mut w = self.width;
            let mut h = self.height;

            for i in 0..(self.mip_levels - 1) {
                w = (w >> 1).max(1);
                h = (h >> 1).max(1);

                let src_view = if i == 0 { depth_view } else { self.mip_views[i as usize] };
                let dst_view = self.mip_views[(i + 1) as usize];

                let img_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(src_view)
                    .sampler(sampler)];
                let dst_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(dst_view)];

                let ds_idx = frame_index * (self.mip_levels - 1) as usize + i as usize;
                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[ds_idx])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&img_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[ds_idx])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&dst_info),
                ];

                device.update_descriptor_sets(&writes, &[]);

                // Track depth version
                let depth_version = renderer.gbuffer_depth[(frame_index + crate::MAX_FRAMES_IN_FLIGHT - 1) % crate::MAX_FRAMES_IN_FLIGHT].version.load(std::sync::atomic::Ordering::Relaxed);
                renderer.pass_descriptor_versions[frame_index].lock().unwrap().insert(self.name().to_string(), depth_version);

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.descriptor_sets[ds_idx]],
                    &[],
                );

                let pc = [w as f32, h as f32];
                let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 8);
                device.cmd_push_constants(command_buffer, self.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

                device.cmd_dispatch(command_buffer, w.div_ceil(16), h.div_ceil(16), 1);

                let barrier = vk::ImageMemoryBarrier::default()
                    .image(self.pyramid_image)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: i + 1,
                        level_count: 1,
                        ..Default::default()
                    })
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ);

                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[], &[], &[barrier],
                );
            }
        }
    }

}
