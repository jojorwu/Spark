use ash::vk;
use crate::vulkan::device::VulkanDevice;

pub struct HiZPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pyramid_view: vk::ImageView,
    pub pyramid_image: vk::Image,
    pub pyramid_memory: vk::DeviceMemory,
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
    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;

        let prev_frame = (current_frame + crate::MAX_FRAMES_IN_FLIGHT - 1) % crate::MAX_FRAMES_IN_FLIGHT;
        self.record_commands_impl(
            &renderer.device.device,
            command_buffer,
            renderer.gbuffer.depth[prev_frame].view,
            renderer.common_sampler,
        );
    }

    fn get_resource_view(&self, name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        if name == "pyramid" {
            Some(self.pyramid_view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &Renderer) {
        unsafe {
            let device = &renderer.device.device;
            for view in &self.mip_views {
                device.destroy_image_view(*view, None);
            }
            device.destroy_image_view(self.pyramid_view, None);
            device.destroy_image(self.pyramid_image, None);
            device.free_memory(self.pyramid_memory, None);
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
    ) -> Self {
        let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

        let (image, memory) = device.create_image(
            &crate::vulkan::device::ImageCreateParams {
                width,
                height,
                mip_levels,
                format: vk::Format::R32_SFLOAT,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_DST,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            }
        );

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
            ).unwrap()
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
            ).unwrap()
        };

        let shader_module = unsafe {
            device.device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(shader_code),
                None,
            ).unwrap()
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
            ).unwrap()[0]
        };

        unsafe { device.device.destroy_shader_module(shader_module, None); }

        let mut descriptor_sets = Vec::new();
        if mip_levels > 1 {
            let layouts = vec![descriptor_set_layout; (mip_levels - 1) as usize];
            descriptor_sets = unsafe {
                device.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&layouts),
                ).unwrap()
            };
        }

        Self {
            pipeline,
            layout,
            descriptor_set_layout,
            pyramid_view,
            pyramid_image: image,
            pyramid_memory: memory,
            mip_views,
            descriptor_sets,
            width,
            height,
            mip_levels,
        }
    }

    pub fn record_commands_impl(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        depth_view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
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

                let writes = [
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[i as usize])
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&img_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[i as usize])
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&dst_info),
                ];

                device.update_descriptor_sets(&writes, &[]);

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    0,
                    &[self.descriptor_sets[i as usize]],
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
