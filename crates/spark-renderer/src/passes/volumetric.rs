use ash::vk;
use crate::resource::Attachment;
use crate::Renderer;

pub struct VolumetricPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_images: Vec<Attachment>,
}

use super::{RenderPass, RenderContext};

impl RenderPass for VolumetricPass {
    fn name(&self) -> &str { "VolumetricPass" }
    fn is_enabled(&self, renderer: &Renderer) -> bool { renderer.enable_volumetric }
    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let out_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.output_images[current_frame].view)];
        let depth_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.gbuffer_depth[current_frame].view).sampler(renderer.common_sampler)];
        let shadow_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.common_shadow_view).sampler(renderer.common_sampler)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&shadow_info),
        ];
        unsafe { renderer.device.device.update_descriptor_sets(&writes, &[]); }
    }

    fn update_descriptor_sets(&self, _renderer: &Renderer) {
        // Handled in prepare
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;

        let global_ds = renderer.frames[current_frame].global_descriptor_set;
        self.record_commands_impl(&renderer.device.device, command_buffer, current_frame, global_ds, renderer.swapchain.extent);
    }

    fn get_resource_view(&self, name: &str, frame_index: usize) -> Option<vk::ImageView> {
        if name == "output" {
            Some(self.output_images[frame_index].view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            for a in self.output_images.drain(..) { a.destroy(device, &renderer.device.allocator); }
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

impl VolumetricPass {
    pub fn new(
        renderer: &Renderer,
        shader_code: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let ds_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[renderer.global_descriptor_set_layout, ds_layout]),
                None,
            )?
        };

        let mut output_images = Vec::new();
        for _ in 0..crate::MAX_FRAMES_IN_FLIGHT {
            output_images.push(Attachment::create_image_resource(
                &renderer.device, extent.width / 2, extent.height / 2,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )?);
        }

        let shader_module = unsafe {
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(shader_code), None)?
        };
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE).module(shader_module).name(&entry);

        let pipeline = unsafe {
            device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout)], None).map_err(|e| e.1)?[0]
        };

        unsafe { device.destroy_shader_module(shader_module, None); }

        let layouts = vec![ds_layout; crate::MAX_FRAMES_IN_FLIGHT];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(renderer.descriptor_pool).set_layouts(&layouts))?
        };

        Ok(Self { pipeline, layout, descriptor_set_layout: ds_layout, descriptor_sets, output_images })
    }

    pub fn update_descriptor_sets(&self, renderer: &Renderer) {
        for i in 0..crate::MAX_FRAMES_IN_FLIGHT {
            let out_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.output_images[i].view)];
            let depth_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.gbuffer_depth[i].view).sampler(renderer.common_sampler)];
            let shadow_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.common_shadow_view).sampler(renderer.common_sampler)];

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&shadow_info),
            ];
            unsafe { renderer.device.device.update_descriptor_sets(&writes, &[]); }
        }
    }

    pub fn record_commands_impl(&self, device: &ash::Device, cb: vk::CommandBuffer, current_frame: usize, global_ds: vk::DescriptorSet, extent: vk::Extent2D) {
        unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .image(self.output_images[current_frame].image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier]);

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[global_ds, self.descriptor_sets[current_frame]], &[]);
            device.cmd_dispatch(cb, (extent.width / 2).div_ceil(8), (extent.height / 2).div_ceil(8), 1);

            let barrier_read = vk::ImageMemoryBarrier::default()
                .image(self.output_images[current_frame].image)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 })
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            device.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[barrier_read]);
        }
    }

}
