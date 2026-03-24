use ash::vk;
use crate::resource::Attachment;
use crate::MAX_FRAMES_IN_FLIGHT;
use crate::Renderer;
use super::{RenderPass, RenderContext};

pub struct SSRPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_images: Vec<Attachment>,
}

impl RenderPass for SSRPass {
    fn name(&self) -> &str { "SSRPass" }
    fn inputs(&self) -> Vec<&'static str> { vec!["GBuffer", "HDRColor", "HiZ"] }
    fn outputs(&self) -> Vec<&'static str> { vec!["SSR"] }

    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let device = &renderer.device.device;
        let sampler = renderer.common_sampler;

        let hiz_view = renderer.get_pass_resource_view("HiZPass", "pyramid", current_frame).unwrap_or(renderer.common_shadow_view);
        let hdr_view = renderer.get_pass_resource_view("LightingPass", "HDRColor", current_frame).unwrap_or(renderer.common_shadow_view);

        let img_infos = [
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.get_pass_resource_view("", "GBufferAlbedo", current_frame).unwrap_or(renderer.common_shadow_view)).sampler(sampler),
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.get_pass_resource_view("", "GBufferNormal", current_frame).unwrap_or(renderer.common_shadow_view)).sampler(sampler),
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.get_pass_resource_view("", "GBufferPBR", current_frame).unwrap_or(renderer.common_shadow_view)).sampler(sampler),
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.get_pass_resource_view("", "GBufferDepth", current_frame).unwrap_or(renderer.common_shadow_view)).sampler(sampler),
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(hdr_view).sampler(sampler),
            vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(hiz_view).sampler(sampler),
        ];

        let out_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::GENERAL).image_view(self.output_images[current_frame].view)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[0..1]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[1..2]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[2..3]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[3..4]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[4..5]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(5).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&img_infos[5..6]),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(6).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).image_info(&out_info),
        ];

        unsafe { device.update_descriptor_sets(&writes, &[]); }
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let device = &ctx.renderer.device.device;
        let extent = ctx.renderer.get_extent();

        unsafe {
            device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(ctx.command_buffer, vk::PipelineBindPoint::COMPUTE, self.layout, 0, &[ctx.renderer.frames[ctx.current_frame].global_descriptor_set, self.descriptor_sets[ctx.current_frame]], &[]);

            let pc = [extent.width, extent.height, 20.0f32.to_bits(), 0.5f32.to_bits(), 0.1f32.to_bits(), 64u32];
            let pc_bytes = std::slice::from_raw_parts(pc.as_ptr() as *const u8, 24);
            device.cmd_push_constants(ctx.command_buffer, self.layout, vk::ShaderStageFlags::COMPUTE, 0, pc_bytes);

            device.cmd_dispatch(ctx.command_buffer, (extent.width + 15) / 16, (extent.height + 15) / 16, 1);
        }
    }

    fn get_resource_view(&self, name: &str, frame_index: usize) -> Option<vk::ImageView> {
        if name == "output" { Some(self.output_images[frame_index].view) } else { None }
    }

    fn on_resize(&mut self, renderer: &mut Renderer, new_extent: vk::Extent2D) {
        for img in self.output_images.drain(..) { img.destroy(&renderer.device.device, &renderer.device.allocator); }
        self.output_images = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            Attachment::create_image_resource(&renderer.device, new_extent.width, new_extent.height, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED, vk::SampleCountFlags::TYPE_1).unwrap()
        }).collect();
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.output_images.drain(..) { img.destroy(device, &renderer.device.allocator); }
        }
    }
}

impl SSRPass {
    pub fn new(renderer: &Renderer, shader_spirv: &[u32]) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        let output_images = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            Attachment::create_image_resource(&renderer.device, extent.width, extent.height, vk::Format::R16G16B16A16_SFLOAT, vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED, vk::SampleCountFlags::TYPE_1).unwrap()
        }).collect::<Vec<_>>();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(5).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default().binding(6).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];

        let ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[renderer.global_descriptor_set_layout, ds_layout]).push_constant_ranges(&[vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 24 }]), None)? };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(renderer.descriptor_pool).set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]))?
        };

        let module = crate::pipeline::Pipeline::create_shader_module(device, shader_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry_point);

        let pipeline = unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(layout)], None).unwrap()[0] };

        unsafe { device.destroy_shader_module(module, None); }

        Ok(Self { pipeline, layout, descriptor_set_layout: ds_layout, descriptor_sets, output_images })
    }
}
