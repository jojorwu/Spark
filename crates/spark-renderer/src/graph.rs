use crate::passes::{RenderContext, RenderPass};
use ash::vk;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    Image,
    Buffer,
    AccelerationStructure,
}

#[derive(Debug, Clone)]
pub struct ResourceDescription {
    pub name: String,
    pub ty: ResourceType,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
}

pub struct RenderGraphResource {
    pub desc: ResourceDescription,
    pub current_layout: vk::ImageLayout,
}

pub struct RenderGraphPassNode {
    pub pass: Box<dyn RenderPass>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

/// Граф рендеринга, управляющий зависимостями между проходами и ресурсами.
/// The RenderGraph manages dependencies between rendering passes and GPU resources.
pub struct RenderGraph {
    pub passes: Vec<RenderGraphPassNode>,
    pub resources: HashMap<String, RenderGraphResource>,
    pub sorted_passes: Vec<usize>,
    /// Постоянные вложения (например, G-Buffer), существующие на протяжении всей работы движка.
    pub physical_attachments: HashMap<String, Vec<crate::resource::Attachment>>,
    /// Временные вложения, создаваемые для нужд конкретных проходов.
    pub transient_attachments: HashMap<String, Vec<crate::resource::Attachment>>,
    /// Карта алиасов ресурсов (имя_ресурса -> имя_физического_ресурса).
    /// Позволяет переиспользовать память для ресурсов с неперекрывающимся временем жизни.
    pub aliased_resources: HashMap<String, String>, // resource name -> backing resource name
    pub descriptor_pool: vk::DescriptorPool,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: HashMap::new(),
            sorted_passes: Vec::new(),
            physical_attachments: HashMap::new(),
            transient_attachments: HashMap::new(),
            aliased_resources: HashMap::new(),
            descriptor_pool: vk::DescriptorPool::null(),
        }
    }

    pub fn add_pass<P: RenderPass + 'static>(
        &mut self,
        pass: P,
        inputs: &[&str],
        outputs: &[&str],
    ) {
        self.passes.push(RenderGraphPassNode {
            pass: Box::new(pass),
            inputs: inputs.iter().map(|&s| s.to_string()).collect(),
            outputs: outputs.iter().map(|&s| s.to_string()).collect(),
            descriptor_sets: Vec::new(),
        });
    }

    pub fn compile(&mut self, renderer: &mut crate::Renderer) {
        if self.descriptor_pool == vk::DescriptorPool::null() {
            let sizes = [
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(100),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(100),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(100),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(100),
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(100),
            ];
            self.descriptor_pool = unsafe {
                renderer
                    .device
                    .device
                    .create_descriptor_pool(
                        &vk::DescriptorPoolCreateInfo::default()
                            .pool_sizes(&sizes)
                            .max_sets(100),
                        None,
                    )
                    .unwrap()
            };
        }

        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut order = Vec::new();

        let pass_count = self.passes.len();

        // Build a map from output resource to the pass that produces it
        let mut resource_producers = HashMap::new();
        for (i, pass) in self.passes.iter().enumerate() {
            for output in &pass.outputs {
                resource_producers.insert(output.clone(), i);
            }
        }

        let name_to_idx: HashMap<String, usize> = self
            .passes
            .iter()
            .enumerate()
            .map(|(i, p)| (p.pass.name().to_string(), i))
            .collect();

        fn visit(
            idx: usize,
            passes: &[RenderGraphPassNode],
            resource_producers: &HashMap<String, usize>,
            name_to_idx: &HashMap<String, usize>,
            order: &mut Vec<usize>,
            visited: &mut HashSet<usize>,
            temp_visited: &mut HashSet<usize>,
        ) {
            if temp_visited.contains(&idx) {
                panic!("Circular dependency detected in RenderGraph!");
            }
            if !visited.contains(&idx) {
                temp_visited.insert(idx);

                // For each input resource, find its producer and visit it
                for input in &passes[idx].inputs {
                    if let Some(&producer_idx) = resource_producers.get(input) {
                        visit(
                            producer_idx,
                            passes,
                            resource_producers,
                            name_to_idx,
                            order,
                            visited,
                            temp_visited,
                        );
                    }
                }

                // Also respect explicit dependencies defined in RenderPass trait
                for dep in passes[idx].pass.dependencies() {
                    if let Some(&dep_idx) = name_to_idx.get(dep) {
                        visit(
                            dep_idx,
                            passes,
                            resource_producers,
                            name_to_idx,
                            order,
                            visited,
                            temp_visited,
                        );
                    }
                }

                temp_visited.remove(&idx);
                visited.insert(idx);
                order.push(idx);
            }
        }

        for i in 0..pass_count {
            visit(
                i,
                &self.passes,
                &resource_producers,
                &name_to_idx,
                &mut order,
                &mut visited,
                &mut temp_visited,
            );
        }

        self.sorted_passes = order;

        // 2. Распределение дескрипторов для проходов.
        // Descriptor set allocation for passes.
        for pass_node in &mut self.passes {
            let layout = pass_node.pass.descriptor_set_layout();
            let bindings = pass_node.pass.bindings();

            if layout == vk::DescriptorSetLayout::null() && bindings.is_empty() {
                continue;
            }

            // If the pass provides bindings but no layout, we might want to auto-generate a layout.
            // For now, assume passes provide their own layouts if they have bindings.

            if layout != vk::DescriptorSetLayout::null() {
                let layouts = [layout; crate::MAX_FRAMES_IN_FLIGHT];
                let ds = unsafe {
                    renderer
                        .device
                        .device
                        .allocate_descriptor_sets(
                            &vk::DescriptorSetAllocateInfo::default()
                                .descriptor_pool(self.descriptor_pool)
                                .set_layouts(&layouts),
                        )
                        .unwrap()
                };
                pass_node.descriptor_sets = ds.clone();
                pass_node.pass.set_descriptor_sets(ds);
            }
        }

        // 3. Улучшенное управление ресурсами: Алиасинг ресурсов.
        // Improved resource management: Resource Aliasing based on lifetime analysis.
        let extent = renderer.get_extent();

        // 1. Collect all declared resources from passes
        let mut declared_resources: HashMap<String, crate::passes::ResourceDesc> = HashMap::new();
        for pass_node in &self.passes {
            declared_resources.extend(pass_node.pass.declared_resources());
        }

        let mut resource_lifetimes: HashMap<String, (usize, usize)> = HashMap::new();

        for (order_idx, &pass_idx) in self.sorted_passes.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for input in &pass.inputs {
                resource_lifetimes
                    .entry(input.clone())
                    .and_modify(|lt| lt.1 = order_idx)
                    .or_insert((order_idx, order_idx));
            }
            for output in &pass.outputs {
                resource_lifetimes
                    .entry(output.clone())
                    .and_modify(|lt| lt.1 = order_idx)
                    .or_insert((order_idx, order_idx));
            }
        }

        self.aliased_resources.clear();
        let mut transient_pool: Vec<(String, vk::Format, vk::ImageUsageFlags, u32, u32, usize)> =
            Vec::new();

        let sorted_resource_names: Vec<String> = {
            let mut names: Vec<_> = resource_lifetimes.keys().cloned().collect();
            names.sort_by_key(|n| resource_lifetimes[n].0);
            names
        };

        for res_name in sorted_resource_names {
            if self.physical_attachments.contains_key(&res_name) {
                continue;
            }

            let (start, end) = resource_lifetimes[&res_name];

            let (format, usage, width, height) = if let Some(desc) = declared_resources.get(&res_name) {
                match desc {
                    crate::passes::ResourceDesc::Image(img) => {
                        let (w, h) = match img.size {
                            crate::passes::AttachmentSize::Absolute(w, h) => (w, h),
                            crate::passes::AttachmentSize::Relative(wf, hf) => {
                                ((extent.width as f32 * wf) as u32, (extent.height as f32 * hf) as u32)
                            }
                        };
                        (img.format, img.usage, w, h)
                    }
                    _ => continue, // Buffers not yet supported in aliasing
                }
            } else {
                let format = if res_name.contains("Depth") {
                    renderer.device.depth_format
                } else if res_name.contains("Normal")
                    || res_name.contains("HDR")
                    || res_name.contains("RTOutput")
                {
                    vk::Format::R16G16B16A16_SFLOAT
                } else {
                    vk::Format::R8G8B8A8_UNORM
                };

                let usage = if res_name.contains("Depth") {
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                } else {
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::STORAGE
                };
                (format, usage, extent.width, extent.height)
            };

            // Try to find an existing transient resource that can be reused
            let mut found_alias = None;
            for (pool_res_name, pool_format, pool_usage, pool_width, pool_height, pool_end) in
                &mut transient_pool
            {
                // For now, only alias if dimensions also match (simple approach)
                // In a perfect world, we'd check if the pooled resource is LARGER than requested.
                if *pool_format == format
                    && *pool_usage == usage
                    && *pool_width == width
                    && *pool_height == height
                    && *pool_end < start
                {
                    found_alias = Some(pool_res_name.clone());
                    *pool_end = end;
                    break;
                }
            }

            if let Some(alias) = found_alias {
                self.aliased_resources.insert(res_name.clone(), alias);
            } else {
                let attachments = (0..crate::MAX_FRAMES_IN_FLIGHT)
                    .map(|_| {
                        crate::resource::Attachment::create_image_resource(
                            &renderer.device,
                            width,
                            height,
                            format,
                            usage,
                            vk::SampleCountFlags::TYPE_1,
                        )
                        .unwrap()
                    })
                    .collect();
                self.transient_attachments
                    .insert(res_name.clone(), attachments);
                transient_pool.push((res_name.clone(), format, usage, width, height, end));
            }
        }
    }

    pub fn execute(&self, ctx: &RenderContext, secondary_commands: &[Vec<vk::CommandBuffer>]) {
        let renderer = ctx.renderer;

        // 2. Execution
        for &idx in &self.sorted_passes {
            let pass_node = &self.passes[idx];

            // 1. Automatic Descriptor Updates
            // Only update if resources have actually changed (handled inside update_descriptor_sets or similar logic)
            // For now, we perform automated binding matching.
            if !pass_node.descriptor_sets.is_empty() {
                let ds = pass_node.descriptor_sets[ctx.current_frame];
                let bindings = pass_node.pass.bindings();

                let mut img_infos = Vec::new();
                let mut buf_infos = Vec::new();
                let mut as_infos = Vec::new();
                let mut as_handles = Vec::new();

                for binding in &bindings {
                    match binding {
                        crate::passes::ResourceBinding::SampledImage(_, name) |
                        crate::passes::ResourceBinding::InputAttachment(_, name) => {
                            let view = renderer
                                .get_pass_resource_view(pass_node.pass.name(), name, ctx.current_frame)
                                .unwrap_or(renderer.common_shadow_view);
                            img_infos.push(vk::DescriptorImageInfo::default()
                                .image_layout(if matches!(binding, crate::passes::ResourceBinding::InputAttachment(_, _)) {
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL // Actually usually SHADER_READ_ONLY_OPTIMAL or COLOR_ATTACHMENT_OPTIMAL for inputs
                                } else {
                                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                                })
                                .image_view(view)
                                .sampler(renderer.common_sampler));
                        }
                        crate::passes::ResourceBinding::StorageImage(_, name) => {
                            let view = renderer
                                .get_pass_resource_view(pass_node.pass.name(), name, ctx.current_frame)
                                .unwrap_or(renderer.common_shadow_view);
                            img_infos.push(vk::DescriptorImageInfo::default()
                                .image_layout(vk::ImageLayout::GENERAL)
                                .image_view(view));
                        }
                        crate::passes::ResourceBinding::StorageBuffer(_, name) |
                        crate::passes::ResourceBinding::UniformBuffer(_, name) => {
                            if let Some(buffer) = renderer.get_resource_buffer(pass_node.pass.name(), name, ctx.current_frame) {
                                buf_infos.push(vk::DescriptorBufferInfo::default()
                                    .buffer(buffer.handle)
                                    .range(buffer.size));
                            } else {
                                // Push dummy if not found to keep indices matching?
                                // Better to just filter them out later.
                            }
                        }
                        crate::passes::ResourceBinding::AccelerationStructure(_, _) => {
                            let as_manager = renderer.as_manager.lock().unwrap();
                            if let Some(ref tlas) = as_manager.current_tlas[ctx.current_frame] {
                                as_handles.push(tlas.handle);
                            }
                        }
                    }
                }

                let mut writes = Vec::new();
                let mut img_idx = 0;
                let mut buf_idx = 0;
                let mut as_idx = 0;

                for binding in bindings {
                    match binding {
                        crate::passes::ResourceBinding::SampledImage(binding_idx, _) => {
                            writes.push(vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding_idx)
                                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                                .descriptor_count(1)
                                .image_info(std::slice::from_ref(&img_infos[img_idx])));
                            img_idx += 1;
                        }
                        crate::passes::ResourceBinding::InputAttachment(binding_idx, _) => {
                            writes.push(vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding_idx)
                                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                                .descriptor_count(1)
                                .image_info(std::slice::from_ref(&img_infos[img_idx])));
                            img_idx += 1;
                        }
                        crate::passes::ResourceBinding::StorageImage(binding_idx, _) => {
                            writes.push(vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding_idx)
                                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                .descriptor_count(1)
                                .image_info(std::slice::from_ref(&img_infos[img_idx])));
                            img_idx += 1;
                        }
                        crate::passes::ResourceBinding::StorageBuffer(binding_idx, name) => {
                            if renderer.get_resource_buffer(pass_node.pass.name(), &name, ctx.current_frame).is_some() {
                                writes.push(vk::WriteDescriptorSet::default()
                                    .dst_set(ds)
                                    .dst_binding(binding_idx)
                                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                    .descriptor_count(1)
                                    .buffer_info(std::slice::from_ref(&buf_infos[buf_idx])));
                                buf_idx += 1;
                            }
                        }
                        crate::passes::ResourceBinding::UniformBuffer(binding_idx, name) => {
                            if renderer.get_resource_buffer(pass_node.pass.name(), &name, ctx.current_frame).is_some() {
                                writes.push(vk::WriteDescriptorSet::default()
                                    .dst_set(ds)
                                    .dst_binding(binding_idx)
                                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                                    .descriptor_count(1)
                                    .buffer_info(std::slice::from_ref(&buf_infos[buf_idx])));
                                buf_idx += 1;
                            }
                        }
                        crate::passes::ResourceBinding::AccelerationStructure(_binding_idx, _) => {
                            if as_idx < as_handles.len() {
                                let info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                                    .acceleration_structures(std::slice::from_ref(&as_handles[as_idx]));
                                as_infos.push(info);
                            }
                            as_idx += 1;
                        }
                    }
                }

                // Second pass for AS because of push_next lifetime issues
                // We need to do the update per AS because of the &mut borrow in push_next
                if !writes.is_empty() {
                    unsafe {
                        renderer.device.device.update_descriptor_sets(&writes, &[]);
                    }
                }

                let mut as_idx_2 = 0;
                for binding in pass_node.pass.bindings() {
                    if let crate::passes::ResourceBinding::AccelerationStructure(binding_idx, _) = binding {
                        if as_idx_2 < as_infos.len() {
                             let as_write = [vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(binding_idx)
                                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                                .descriptor_count(1)
                                .push_next(&mut as_infos[as_idx_2])];
                             unsafe {
                                 renderer.device.device.update_descriptor_sets(&as_write, &[]);
                             }
                             as_idx_2 += 1;
                        }
                    }
                }
            }

            // 2. Automated Image Barrier Injection
            for (mut res_name, dst_access, dst_stage) in pass_node.pass.gpu_resource_access() {
                if let Some(alias) = self.aliased_resources.get(&res_name) {
                    res_name = alias.clone();
                }

                let attachments = self
                    .physical_attachments
                    .get(&res_name)
                    .or_else(|| self.transient_attachments.get(&res_name));

                if let Some(attachments) = attachments {
                    let attachment = &attachments[ctx.current_frame];
                    let aspect = if res_name.contains("Depth") {
                        vk::ImageAspectFlags::DEPTH
                    } else {
                        vk::ImageAspectFlags::COLOR
                    };

                    let new_layout = if dst_access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::SHADER_READ)
                        && !dst_stage.contains(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR)
                    {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::TRANSFER_READ) {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::TRANSFER_WRITE) {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    } else {
                        vk::ImageLayout::GENERAL
                    };

                    renderer.resource_tracker.transition_image(
                        ctx.command_buffer,
                        &renderer.device.device,
                        attachment.image,
                        new_layout,
                        vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
                        dst_access,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        dst_stage,
                        aspect,
                    );
                }
            }

            if !secondary_commands[idx].is_empty() {
                unsafe {
                    renderer
                        .device
                        .device
                        .cmd_execute_commands(ctx.command_buffer, &secondary_commands[idx]);
                }
            }

            // 3. Automated Buffer Barrier Injection
            for (res_name, dst_access, dst_stage) in pass_node.pass.gpu_resource_buffer_access() {
                if let Some(buffer) = renderer.get_resource_buffer(pass_node.pass.name(), &res_name, ctx.current_frame) {
                    let barrier = vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
                        .dst_stage_mask(vk::PipelineStageFlags2::from_raw(dst_stage.as_raw() as u64))
                        .dst_access_mask(vk::AccessFlags2::from_raw(dst_access.as_raw() as u64))
                        .buffer(buffer.handle)
                        .offset(0)
                        .size(buffer.size);

                    let dependency_info = vk::DependencyInfo::default()
                        .buffer_memory_barriers(std::slice::from_ref(&barrier));

                    unsafe {
                        renderer.device.device.cmd_pipeline_barrier2(ctx.command_buffer, &dependency_info);
                    }
                }
            }

            pass_node.pass.record_commands(ctx);
        }
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}
