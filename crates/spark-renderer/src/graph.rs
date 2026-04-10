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
    pub resource_versions: Vec<std::sync::Mutex<HashMap<String, u64>>>,
}

/// Граф рендеринга, управляющий зависимостями между проходами и ресурсами.
/// The RenderGraph manages dependencies between rendering passes and GPU resources.
/// It performs topological sorting of passes and automatic resource lifecycle management.
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

    pub fn destroy_resources(
        &mut self,
        device: &ash::Device,
        allocator: &std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>,
    ) {
        unsafe {
            for (_, attachments) in self.transient_attachments.drain() {
                for a in attachments {
                    a.destroy(device, allocator);
                }
            }

            if self.descriptor_pool != vk::DescriptorPool::null() {
                device.destroy_descriptor_pool(self.descriptor_pool, None);
                self.descriptor_pool = vk::DescriptorPool::null();
            }
        }
    }

    pub fn add_pass<P: RenderPass + 'static>(
        &mut self,
        pass: P,
        inputs: &[&str],
        outputs: &[&str],
    ) {
        let mut resource_versions = Vec::with_capacity(crate::MAX_FRAMES_IN_FLIGHT);
        for _ in 0..crate::MAX_FRAMES_IN_FLIGHT {
            resource_versions.push(std::sync::Mutex::new(HashMap::new()));
        }

        self.passes.push(RenderGraphPassNode {
            pass: Box::new(pass),
            inputs: inputs.iter().map(|&s| s.to_string()).collect(),
            outputs: outputs.iter().map(|&s| s.to_string()).collect(),
            descriptor_sets: Vec::new(),
            resource_versions,
        });
    }

    pub fn compile(&mut self, renderer: &mut crate::Renderer) {
        log::debug!("Compiling RenderGraph with {} passes", self.passes.len());
        self.ensure_descriptor_pool(renderer);
        self.sort_passes();
        self.allocate_pass_descriptors(renderer);
        self.analyze_resources_and_alias(renderer);
    }

    fn ensure_descriptor_pool(&mut self, renderer: &crate::Renderer) {
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
                    .expect("Failed to create RenderGraph descriptor pool")
            };
        }
    }

    fn sort_passes(&mut self) {
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();
        let mut order = Vec::new();

        let pass_count = self.passes.len();

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

        for i in 0..pass_count {
            Self::visit_pass(
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
    }

    fn visit_pass(
        idx: usize,
        passes: &[RenderGraphPassNode],
        resource_producers: &HashMap<String, usize>,
        name_to_idx: &HashMap<String, usize>,
        order: &mut Vec<usize>,
        visited: &mut HashSet<usize>,
        temp_visited: &mut HashSet<usize>,
    ) {
        if temp_visited.contains(&idx) {
            panic!(
                "Circular dependency detected in RenderGraph! Pass: {}",
                passes[idx].pass.name()
            );
        }
        if !visited.contains(&idx) {
            temp_visited.insert(idx);

            for input in &passes[idx].inputs {
                if let Some(&producer_idx) = resource_producers.get(input) {
                    Self::visit_pass(
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

            for dep in passes[idx].pass.dependencies() {
                if let Some(&dep_idx) = name_to_idx.get(dep) {
                    Self::visit_pass(
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

    fn allocate_pass_descriptors(&mut self, renderer: &crate::Renderer) {
        for pass_node in &mut self.passes {
            let layout = pass_node.pass.descriptor_set_layout();
            let bindings = pass_node.pass.bindings();

            if layout == vk::DescriptorSetLayout::null() && bindings.is_empty() {
                continue;
            }

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
                        .expect("Failed to allocate descriptor sets for RenderPass")
                };
                pass_node.descriptor_sets = ds.clone();
                pass_node.pass.set_descriptor_sets(ds);
            }
        }
    }

    fn analyze_resources_and_alias(&mut self, renderer: &crate::Renderer) {
        let extent = renderer.get_extent();
        let declared_resources = self.collect_declared_resources();
        let resource_lifetimes = self.calculate_resource_lifetimes();

        self.aliased_resources.clear();
        let mut transient_pool: Vec<(String, vk::Format, vk::ImageUsageFlags, u32, u32, usize)> =
            Vec::new();

        let mut sorted_resource_names: Vec<String> = resource_lifetimes.keys().cloned().collect();
        sorted_resource_names.sort_by_key(|n| resource_lifetimes[n].0);

        for res_name in sorted_resource_names {
            if self.physical_attachments.contains_key(&res_name) {
                continue;
            }

            let (start, end) = resource_lifetimes[&res_name];
            let (format, usage, width, height) = self.resolve_resource_specs(
                &res_name,
                &declared_resources,
                extent,
                renderer.device.depth_format,
            );

            if let Some(alias) =
                self.find_compatible_alias(&transient_pool, format, usage, width, height, start)
            {
                self.aliased_resources
                    .insert(res_name.clone(), alias.clone());
                // Update the end lifetime of the alias in the pool
                if let Some(pool_entry) = transient_pool.iter_mut().find(|e| e.0 == alias) {
                    pool_entry.5 = end;
                }
            } else {
                let attachments =
                    self.create_transient_attachments(renderer, width, height, format, usage);
                self.transient_attachments
                    .insert(res_name.clone(), attachments);
                transient_pool.push((res_name.clone(), format, usage, width, height, end));
            }
        }
    }

    fn collect_declared_resources(&self) -> HashMap<&'static str, crate::passes::ResourceDesc> {
        let mut declared_resources = HashMap::new();
        for pass_node in &self.passes {
            declared_resources.extend(pass_node.pass.declared_resources());
        }
        declared_resources
    }

    fn calculate_resource_lifetimes(&self) -> HashMap<String, (usize, usize)> {
        let mut resource_lifetimes: HashMap<String, (usize, usize)> = HashMap::new();
        for (order_idx, &pass_idx) in self.sorted_passes.iter().enumerate() {
            let pass = &self.passes[pass_idx];
            for name in pass.inputs.iter().chain(pass.outputs.iter()) {
                resource_lifetimes
                    .entry(name.clone())
                    .and_modify(|lt| lt.1 = order_idx)
                    .or_insert((order_idx, order_idx));
            }
        }
        resource_lifetimes
    }

    fn resolve_resource_specs(
        &self,
        name: &str,
        declared: &HashMap<&'static str, crate::passes::ResourceDesc>,
        extent: vk::Extent2D,
        depth_format: vk::Format,
    ) -> (vk::Format, vk::ImageUsageFlags, u32, u32) {
        if let Some(crate::passes::ResourceDesc::Image(img)) = declared.get(name) {
            let (w, h) = match img.size {
                crate::passes::AttachmentSize::Absolute(w, h) => (w, h),
                crate::passes::AttachmentSize::Relative(wf, hf) => (
                    (extent.width as f32 * wf) as u32,
                    (extent.height as f32 * hf) as u32,
                ),
            };
            return (img.format, img.usage, w, h);
        }

        // Heuristic defaults
        let format = if name.contains("Depth") {
            depth_format
        } else if name.contains("Normal") || name.contains("HDR") || name.contains("RTOutput") {
            vk::Format::R16G16B16A16_SFLOAT
        } else {
            vk::Format::R8G8B8A8_UNORM
        };

        let usage = if name.contains("Depth") {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE
        };

        (format, usage, extent.width, extent.height)
    }

    fn find_compatible_alias(
        &self,
        pool: &[(String, vk::Format, vk::ImageUsageFlags, u32, u32, usize)],
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        width: u32,
        height: u32,
        start: usize,
    ) -> Option<String> {
        for (pool_res_name, pool_format, pool_usage, pool_width, pool_height, pool_end) in pool {
            if *pool_format == format
                && *pool_usage == usage
                && *pool_width == width
                && *pool_height == height
                && *pool_end < start
            {
                return Some(pool_res_name.clone());
            }
        }
        None
    }

    fn create_transient_attachments(
        &self,
        renderer: &crate::Renderer,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Vec<crate::resource::Attachment> {
        (0..crate::MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                crate::resource::Attachment::create_image_resource(
                    &renderer.device,
                    width,
                    height,
                    format,
                    usage,
                    vk::SampleCountFlags::TYPE_1,
                )
                .expect("Failed to create transient attachment in RenderGraph")
            })
            .collect()
    }

    /// Выполняет граф рендеринга, проходя по отсортированным проходам.
    /// Автоматически обрабатывает обновление дескрипторов и инъекцию барьеров памяти.
    ///
    /// Executes the render graph by iterating through sorted passes.
    /// Automatically handles descriptor updates and memory barrier injection.
    pub fn execute(&self, ctx: &RenderContext, _secondary_commands: &[Vec<vk::CommandBuffer>]) {
        for &idx in &self.sorted_passes {
            let pass_node = &self.passes[idx];

            if !pass_node.pass.is_enabled(ctx.renderer) {
                continue;
            }

            self.execute_pass(ctx, pass_node);
        }
    }

    fn execute_pass(&self, ctx: &RenderContext, pass_node: &RenderGraphPassNode) {
        // 1. Update descriptor sets for the pass if any of its bound resources have changed.
        self.update_pass_descriptors(ctx, pass_node);

        // 2. Inject memory barriers based on declarative access requirements.
        self.inject_barriers(ctx, pass_node);

        // 3. Record primary commands for the pass.
        pass_node.pass.record_commands(ctx);
    }

    fn inject_barriers(&self, ctx: &RenderContext, pass_node: &RenderGraphPassNode) {
        self.inject_image_barriers(ctx, pass_node);
        self.inject_buffer_barriers(ctx, pass_node);
    }

    fn update_pass_descriptors(&self, ctx: &RenderContext, pass_node: &RenderGraphPassNode) {
        let renderer = ctx.renderer;
        if pass_node.descriptor_sets.is_empty() {
            return;
        }

        let ds = pass_node.descriptor_sets[ctx.current_frame];
        let bindings = pass_node.pass.bindings();

        let mut needs_update = false;
        let mut current_versions = HashMap::new();

        for binding in &bindings {
            let name = match binding {
                crate::passes::ResourceBinding::SampledImage(_, n) => *n,
                crate::passes::ResourceBinding::InputAttachment(_, n) => *n,
                crate::passes::ResourceBinding::StorageImage(_, n) => *n,
                crate::passes::ResourceBinding::StorageBuffer(_, n) => *n,
                crate::passes::ResourceBinding::UniformBuffer(_, n) => *n,
                crate::passes::ResourceBinding::AccelerationStructure(_, n) => *n,
            };

            let version = if let Some(attachments) = self
                .physical_attachments
                .get(name)
                .or_else(|| self.transient_attachments.get(name))
            {
                attachments[ctx.current_frame]
                    .version
                    .load(std::sync::atomic::Ordering::Relaxed)
            } else if let Some(buffer) =
                renderer.get_resource_buffer(pass_node.pass.name(), name, ctx.current_frame)
            {
                buffer.version.load(std::sync::atomic::Ordering::Relaxed)
            } else {
                0
            };

            current_versions.insert(name.to_string(), version);
            let versions = pass_node.resource_versions[ctx.current_frame]
                .lock()
                .expect("Failed to lock resource versions");
            if versions.get(name) != Some(&version) {
                needs_update = true;
            }
        }

        if needs_update {
            *pass_node.resource_versions[ctx.current_frame]
                .lock()
                .expect("Failed to lock resource versions for update") = current_versions;

            self.perform_descriptor_update(ctx, pass_node, ds, &bindings);
        }
    }

    /// Updates Vulkan descriptor sets for a specific pass using current frame resources.
    /// Uses a robust resolution process to build descriptor writes from pass requirements.
    fn perform_descriptor_update(
        &self,
        ctx: &RenderContext,
        pass_node: &RenderGraphPassNode,
        ds: vk::DescriptorSet,
        bindings: &[crate::passes::ResourceBinding],
    ) {
        let renderer = ctx.renderer;

        // Use temporary storage to maintain lifetimes for `update_descriptor_sets`.
        let mut img_infos = Vec::new();
        let mut buf_infos = Vec::new();
        let mut as_infos = Vec::new();
        let mut as_handles = Vec::new();

        // Pass 1: Resolve all resources.
        for binding in bindings {
            match binding {
                crate::passes::ResourceBinding::SampledImage(_, name)
                | crate::passes::ResourceBinding::InputAttachment(_, name)
                | crate::passes::ResourceBinding::StorageImage(_, name) => {
                    let view = renderer
                        .get_pass_resource_view(pass_node.pass.name(), name, ctx.current_frame)
                        .unwrap_or(renderer.common_shadow_view);

                    let layout = match binding {
                        crate::passes::ResourceBinding::StorageImage(_, _) => {
                            vk::ImageLayout::GENERAL
                        }
                        _ => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    };

                    img_infos.push(
                        vk::DescriptorImageInfo::default()
                            .image_layout(layout)
                            .image_view(view)
                            .sampler(renderer.common_sampler),
                    );
                }
                crate::passes::ResourceBinding::StorageBuffer(_, name)
                | crate::passes::ResourceBinding::UniformBuffer(_, name) => {
                    if let Some(buffer) =
                        renderer.get_resource_buffer(pass_node.pass.name(), name, ctx.current_frame)
                    {
                        buf_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer.handle)
                                .range(buffer.size),
                        );
                    }
                }
                crate::passes::ResourceBinding::AccelerationStructure(_, _) => {
                    let as_manager = renderer
                        .as_manager
                        .lock()
                        .expect("Failed to lock AS manager during descriptor update");
                    if let Some(ref tlas) = as_manager.current_tlas[ctx.current_frame] {
                        as_handles.push([tlas.handle]);
                    }
                }
            }
        }

        // Pass 2: Build writes.
        let mut writes = Vec::new();
        let mut img_ptr = 0;
        let mut buf_ptr = 0;
        let mut as_ptr = 0;

        for binding in bindings {
            match binding {
                crate::passes::ResourceBinding::SampledImage(binding_idx, _)
                | crate::passes::ResourceBinding::InputAttachment(binding_idx, _)
                | crate::passes::ResourceBinding::StorageImage(binding_idx, _) => {
                    let ty = match binding {
                        crate::passes::ResourceBinding::SampledImage(_, _) => {
                            vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                        }
                        crate::passes::ResourceBinding::InputAttachment(_, _) => {
                            vk::DescriptorType::INPUT_ATTACHMENT
                        }
                        crate::passes::ResourceBinding::StorageImage(_, _) => {
                            vk::DescriptorType::STORAGE_IMAGE
                        }
                        _ => unreachable!(),
                    };

                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(*binding_idx)
                            .descriptor_type(ty)
                            .descriptor_count(1)
                            .image_info(std::slice::from_ref(&img_infos[img_ptr])),
                    );
                    img_ptr += 1;
                }
                crate::passes::ResourceBinding::StorageBuffer(binding_idx, name)
                | crate::passes::ResourceBinding::UniformBuffer(binding_idx, name) => {
                    if renderer
                        .get_resource_buffer(pass_node.pass.name(), name, ctx.current_frame)
                        .is_some()
                    {
                        let ty = match binding {
                            crate::passes::ResourceBinding::StorageBuffer(_, _) => {
                                vk::DescriptorType::STORAGE_BUFFER
                            }
                            crate::passes::ResourceBinding::UniformBuffer(_, _) => {
                                vk::DescriptorType::UNIFORM_BUFFER
                            }
                            _ => unreachable!(),
                        };

                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(*binding_idx)
                                .descriptor_type(ty)
                                .descriptor_count(1)
                                .buffer_info(std::slice::from_ref(&buf_infos[buf_ptr])),
                        );
                        buf_ptr += 1;
                    }
                }
                crate::passes::ResourceBinding::AccelerationStructure(binding_idx, _) => {
                    if as_ptr < as_handles.len() {
                        as_infos.push(
                            vk::WriteDescriptorSetAccelerationStructureKHR::default()
                                .acceleration_structures(&as_handles[as_ptr]),
                        );
                        writes.push(
                            vk::WriteDescriptorSet::default()
                                .dst_set(ds)
                                .dst_binding(*binding_idx)
                                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                                .descriptor_count(1),
                        );
                        as_ptr += 1;
                    }
                }
            }
        }

        // Link Acceleration Structure infos manually using pointers to maintain borrows.
        let mut current_as_idx = 0;
        for write in &mut writes {
            if write.descriptor_type == vk::DescriptorType::ACCELERATION_STRUCTURE_KHR {
                write.p_next = &as_infos[current_as_idx] as *const _ as *const _;
                current_as_idx += 1;
            }
        }

        if !writes.is_empty() {
            unsafe {
                renderer.device.device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    fn inject_image_barriers(&self, ctx: &RenderContext, pass_node: &RenderGraphPassNode) {
        let renderer = ctx.renderer;
        for access in pass_node.pass.gpu_resource_access() {
            let mut res_name: &str = access.resource_name;
            let dst_access = access.access_flags;
            let dst_stage = access.stage_flags;

            if let Some(alias) = self.aliased_resources.get(res_name) {
                res_name = alias;
            }

            let attachments = self
                .physical_attachments
                .get(res_name)
                .or_else(|| self.transient_attachments.get(res_name));

            if let Some(attachments) = attachments {
                let attachment = &attachments[ctx.current_frame];
                let aspect = if attachment.format == vk::Format::D32_SFLOAT
                    || attachment.format == vk::Format::D32_SFLOAT_S8_UINT
                    || attachment.format == vk::Format::D24_UNORM_S8_UINT
                    || attachment.format == vk::Format::D16_UNORM
                {
                    vk::ImageAspectFlags::DEPTH
                } else {
                    vk::ImageAspectFlags::COLOR
                };

                let new_layout = self.determine_layout(dst_access, dst_stage);

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
    }

    fn determine_layout(
        &self,
        access: vk::AccessFlags,
        stage: vk::PipelineStageFlags,
    ) -> vk::ImageLayout {
        if access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) {
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        } else if access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        } else if access.contains(vk::AccessFlags::SHADER_READ)
            && !stage.contains(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR)
        {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else if access.contains(vk::AccessFlags::TRANSFER_READ) {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else if access.contains(vk::AccessFlags::TRANSFER_WRITE) {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        } else {
            vk::ImageLayout::GENERAL
        }
    }

    fn inject_buffer_barriers(&self, ctx: &RenderContext, pass_node: &RenderGraphPassNode) {
        let renderer = ctx.renderer;
        for access in pass_node.pass.gpu_resource_buffer_access() {
            let res_name = access.resource_name;
            let dst_access = access.access_flags;
            let dst_stage = access.stage_flags;

            if let Some(buffer) =
                renderer.get_resource_buffer(pass_node.pass.name(), res_name, ctx.current_frame)
            {
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
                    renderer
                        .device
                        .device
                        .cmd_pipeline_barrier2(ctx.command_buffer, &dependency_info);
                }
            }
        }
    }
}

impl Default for RenderGraph {
    fn default() -> Self {
        Self::new()
    }
}
