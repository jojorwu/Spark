use ash::vk;
use std::collections::{HashMap, HashSet};
use crate::passes::{RenderPass, RenderContext};

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
}

pub struct RenderGraph {
    pub passes: Vec<RenderGraphPassNode>,
    pub resources: HashMap<String, RenderGraphResource>,
    pub sorted_passes: Vec<usize>,
    pub physical_attachments: HashMap<String, Vec<crate::resource::Attachment>>,
    pub transient_attachments: HashMap<String, Vec<crate::resource::Attachment>>,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: HashMap::new(),
            sorted_passes: Vec::new(),
            physical_attachments: HashMap::new(),
            transient_attachments: HashMap::new(),
        }
    }

    pub fn add_pass<P: RenderPass + 'static>(&mut self, pass: P, inputs: &[&str], outputs: &[&str]) {
        self.passes.push(RenderGraphPassNode {
            pass: Box::new(pass),
            inputs: inputs.iter().map(|&s| s.to_string()).collect(),
            outputs: outputs.iter().map(|&s| s.to_string()).collect(),
        });
    }

    pub fn compile(&mut self, renderer: &mut crate::Renderer) {
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

        let name_to_idx: HashMap<String, usize> = self.passes.iter().enumerate()
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
                        visit(producer_idx, passes, resource_producers, name_to_idx, order, visited, temp_visited);
                    }
                }

                // Also respect explicit dependencies defined in RenderPass trait
                for dep in passes[idx].pass.dependencies() {
                    if let Some(&dep_idx) = name_to_idx.get(dep) {
                        visit(dep_idx, passes, resource_producers, name_to_idx, order, visited, temp_visited);
                    }
                }

                temp_visited.remove(&idx);
                visited.insert(idx);
                order.push(idx);
            }
        }

        for i in 0..pass_count {
            visit(i, &self.passes, &resource_producers, &name_to_idx, &mut order, &mut visited, &mut temp_visited);
        }

        self.sorted_passes = order;

        // Automatically create transient resources for outputs that are not physical
        let extent = renderer.get_extent();
        for pass in &self.passes {
            for output in &pass.outputs {
                if !self.physical_attachments.contains_key(output) && !self.transient_attachments.contains_key(output) {
                    let format = if output.contains("Depth") { renderer.device.depth_format } else { vk::Format::R16G16B16A16_SFLOAT };
                    let usage = if output.contains("Depth") {
                        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                    } else {
                        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE
                    };

                    let attachments = (0..crate::MAX_FRAMES_IN_FLIGHT).map(|_| {
                        crate::resource::Attachment::create_image_resource(
                            &renderer.device,
                            extent.width,
                            extent.height,
                            format,
                            usage,
                            vk::SampleCountFlags::TYPE_1,
                        ).unwrap()
                    }).collect();
                    self.transient_attachments.insert(output.clone(), attachments);
                }
            }
        }
    }

    pub fn execute(&self, ctx: &RenderContext, secondary_commands: &[Vec<vk::CommandBuffer>]) {
        let renderer = ctx.renderer;
        // 2. Execution
        for &idx in &self.sorted_passes {
            let pass_node = &self.passes[idx];

            // Automated Barrier Injection
            for (res_name, dst_access, dst_stage) in pass_node.pass.gpu_resource_access() {
                let attachments = self.physical_attachments.get(&res_name)
                    .or_else(|| self.transient_attachments.get(&res_name));

                if let Some(attachments) = attachments {
                    let attachment = &attachments[ctx.current_frame];
                    let aspect = if res_name.contains("Depth") { vk::ImageAspectFlags::DEPTH } else { vk::ImageAspectFlags::COLOR };

                    let new_layout = if dst_access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::SHADER_READ) && !dst_stage.contains(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR) {
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::TRANSFER_READ) {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::TRANSFER_WRITE) {
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    } else if dst_stage.contains(vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR) || dst_stage.contains(vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR) {
                        vk::ImageLayout::GENERAL
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
                unsafe { renderer.device.device.cmd_execute_commands(ctx.command_buffer, &secondary_commands[idx]); }
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
