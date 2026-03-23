use ash::vk;
use std::collections::{HashMap, HashSet};
use crate::passes::{RenderPass, RenderContext};
use crate::Renderer;

pub enum ResourceType {
    Image,
    Buffer,
}

pub struct RenderGraphResource {
    pub name: String,
    pub ty: ResourceType,
    pub format: vk::Format,
    pub usage: vk::ImageUsageFlags,
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
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: HashMap::new(),
            sorted_passes: Vec::new(),
        }
    }

    pub fn add_pass<P: RenderPass + 'static>(&mut self, pass: P, inputs: &[&str], outputs: &[&str]) {
        self.passes.push(RenderGraphPassNode {
            pass: Box::new(pass),
            inputs: inputs.iter().map(|&s| s.to_string()).collect(),
            outputs: outputs.iter().map(|&s| s.to_string()).collect(),
        });
    }

    pub fn compile(&mut self) {
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
    }

    pub fn execute(&self, ctx: &RenderContext, renderer: &mut Renderer) {
        // 1. Prepare
        for &idx in &self.sorted_passes {
            self.passes[idx].pass.prepare(renderer, ctx.current_frame);
        }

        // 2. Barrier injection & Recording
        for &idx in &self.sorted_passes {
            let pass_node = &self.passes[idx];

            // Automated Barrier Injection
            for (res_name, dst_access, dst_stage) in pass_node.pass.gpu_resource_access() {
                // Find image resource by name in the renderer's attachment vectors
                let (image, aspect) = match res_name.as_str() {
                    "GBufferHDR" => (renderer.gbuffer_hdr[ctx.current_frame].image, vk::ImageAspectFlags::COLOR),
                    "GBufferAlbedo" => (renderer.gbuffer_albedo[ctx.current_frame].image, vk::ImageAspectFlags::COLOR),
                    "GBufferNormal" => (renderer.gbuffer_normal[ctx.current_frame].image, vk::ImageAspectFlags::COLOR),
                    "GBufferPBR" => (renderer.gbuffer_pbr[ctx.current_frame].image, vk::ImageAspectFlags::COLOR),
                    "GBufferVelocity" => (renderer.gbuffer_velocity[ctx.current_frame].image, vk::ImageAspectFlags::COLOR),
                    "GBufferDepth" => (renderer.gbuffer_depth[ctx.current_frame].image, vk::ImageAspectFlags::DEPTH),
                    _ => (vk::Image::null(), vk::ImageAspectFlags::empty()),
                };

                if image != vk::Image::null() {
                    let new_layout = if dst_access.contains(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) {
                        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE) {
                        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
                    } else if dst_access.contains(vk::AccessFlags::SHADER_READ) {
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
                        image,
                        new_layout,
                        vk::AccessFlags::empty(), // Simplified src access
                        dst_access,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE, // Conservative src stage
                        dst_stage,
                        aspect,
                    );
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
