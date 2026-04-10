use crate::resource::{RenderFrame, MAX_FRAMES_IN_FLIGHT};
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::swapchain::VulkanSwapchain;
use ash::vk;

pub struct FrameManager {
    pub frames: [RenderFrame; MAX_FRAMES_IN_FLIGHT],
    pub current_frame: usize,
    pub frame_index: u64,
    pub culling_finished_semaphores: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
}

impl FrameManager {
    pub fn new(device: &VulkanDevice) -> Result<Self, crate::error::RendererError> {
        let (av, fi, in_f) = Self::create_sync_objects(&device.device);
        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(device.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cb = unsafe {
                    device
                        .device
                        .allocate_command_buffers(&alloc_info)
                        .expect("Failed to allocate per-frame command buffer")[0]
                };
                RenderFrame {
                    command_buffer: cb,
                    image_available: av[i],
                    render_finished: fi[i],
                    in_flight: in_f[i],
                    global_buffer: None,
                    light_buffer: None,
                    global_descriptor_set: vk::DescriptorSet::null(),
                    instance_pool: Vec::new(),
                    instance_index: 0,
                    indirect_commands_buffer: None,
                    object_data_buffer: None,
                    draw_count_buffer: None,
                    transparent_indirect_buffer: None,
                    transparent_object_buffer: None,
                    secondary_command_buffers: Vec::new(),
                    light_view_projs: [spark_math::Mat4::IDENTITY; 4],
                    scratch_buffers: Vec::new(),
                    texture_staging_buffer: None,
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .expect("Failed to convert per-frame buffer list to fixed-size array");
        let mut culling_finished_semaphores = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
        for sem in culling_finished_semaphores.iter_mut() {
            *sem = unsafe {
                device
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
            };
        }
        Ok(Self {
            frames,
            current_frame: 0,
            frame_index: 0,
            culling_finished_semaphores,
        })
    }
    fn create_sync_objects(
        device: &ash::Device,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut av = Vec::new();
        let mut fi = Vec::new();
        let mut in_f = Vec::new();
        let s_info = vk::SemaphoreCreateInfo::default();
        let f_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                av.push(
                    device
                        .create_semaphore(&s_info, None)
                        .expect("Failed to create image available semaphore"),
                );
                fi.push(
                    device
                        .create_semaphore(&s_info, None)
                        .expect("Failed to create render finished semaphore"),
                );
                in_f.push(
                    device
                        .create_fence(&f_info, None)
                        .expect("Failed to create in-flight fence"),
                );
            }
        }
        (av, fi, in_f)
    }
    pub fn acquire_next_image(
        &mut self,
        device: &ash::Device,
        swapchain: &VulkanSwapchain,
    ) -> Result<u32, vk::Result> {
        let frame = &self.frames[self.current_frame];
        unsafe {
            device.wait_for_fences(&[frame.in_flight], true, u64::MAX)?;
            let result = swapchain.loader.acquire_next_image(
                swapchain.handle,
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            );
            match result {
                Ok((index, _)) => {
                    device.reset_fences(&[frame.in_flight])?;
                    Ok(index)
                }
                Err(e) => Err(e),
            }
        }
    }
    pub fn advance_frame(&mut self) {
        self.frame_index += 1;
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    pub fn destroy(&mut self, device: &VulkanDevice) {
        unsafe {
            for frame in &mut self.frames {
                if let Some(lb) = frame.light_buffer.take() {
                    device.destroy_buffer(lb);
                }
                if let Some(gb) = frame.global_buffer.take() {
                    device.destroy_buffer(gb);
                }
                for ib in frame.instance_pool.drain(..) {
                    device.destroy_buffer(ib);
                }
                if let Some(ib) = frame.indirect_commands_buffer.take() {
                    device.destroy_buffer(ib);
                }
                if let Some(ob) = frame.object_data_buffer.take() {
                    device.destroy_buffer(ob);
                }
                if let Some(dc) = frame.draw_count_buffer.take() {
                    device.destroy_buffer(dc);
                }
                if let Some(ts) = frame.texture_staging_buffer.take() {
                    device.destroy_buffer(ts);
                }
                device.device.destroy_semaphore(frame.image_available, None);
                device.device.destroy_semaphore(frame.render_finished, None);
                device.device.destroy_fence(frame.in_flight, None);
            }
            for sem in self.culling_finished_semaphores {
                device.device.destroy_semaphore(sem, None);
            }
        }
    }
}
