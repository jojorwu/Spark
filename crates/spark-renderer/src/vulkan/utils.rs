use ash::vk;

pub struct RenderingAttachmentBuilder {
    image_view: vk::ImageView,
    image_layout: vk::ImageLayout,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    clear_value: vk::ClearValue,
}

impl RenderingAttachmentBuilder {
    pub fn new(view: vk::ImageView) -> Self {
        Self {
            image_view: view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
        }
    }

    pub fn with_layout(mut self, layout: vk::ImageLayout) -> Self {
        self.image_layout = layout;
        self
    }

    pub fn with_load_op(mut self, op: vk::AttachmentLoadOp) -> Self {
        self.load_op = op;
        self
    }

    pub fn with_clear_color(mut self, color: [f32; 4]) -> Self {
        self.clear_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        };
        self
    }

    pub fn with_clear_depth(mut self, depth: f32) -> Self {
        self.clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
        };
        self.image_layout = vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL;
        self
    }

    pub fn build(&self) -> vk::RenderingAttachmentInfo<'static> {
        vk::RenderingAttachmentInfo::default()
            .image_view(self.image_view)
            .image_layout(self.image_layout)
            .load_op(self.load_op)
            .store_op(self.store_op)
            .clear_value(self.clear_value)
    }
}

/// Calculates the Halton sequence value for a given index and base.
pub fn halton(index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0;
    let mut i = index;
    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i /= base;
    }
    result
}
