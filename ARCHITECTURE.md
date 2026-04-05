# Spark Engine Architecture

Spark is a high-performance, multi-threaded, modular game engine written in Rust. It leverages a data-driven approach with a stage-based execution model.

## 1. Core Principles

- **Thread Safety by Design**: Access to core resources (Scene, Renderer, AssetManager) is mediated through a `FrameContext` and a `Scheduler` that guarantees disjoint access for parallel systems.
- **Deferred Mutation**: Structural changes to the engine state (e.g., adding/removing nodes) are deferred using a `CommandQueue` to avoid data races during parallel updates.
- **Event-Driven consistency**: A double-buffered `EventBus` ensures that all systems see a consistent view of events for the entire duration of a frame.

## 2. Main Loop Lifecycle

The engine's main loop (found in `crates/spark-core/src/lib.rs`) follows these phases:

1.  **Frame Start**:
    - Swap event buffers.
    - Update frame timing and FPS metrics.
    - Apply pending engine state changes (e.g., loading a new scene or switching editor modes).
2.  **Update Phase**:
    - Process input events and update `InputManager`.
    - Execute systems registered in `CoreStage::First` through `CoreStage::Last`.
    - **Parallel Execution**: Within each stage, systems are executed in parallel batches if they don't have resource conflicts.
3.  **Deferred Execution**:
    - Execute all commands in the `CommandQueue` (e.g., applying physics transforms, spawning objects).
4.  **Render Phase**:
    - Find the active camera and calculate frustum planes.
    - Collect visible objects and light data from the `Scene` hierarchy.
    - Prepare GPU buffers and execute the `RenderGraph`.

## 3. Systems and Scheduling

All logic in Spark is implemented as a `System`.

- **ResourceAccess**: Systems declare which resources they need (Read or Write). The `Scheduler` uses this to build an execution graph.
- **Stages**:
    - `First`: Logic that must run before anything else (e.g., timer updates).
    - `PreUpdate`: Early logic.
    - `Update`: Main gameplay and component logic.
    - `PostUpdate`: Logic that depends on the results of the update.
    - `Last`: Final cleanup and preparation for rendering.

## 4. Rendering Architecture

Spark uses a **Render Graph** to manage the complexity of its hybrid rendering pipeline.

- **Hybrid Rendering**: Combines traditional rasterization (G-Buffer) with hardware-accelerated ray tracing for secondary effects (reflections, AO, GI).
- **Clustered Shading**: Lighting is performed in a deferred pass using a cluster grid to efficiently handle thousands of light sources.
- **Bindless Textures**: Most textures are bound globally to a single descriptor set, allowing materials to index them dynamically without changing descriptor sets between draw calls.

## 5. Directory Structure

- `crates/spark-core/`: Core engine logic, ECS-like system, scene tree, and main loop.
- `crates/spark-renderer/`: Vulkan implementation using `ash`.
- `crates/spark-math/`: Optimized math library based on `glam`.
- `crates/spark-editor/`: Built-in editor interface using `egui`.
