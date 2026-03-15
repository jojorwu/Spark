# Spark Engine Architecture

Spark is a high-performance, multi-threaded, modular game engine written in Rust. It draws inspiration from Godot (Scene Tree) and Unity (Ease of use) while leveraging Rust's safety and performance.

## 1. High-Level Overview

```ascii
+-------------------------------------------------------------+
|                        SPARK EDITOR                         |
|  (UI, Asset Browser, Scene Inspector, Script Editor, etc.)  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                        SPARK ENGINE                         |
+-------------------------------------------------------------+
|  +-----------------------+       +-----------------------+  |
|  |       SCENE TREE      | <---> |    SCRIPTING HOST     |  |
|  | (Nodes, Hierarchy)    |       | (Rust .dll / C# .NET) |  |
|  +-----------+-----------+       +-----------+-----------+  |
|              |                               |              |
|              v                               v              |
|  +-----------------------+       +-----------------------+  |
|  |    RENDERER (ASH)     | <---> |   TASK SYSTEM (POOL)  |  |
|  | (Vulkan, Pipelines)   |       | (Parallel Processing) |  |
|  +-----------+-----------+       +-----------+-----------+  |
|              |                               |              |
+--------------|-------------------------------|--------------+
               |                               |
               v                               v
+---------------------------+    +---------------------------+
|      GRAPHICS DRIVER      |    |        OS THREADS         |
|         (VULKAN)          |    |  (Windows/Linux/Android)  |
+---------------------------+    +---------------------------+
```

## 2. Core Components

### 2.1 Scene Tree (Godot-style)
The engine uses a node-based hierarchy. Since Rust has strict ownership rules, nodes are managed via a **Handle System** (e.g., `slotmap` or unique IDs) rather than direct pointers.

- **Node**: Base unit of the scene.
- **Transform**: Built-in component for spatial nodes.
- **Resource System**: Handles loading and caching of textures, meshes, and materials.

```ascii
Scene (Root)
 ├── Camera3D (Node)
 ├── Player (Node + Rust/C# Script)
 │    └── MeshInstance (Node)
 └── Light (Node)
```

### 2.2 Renderer (Vulkan/Ash)
The renderer is built on top of `ash` for low-level Vulkan access. It uses a **Render Graph** approach to manage dependencies between passes (shadows, G-buffer, lighting, post-processing).

- **Abstraction Layer**: Hides Vulkan verbosity behind a clean API.
- **Multi-threading**: Command buffers are recorded in parallel using the Task System.
- **Shader System**: SPIR-V based, with support for hot-reloading.

### 2.3 Scripting Host
Two primary scripting methods:
1. **Rust Plugins**: Dynamic loading of `.so`/`.dll` files. Uses a stable ABI or `abi_stable` crate to ensure compatibility.
2. **C# .NET Hosting**: Integrates the .NET Runtime (nethost) to run C# 12+ scripts. High-level C# wrappers call into the Rust engine via FFI.

### 2.4 Multi-threading (Task System)
Spark uses a **Global Thread Pool** (likely using `rayon` or a custom implementation for fine-grained control).
- **Parallel Updates**: Independent nodes can update their logic in parallel.
- **Async Asset Loading**: Assets are loaded and processed on background threads.
- **Physics**: Runs on a separate fixed-step thread or parallelized via the task system.

## 3. Project Structure (Cargo Workspace)

```ascii
spark/
├── Cargo.toml          # Workspace root
├── crates/
│   ├── spark-core/     # Main loop, Scene Tree, Event handling
│   ├── spark-renderer/ # Vulkan implementation (ash)
│   ├── spark-math/     # Vector, Matrix, Quaternions
│   ├── spark-script/   # Rust dynamic loading & .NET hosting
│   └── spark-editor/   # The GUI editor (built on spark-core)
├── plugins/            # Sample Rust plugins
└── examples/           # Demo projects
```

## 4. Scalability and Modularity
The engine is designed as a **Micro-Kernel**. The `spark-core` only contains essential logic. Everything else (Physics, Audio, Networking) is a module that can be added or removed.

- **Traits for Modules**: Modules implement standard traits like `on_init`, `on_update`, `on_render`.
- **Hot Reloading**: Rust plugins and C# scripts can be reloaded without restarting the engine.
