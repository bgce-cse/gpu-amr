# TerraDG Framework - Adaptive Mesh Refinement

The TerraDG framework now features a fully integrated Adaptive Mesh Refinement (AMR) system built on a QuadTree data structure with comprehensive documentation and testing. 

## Key Features

- **Hierarchical mesh refinement** driven by solution gradient analysis for optimal computational efficiency
- **Seamless integration** with existing TerraDG codebase - no disruption to current workflows
- **Runtime configurability** via YAML settings - easily toggle AMR on/off without code changes
- **Balance constraints** to maintain mesh quality

## Technical Implementation

The AMR system utilizes a QuadTree approach that dynamically refines mesh resolution in regions of high solution gradients while maintaining computational efficiency in smoother areas. This selective refinement strategy significantly reduces computational overhead while preserving solution accuracy.

## Usage

AMR functionality can be controlled through the configuration YAML file

**Configuration parameters**:
   - `amr`: Boolean to enable or disable AMR without recompilation
   - `max_level`: Maximum allowed refinement level

**Workflow:**

Use julia REPL (Terminal).

      julia --project=.

In most editors, you can also start a Julia REPL directly. These are mostly
spawned inside your current project.

Always load Revise before loading TerraDG.

     using Revise
     using TerraDG

In this way, changes you make to the document are automatically loaded in your
current environment. Exception: Changing structs.
You can check for compilation errors by using the command Revise.errors()

You can enter package mode by pressing ].
When first loading the module, run instantiate in this mode.

    ] activate .
    ] instantiate

This will install all needed packages automatically.
You can run the unit tests by running the command test in package mode.

    ] test

If you get errors about missing packages you can often fix this by running

    ] resolve

You can look at the julia documentation in the terminal inside the documentation
mode. You can enter it by typing ?, followed by some function.

You can exit these specials modes by typing backspace in the empty command prompt.

**Running TerraDG**

Using julia REPL (Terminal)

    TerraDG.main("path-to-config")
    




