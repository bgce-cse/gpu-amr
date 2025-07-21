# TerraDG.jl
In this worksheet we implement a high-order basis, as well as projections to that basis and
quadrature.


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


**Building the TerraDG Documentation:**

To build the documentation run 

    julia --project=. --color=yes docs/make.jl 

in the root of the repository.

The documentation will then be built in a subdirectory docs/build,
you can open the documentation, eg. in your browser by opening the index.html

**Running TerraDG**

Using julia REPL (Terminal)

    TerraDG.main("path-to-config")
    


