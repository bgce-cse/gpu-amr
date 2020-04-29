# TerraDG.jl

This is a simple DG Teaching Code. It provides a mesh, plotters, some tests
as well as a framework for the numerical scheme.

In most files the missing implementation is marked with a TODO, others a simply
empty. In the course you fill in these missing pieces.

Currently the tests do not pass successfully, at the end of the course, all tests
should and you will have written some tests of your own.


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
