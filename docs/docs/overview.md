# Overview

Halsey's main driver function is called `run`, and it's located in `halsey.console.run`. This function sets up the session by parsing the command-line options as well as the parameter file. `run` then oversees Halsey's top-level operations: **training**, **evaluating**, and **analyzing** a neural network and its performance.

## Training

The central object of interest when training is called the **instructor**. You can think of this object as the manager for the training process. The instructor is responsible for three things:

* The **brain** object
* The **navigator** object
* The **training loop**

The instructor objects (the base class and its subclasses) all live inside `halsey.instructors`.

Training is enabled by passing either the `-t` or `--train` options at the command-line, e.g.,

```bash
poetry run halsey params.gin -t
```

## Evaluating

The central object of interest when evaluating a model is called the **proctor**. Proctors are not yet implemented.

## Analyzing

The central object of interest when analyzing a model is called the **analyst**. Analysts are not yet implemented.
