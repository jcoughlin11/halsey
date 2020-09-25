# Objects

All of Halsey's main components are contained within classes, and each class type is kept as independent as possible. Additionally, Halsey's design uses a top-down, managerial, approach to object management. Halsey's highest level is the `run` function. Run oversees three objects:

* An instructor (training)
* A proctor (evaluating)
* An analyst (analyzing)

Each of these three objects, in turn, oversees a small handful of other objects. These are described below.

Lastly, Halsey employs [abstract base classes](https://docs.python.org/3/library/abc.html) in order to help make extending the code easier, since they ensure that all required methods must be present and accounted for before the code will even run.

## Instructors

The instructor object is responsible for handling the training process, and they live in `halsey.instructors`. The instructor oversees three things:

* The brain
* The navigator
* The training loop
