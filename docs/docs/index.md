# Introduction

Halsey is a research tool for deep reinforcement learning. It is written in Python and utilizes [tensorflow 2.0](https://www.tensorflow.org/). It is meant to be:

* Easy to read
* Easy to use
* Modular

This makes Halsey straightforward to extend and customize so you can hopefully spend more time investigating research questions! In fact, Halsey's golden rule is: *readability is king*. This means that, inevitably, efficiency is sacrificed in certain places, but Halsey isn't meant for any kind of enterprise use, so that should be alright.

## Installation

Currently, Halsey must be installed from source and has one prerequisite: [Poetry](https://python-poetry.org/docs/#installation). Once Poetry is installed, you can install Halsey with the following commands:

```bash
git clone git@github.com:jcoughlin11/halsey.git halsey_repo
cd halsey_repo/
poetry install
```

## Basic Usage

Halsey is meant to be used as a stand alone code, more akin to a C/C++ executable than a more traditional Python module that gets imported and used elsewhere.

The reinforcement learning options are controlled via a parameter file, which uses the [gin](https://github.com/google/gin-config) format. Once the parameter file has been set up to your liking, a training run can be done by executing:

```bash
poetry run halsey params.gin -t
```
