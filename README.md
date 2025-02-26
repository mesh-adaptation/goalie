## Goalie Goal-Oriented Mesh Adaptation Toolkit

![GitHub top language](https://img.shields.io/github/languages/top/mesh-adaptation/goalie)
![GitHub repo size](https://img.shields.io/github/repo-size/mesh-adaptation/goalie)
[![Slack](https://img.shields.io/badge/Goalie_Slack_Channel-4A154B?logo=slack&logoColor=fff)](https://firedrakeproject.slack.com/archives/C07L813FG74)

Goalie provides goal-oriented mesh adaptation pipelines for solving partial differential equations (PDEs) using adapted meshes built on the Python-based finite element library [Firedrake](http://www.firedrakeproject.org/).  It runs a fixed point iteration loop for progressively solving time-dependent PDEs and their adjoints on sequences of meshes, performing goal-oriented error estimation, and adapting the meshes in sequence with a user-provided adaptor function until defined convergence criteria have been met. It is recommended that users are familiar with adjoint methods, mesh adaptation and the goal-oriented framework before starting with Goalie.

For more information on Firedrake, please see: [Firedrake documentation](https://firedrakeproject.org/documentation.html).

For more information on the implementation of the adjoint method, please see: [dolfin-adjoint documentation](http://www.dolfin-adjoint.org/en/latest/documentation/maths/index.html). 

For more information on the goal-oriented mesh adaptation, please see: [Goalie documentation](https://mesh-adaptation.github.io/goalie/index.html)

## Installation

For installation instructions, we refer to the [Wiki page](https://github.com/mesh-adaptation/mesh-adaptation-docs/wiki/Installation-Instructions).
