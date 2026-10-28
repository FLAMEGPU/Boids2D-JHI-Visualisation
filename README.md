# Python Boids example

This is an example of Boids 2D which has been customised for display on the James Hutton Institute Visualisation suite with a display output of 13560 x 1200. It has toroidal wrapping which should ensure that Boids fly continuously around the 360 degree screen.

This FLAME GPU 2 example uses the `pyflamegpu` library. It uses Spatial 2D messaging to demonstrate communication over a fixed radius. The agent function behaviour is specified using ["Agent Python"](https://docs.flamegpu.com/guide/agent-functions/defining-agent-functions.html) a subset of Python which is translated to C++ and compiled just in time. The environment in this example is wrapped. Agents will communicate over boundaries and teleport to form a continuous torus environment.

# Running the example

To run this template locally you will require:

+ Python `>= 3.10`
+ CUDA `>= 13.0` and a [Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 7.5 NVIDIA GPU (note you can use CUDA 12.0+ but will need to modify the extra-index-url in the requirements file)
+ Linux with `glibc >= 2.28`
  + Windows support/instructions will be introduced at a later date
+ An `x86_64` CPU
  + Including [`NVRTC`](https://docs.nvidia.com/cuda/nvrtc/index.html)
  + If you wish to run this template on an `ppc64le` or `arm64-sbsa` based-system, you will need to build `pyflamegpu` from source.
  Please see the main [FLAMEGPU/FLAMEGPU2](https://github.com/FLAMEGPU/FLAMEGPU2) repository for instructions on building from source.

1. Clone the repository if you have not already done so
2. Create a new python `venv` or conda environment and activate it
  
    ```bash
    # I.e. if using a venv on linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install Dependencies

```
python3 -m pip install -r requirements.txt
```

4. Run the boids model.

`python boids_spatial2D.py`

or just

`./boids_spatial3D.py`

in Linux.

# Visualisation

If running with visualisation enabled then passing the argument `--steps 0` will ensure the simulation runs indefinitely. E.g. 

`python boids_spatial3D.py --steps 0`

Alternatively, specifying 100 steps would allow you to explore the final simulation state after 100 steps have completed. The visualisation in this example allows you to control some of the environment variables effecting behaviour. E.g. 

* **TIME_SCALE**: Controls the time steps each simulation step represents. Increasing this value will increase simulation speed but reduce the numerical accuracy of the velocity calculations. 
* **GLOBAL_SCALE**: Controls the maximum force that can be applied to an agent per time step.
* **SEPARATION_RADIUS**: Controls the distance of the separation force. This is the distance over which agents will avoid each other due to collision.
* **STEER_SCALE**: Controls the weight of the steering force (cohesion) applied to the agents per time step. The steering force encourages agents to form cohesive groups.
* **COLLISION_SCALE**: Controls the weight of the collision (separation) force applied to the agents per time step. The collision force is applied over a small distance (the SEPARATION_RADIUS).
* **MATCH_SCALE**:  Controls the weight of the match speed force  (alignment) applied to the agents per time step. The match speed force encourages agents to steer towards the average heading.
