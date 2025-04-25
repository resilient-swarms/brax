# go2.py
import xml.etree.ElementTree as ET
import jax
import jax.numpy as jp
from pathlib import Path
from brax.envs.base import PipelineEnv, State
from brax.io.mjcf import loads as mjcf_loads


class Go2(PipelineEnv):
    """Unitree Go2 quadruped via MuJoCo XML under Brax Pipelines.

    XML source:
      - Using Unitree Robotics official model: 
        https://github.com/unitreerobotics/unitree_mujoco/tree/main/unitree_robots/go2
    Save go2.xml and assets under `assets/go2.xml` next to this file.

    Methods:
      - reset(rng) → State: noisy initialization
      - step(state, action) → State: apply torque, simulate, compute reward & metrics
      - validate_model() → prints model dimensions & names

    Action space (12-dim torque commands in [-1,1]):

      | Index | Joint                     | Control Min | Control Max | Actuator Name | Unit      |
      |-------|---------------------------|-------------|-------------|---------------|-----------|
      | 0     | Front Right Hip Abduction | -1          | 1           | `FR_hip`      | Torque(Nm)|
      | 1     | Front Right Hip          | -1          | 1           | `FR_thigh`    | Torque(Nm)|
      | 2     | Front Right Knee         | -1          | 1           | `FR_calf`     | Torque(Nm)|
      | 3     | Front Left Hip Abduction  | -1          | 1           | `FL_hip`      | Torque(Nm)|
      | 4     | Front Left Hip           | -1          | 1           | `FL_thigh`    | Torque(Nm)|
      | 5     | Front Left Knee          | -1          | 1           | `FL_calf`     | Torque(Nm)|
      | 6     | Rear Right Hip Abduction  | -1          | 1           | `RR_hip`      | Torque(Nm)|
      | 7     | Rear Right Hip          | -1          | 1           | `RR_thigh`    | Torque(Nm)|
      | 8     | Rear Right Knee         | -1          | 1           | `RR_calf`     | Torque(Nm)|
      | 9     | Rear Left Hip Abduction   | -1          | 1           | `RL_hip`      | Torque(Nm)|
      | 10    | Rear Left Hip           | -1          | 1           | `RL_thigh`    | Torque(Nm)|
      | 11    | Rear Left Knee          | -1          | 1           | `RL_calf`     | Torque(Nm)|

    Observation:
      - Concatenation of body pose (root pos + root orient
        + 12 joint angles) and body velocities (root lin+ang vel
        + 12 joint velocities)
      - Dimension = observation_size

    Config parameters:
      - ctrl_cost_weight: control penalty coefficient
      - healthy_reward: survival bonus per step
      - terminate_when_unhealthy: end episode on fall
      - healthy_z_range: valid height range (min, max)
      - reset_noise_scale: noise scale for reset
      - exclude_root_pos: drop root pos from obs
      - backend: simulation backend ("generalized" default)
      - dt: physics time step
      - substeps: solver substeps
    """
    def __init__(
        self,
        ctrl_cost_weight: float = 0.5,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: tuple = (0.2, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_root_pos: bool = False,
        backend: str = 'generalized',
        xml_path: str = None,
        #dt: float = 0.02,
        #substeps: int = 4,
        **kwargs
    ):
        # locate XML asset
        if xml_path is None:
            xml_path = Path(__file__).parent / 'assets' / 'go2.xml'
        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f'Go2 XML not found at: {xml_path}')

        # load model with full collision compatibility
        #system = mjcf_load(str(xml_path), full_compat=True)

        #xml_text = (Path(__file__).parent / "assets" / "go2.xml").read_text()

        # 1) Read the raw XML text
        xml_text = xml_path.read_text()

        # 2) Parse & rewrite all cylinders → capsules
        root = ET.fromstring(xml_text)

        opt = root.find("option")
        if opt is None:
            opt = ET.SubElement(root, "option")
        opt.set("cone", "pyramidal")
        opt.set("impratio",  "1")
        # (only supports impratio as 1)

        for geom in root.findall(".//geom"):
          if "priority" in geom.attrib:
              del geom.attrib["priority"]

        for geom in root.findall(".//geom"):
            if geom.get("type") == "cylinder":
                geom.set("type", "capsule")
                # MuJoCo capsules only need radius and half-length:
                sizes = geom.get("size", "").split()
                # keep only first two numbers (radius, half-length)
                if len(sizes) >= 2:
                    geom.set("size", f"{sizes[0]} {sizes[1]}")
        # back to string
        patched_xml = ET.tostring(root, encoding="unicode")

        print(f"Loading assets from: {xml_path.parent}")  # debug

        #system   = mjcf_loads(
        #  xml_text,
        #  asset_path=Path(__file__).parent / "assets",
        #  full_compat=True   # <-- this enables cylinder-box, mesh-mesh, etc.
        #)

        # mjcf_loads arguments:
        #   xml: str         -> full MJCF XML as a string
        #   asset_path: Path -> directory to resolve mesh and asset references
        #   full_compat: bool-> if True, allows all MuJoCo collision types (cylinder-box, mesh-mesh)
        # returns: Brax System object
        # parse using string loader with full compatibility for collisions
        sys = mjcf_loads(
            patched_xml,
            asset_path=xml_path.parent
        )#, full_compat=True


        n_frames = 5

        if backend in ['spring', 'positional']:
          sys = sys.tree_replace({'opt.timestep': 0.005})
          n_frames = 10

        if backend == 'mjx':
          sys = sys.tree_replace({
              'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
              'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
              'opt.iterations': 1,
              'opt.ls_iterations': 4,
          })

        if backend == 'positional':
          # TODO: does the same actuator strength work as in spring
          sys = sys.replace(
              actuator=sys.actuator.replace(
                  gear=200 * jp.ones_like(sys.actuator.gear)
              )
          )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)


        # initialize PipelineEnv
        super().__init__(sys=sys, backend=backend, **kwargs)

        # parameters
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_root_pos = exclude_root_pos

    def reset(self, rng: jax.random.PRNGKey) -> State:
        """Resets with noise on q and qd, just like Ant."""
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale

        # joint positions around the XML default
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi)

        # joint velocities sampled as noise around zero
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # init the pipeline, build obs
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        # reward, done, metrics all start at zero
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=jp.zeros(()),    # ← here we explicitly reset the reward to 0
            done=jp.zeros(()),      # ← and the done flag to 0
            metrics={})

    def step(self, state: State, action: jp.ndarray) -> State:
        """Apply action, step physics, compute reward & metrics."""
        p0 = state.pipeline_state
        assert p0 is not None
        p = self.pipeline_step(p0, action)
        # forward velocity
        dx = p.x.pos[0, 0] - p0.x.pos[0, 0]
        v_fwd = dx / self.dt
        # survival bonus
        z = p.x.pos[0, 2]
        healthy = (z >= self._healthy_z_range[0]) & (z <= self._healthy_z_range[1])
        survive = self._healthy_reward if healthy else 0.0
        # control penalty
        cost = self._ctrl_cost_weight * jp.sum(action**2)
        # total reward
        reward = v_fwd + survive - cost
        done = 0.0 if healthy or not self._terminate_when_unhealthy else 1.0
        obs = self._get_obs(p)
        metrics = {'forward_vel': v_fwd, 'survive': survive, 'ctrl_cost': -cost, 'z': z}
        return state.replace(pipeline_state=p, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, ps) -> jp.ndarray:
        """Concatenate q & qd, optionally drop root pos."""
        q = ps.q
        if self._exclude_root_pos:
            q = q[2:]
        return jp.concatenate([q, ps.qd])

    def validate_model(self):
        """Print model stats for sanity check."""
        print('q_size:', self.sys.q_size(), 'qd_size:', self.sys.qd_size(), 'act_size:', self.sys.act_size())
        print('link_names:',      self.sys.link_names)
        print('link_types:',      self.sys.link_types)
        print('link_parents:',    self.sys.link_parents)
        print('actuator.q_id:',   self.sys.actuator.q_id)
        print('actuator.qd_id:',  self.sys.actuator.qd_id)

        # --- NEW geom print-out ---
        # raw integer codes:
        codes = set(int(t) for t in self.sys.mj_model.geom_type)
        print("geom type codes:", codes)

    @property
    def observation_size(self) -> int:
        return self.sys.q_size() + self.sys.qd_size()

    @property
    def action_size(self) -> int:
        return self.sys.act_size()

# Example:
# env = Go2()
# env.validate_model()
