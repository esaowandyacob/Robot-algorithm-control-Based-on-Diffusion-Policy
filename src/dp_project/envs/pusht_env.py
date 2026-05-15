from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pymunk
import shapely.geometry as sg
from pymunk.vec2d import Vec2d


def pymunk_to_shapely(body: pymunk.Body, shapes: list[pymunk.Shape]) -> sg.MultiPolygon:
    geometries = []
    for shape in shapes:
        if not isinstance(shape, pymunk.shapes.Poly):
            raise RuntimeError(f"Unsupported shape type: {type(shape)}")
        vertices = [body.local_to_world(vertex) for vertex in shape.get_vertices()]
        vertices += [vertices[0]]
        geometries.append(sg.Polygon(vertices))
    return sg.MultiPolygon(geometries)


@dataclass
class PushTStepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]


class PushTEnv:
    metadata = {"video.frames_per_second": 10}

    def __init__(
        self,
        legacy: bool = False,
        block_cog: tuple[float, float] | None = None,
        damping: float | None = None,
        reset_to_state: np.ndarray | None = None,
    ) -> None:
        self._seed: int | None = None
        self.np_random = np.random.default_rng()
        self.window_size = 512
        self.sim_hz = 100
        self.control_hz = self.metadata["video.frames_per_second"]
        self.k_p = 100.0
        self.k_v = 20.0
        self.legacy = legacy
        self.block_cog = block_cog
        self.damping = damping
        self.reset_to_state = reset_to_state

        self.space: pymunk.Space | None = None
        self.agent: pymunk.Body | None = None
        self.block: pymunk.Body | None = None
        self.goal_pose = np.array([256.0, 256.0, np.pi / 4], dtype=np.float64)
        self.n_contact_points = 0
        self.success_threshold = 0.95
        self.latest_action: np.ndarray | None = None
        self.seed()

    def seed(self, seed: int | None = None) -> None:
        if seed is None:
            seed = int(np.random.randint(0, 65536))
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.seed(seed)
        self._setup()

        assert self.block is not None
        assert self.space is not None
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=self._seed)
            state = np.array(
                [
                    rs.randint(50, 450),
                    rs.randint(50, 450),
                    rs.randint(100, 400),
                    rs.randint(100, 400),
                    rs.randn() * 2 * np.pi - np.pi,
                ],
                dtype=np.float64,
            )
        self._set_state(np.asarray(state, dtype=np.float64))
        return self._get_obs()

    def step(self, action: np.ndarray) -> PushTStepResult:
        assert self.agent is not None
        assert self.block is not None
        assert self.space is not None

        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        action = np.asarray(action, dtype=np.float64)
        self.latest_action = action

        for _ in range(n_steps):
            acceleration = self.k_p * (action - np.array(self.agent.position)) + self.k_v * (
                np.zeros(2, dtype=np.float64) - np.array(self.agent.velocity)
            )
            self.agent.velocity += Vec2d(*acceleration) * dt
            self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, list(self.block.shapes))
        block_geom = pymunk_to_shapely(self.block, list(self.block.shapes))
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = float(np.clip(coverage / self.success_threshold, 0.0, 1.0))
        done = bool(coverage > self.success_threshold)
        info = self._get_info()
        info["coverage"] = float(coverage)
        return PushTStepResult(
            observation=self._get_obs(),
            reward=reward,
            done=done,
            info=info,
        )

    def close(self) -> None:
        self.space = None
        self.agent = None
        self.block = None

    def _handle_collision(self, arbiter: pymunk.Arbiter, _space: pymunk.Space, _data: Any) -> None:
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _setup(self) -> None:
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0

        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2),
        ]
        self.space.add(*walls)
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        collision_handler = self.space.add_collision_handler(0, 0)
        collision_handler.post_solve = self._handle_collision
        self.n_contact_points = 0

    def _get_obs(self) -> np.ndarray:
        assert self.agent is not None
        assert self.block is not None
        return np.array(
            tuple(self.agent.position) + tuple(self.block.position) + (self.block.angle % (2 * np.pi),),
            dtype=np.float64,
        )

    def _get_goal_pose_body(self, pose: np.ndarray) -> pymunk.Body:
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = float(pose[2])
        return body

    def _get_info(self) -> dict[str, Any]:
        assert self.agent is not None
        assert self.block is not None
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        return {
            "pos_agent": np.array(self.agent.position, dtype=np.float64),
            "vel_agent": np.array(self.agent.velocity, dtype=np.float64),
            "block_pose": np.array(
                [self.block.position[0], self.block.position[1], self.block.angle],
                dtype=np.float64,
            ),
            "goal_pose": self.goal_pose.copy(),
            "n_contacts": n_contact_points_per_step,
        }

    def _set_state(self, state: np.ndarray) -> None:
        assert self.agent is not None
        assert self.block is not None
        assert self.space is not None
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = float(state[4])
        self.agent.position = pos_agent.tolist()
        if self.legacy:
            self.block.position = pos_block.tolist()
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block.tolist()
        self.space.step(1.0 / self.sim_hz)

    def _add_segment(self, a: tuple[float, float], b: tuple[float, float], radius: float) -> pymunk.Shape:
        assert self.space is not None
        return pymunk.Segment(self.space.static_body, a, b, radius)

    def add_circle(self, position: tuple[float, float], radius: float) -> pymunk.Body:
        assert self.space is not None
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        self.space.add(body, shape)
        return body

    def add_tee(
        self,
        position: tuple[float, float],
        angle: float,
        scale: float = 30.0,
        mask: int = pymunk.ShapeFilter.ALL_MASKS(),
    ) -> pymunk.Body:
        assert self.space is not None
        mass = 1
        length = 4
        vertices1 = [
            (-length * scale / 2, scale),
            (length * scale / 2, scale),
            (length * scale / 2, 0),
            (-length * scale / 2, 0),
        ]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [
            (-scale / 2, scale),
            (-scale / 2, length * scale),
            (scale / 2, length * scale),
            (scale / 2, scale),
        ]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

