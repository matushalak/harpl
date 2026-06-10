"""Attention-task stimuli built from the moving animal sprites.

The dataset intentionally separates stimulus generation from the existing
single-sprite RPL training dataset. Each sample returns raw frames plus dense
target and prompt labels suitable for future decoder training.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import kornia as K
from PIL import Image
from torch.utils.data import Dataset


NO_PROMPT_CLASS = -1

TASK_TO_ID = {
    "object_recognition": 0,
    "popout": 1,
    "tracking": 2,
    "top_down_search": 3,
    "perceptual_grouping": 4,
}

ID_TO_TASK = {task_id: task_name for task_name, task_id in TASK_TO_ID.items()}

POPOUT_MODE_TO_ID = {
    "none": -1,
    "class": 0,
    "rotation": 1,
    "velocity": 2,
}


@dataclass(frozen=True)
class _ObjectSpec:
    class_id: int
    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray
    velocity: np.ndarray
    angular_speed: float
    speed: float


class MovingAnimalAttentionDataset(Dataset):
    """Generate moving-animal attention task videos on a configurable canvas.

    Tasks:
        object_recognition: one sprite in high noise with optional static blots.
        popout: class-, rotation-, or velocity-defined odd-one-out.
        tracking: target appears alone, then distractors enter from borders.
        top_down_search: crowded scene with a class prompt.
        perceptual_grouping: fixation marker traces the target before border distractors are revealed.

    Samples are deterministic by index. Objects only bounce off walls; there is
    no object-object collision response, so sprites pass over each other.
    """

    valid_tasks = tuple(TASK_TO_ID) + ("mixed",)
    valid_popout_modes = ("class", "rotation", "velocity", "mixed")
    valid_noise_types = (None, "gaussian", "salt_pepper")

    def __init__(
        self,
        data_dir: str | Path,
        task: str | Iterable[int | str] = "mixed",
        tasks: Iterable[int | str] | None = None,
        output_size: tuple[int, int] = (64, 64),
        base_output_size: tuple[int, int] = (64, 64),
        scale_pixel_parameters: bool = True,
        seq_len: int = 32,
        num_sequences: int = 1000,
        sprite_img_dir: str = "animals",
        max_sprites: int | None = None,
        seed: int = 42,
        background: float = 0.5,
        device: str | torch.device = "cpu",
        noise_type: str | None = "gaussian",
        noise_level: float | None = None,
        training_noise_level: float = 0.1,
        object_recognition_noise_level: float = 0.35,
        freeze_noise: bool = False,
        noise_on_top: bool = True,
        popout_mode: str = "class",
        num_distractors: int = 3,
        crowd_size: int = 5,
        cue_frames: int = 6,
        occluder_count: int = 4,
        occluder_min_size: int = 8,
        occluder_max_size: int = 18,
        fixation_size: int = 3,
        scale_range: tuple[float, float] = (0.5, 1.0),
        speed_range: tuple[float, float] = (1.25, 4.0),
        slow_speed_range: tuple[float, float] = (0.8, 2.0),
        fast_speed_range: tuple[float, float] = (4.8, 7.0),
        angular_speed_range: tuple[float, float] = (-18.0, 18.0),
        rotation_popout_speed: float = 16.0,
        velocity_popout_kind: str = "fast",
        normalize: bool = False,
        mean: float | tuple[float, float, float] | None = None,
        std: float | tuple[float, float, float] | None = None,
        return_metadata: bool = False,
    ):
        self.task_pool = self._normalize_task_pool(task, tasks)
        if popout_mode not in self.valid_popout_modes:
            raise ValueError(f"popout_mode must be one of {self.valid_popout_modes}")
        if noise_type not in self.valid_noise_types:
            raise ValueError(f"noise_type must be one of {self.valid_noise_types}")
        if velocity_popout_kind not in ("fast", "slow", "mixed"):
            raise ValueError("velocity_popout_kind must be 'fast', 'slow', or 'mixed'")
        if cue_frames < 1:
            raise ValueError("cue_frames must be at least 1")
        if seq_len <= cue_frames:
            raise ValueError("seq_len must be greater than cue_frames")
        if crowd_size < 2:
            raise ValueError("crowd_size must be at least 2")
        if num_distractors < 1:
            raise ValueError("num_distractors must be at least 1")

        self.data_dir = Path(data_dir)
        self.task = self.task_pool[0] if len(self.task_pool) == 1 else "mixed"
        self.tasks = self.task_pool
        self.task_ids = tuple(TASK_TO_ID[task_name] for task_name in self.task_pool)
        self.output_size = self._as_size_tuple(output_size, "output_size")
        self.base_output_size = self._as_size_tuple(base_output_size, "base_output_size")
        self.scale_pixel_parameters = scale_pixel_parameters
        self.pixel_scale = self._compute_pixel_scale()
        self.seq_len = seq_len
        self.num_sequences = num_sequences
        self.sprite_img_dir = sprite_img_dir
        self.max_sprites = max_sprites
        self.seed = seed
        self.background = float(background)
        self.device = torch.device(device)
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.training_noise_level = float(training_noise_level)
        self.object_recognition_noise_level = float(object_recognition_noise_level)
        self.freeze_noise = freeze_noise
        self.noise_on_top = noise_on_top
        self.popout_mode = popout_mode
        self.num_distractors = num_distractors
        self.crowd_size = crowd_size
        self.cue_frames = cue_frames
        self.occluder_count = occluder_count
        self.occluder_min_size = self._scale_pixel_int(occluder_min_size)
        self.occluder_max_size = self._scale_pixel_int(occluder_max_size)
        self.fixation_size = self._scale_pixel_float(fixation_size)
        self.scale_range = self._scale_pixel_range(scale_range)
        self.speed_range = self._scale_pixel_range(speed_range)
        self.slow_speed_range = self._scale_pixel_range(slow_speed_range)
        self.fast_speed_range = self._scale_pixel_range(fast_speed_range)
        self.angular_speed_range = angular_speed_range
        self.rotation_popout_speed = float(rotation_popout_speed)
        self.velocity_popout_kind = velocity_popout_kind
        self.normalize = normalize
        self.mean = self._as_channel_array(mean) if mean is not None else None
        self.std = self._as_channel_array(std) if std is not None else None
        self.return_metadata = return_metadata
        self.max_objects = max(1, num_distractors + 1, crowd_size)

        if self.normalize and (self.mean is None or self.std is None):
            raise ValueError("mean and std are required when normalize=True")
        if self.occluder_min_size > self.occluder_max_size:
            raise ValueError("occluder_min_size must be <= occluder_max_size after scaling")

        self.sprite_paths = self._find_sprite_paths()
        self.sprites = [self._load_sprite_tensor(path) for path in self.sprite_paths]
        self._sprites_have_common_shape = len({tuple(sprite.shape) for sprite in self.sprites}) == 1
        self.class_names = [path.stem for path in self.sprite_paths]
        if len(self.sprites) < 2:
            raise ValueError("At least two sprites are required for attention tasks")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | dict[str, Any]]:
        rng = np.random.default_rng(self.seed + idx * 9973)
        task = self._choose_task(rng)
        popout_mode = self._choose_popout_mode(rng) if task == "popout" else "none"

        if task == "object_recognition":
            spec = self._make_object_recognition(rng)
        elif task == "popout":
            spec = self._make_popout(rng, popout_mode)
        elif task == "tracking":
            spec = self._make_tracking(rng)
        elif task == "top_down_search":
            spec = self._make_top_down_search(rng)
        elif task == "perceptual_grouping":
            spec = self._make_perceptual_grouping(rng)
        else:
            raise RuntimeError(f"Unhandled task: {task}")

        video = self._render(spec, rng)
        if self.normalize:
            mean = torch.as_tensor(self.mean, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
            std = torch.as_tensor(self.std, dtype=video.dtype, device=video.device).view(1, 3, 1, 1)
            video = (video - mean) / std
        labels = self._make_labels(spec)
        if not self.return_metadata:
            return video, labels["task_info"]
        return video, labels

    def _find_sprite_paths(self) -> list[Path]:
        sprite_dir = self.data_dir / self.sprite_img_dir
        paths = sorted(sprite_dir.glob("sprite_*.png"), key=self._sprite_sort_key)
        if self.max_sprites is not None:
            paths = paths[: self.max_sprites]
        if not paths:
            raise FileNotFoundError(f"No sprite_*.png files found in {sprite_dir}")
        return paths

    def _load_sprite_tensor(self, path: Path) -> torch.Tensor:
        """Load an RGBA sprite as RGB+alpha, matching SpriteVideoDataset."""
        pil_img = Image.open(path).convert("RGBA")
        alpha = pil_img.split()[3]
        rgb = Image.new("RGB", pil_img.size, (0, 0, 0))
        rgb.paste(pil_img, mask=alpha)

        rgb_np = np.asarray(rgb, dtype=np.float32) / 255.0
        alpha_np = np.asarray(alpha, dtype=np.float32) / 255.0
        sprite_np = np.concatenate([rgb_np.transpose(2, 0, 1), alpha_np[None]], axis=0)
        return torch.from_numpy(sprite_np).to(self.device)

    def _sprite_radius(self, class_id: int, scale: float) -> float:
        height, width = self.sprites[class_id].shape[-2:]
        return max(height, width) * scale * 0.5

    @staticmethod
    def _sprite_sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"sprite_(\d+)", path.stem)
        return (int(match.group(1)) if match else 10**9, path.name)

    @staticmethod
    def _as_size_tuple(value: tuple[int, int], name: str) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"{name} must be a length-2 (height, width) tuple")
        height, width = int(value[0]), int(value[1])
        if height < 1 or width < 1:
            raise ValueError(f"{name} dimensions must be positive")
        return height, width

    def _compute_pixel_scale(self) -> float:
        if not self.scale_pixel_parameters:
            return 1.0
        return min(
            self.output_size[0] / self.base_output_size[0],
            self.output_size[1] / self.base_output_size[1],
        )

    def _scale_pixel_float(self, value: float) -> float:
        return float(value) * self.pixel_scale

    def _scale_pixel_int(self, value: int) -> int:
        return max(1, int(round(float(value) * self.pixel_scale)))

    def _scale_pixel_range(self, value: tuple[float, float]) -> tuple[float, float]:
        return (
            self._scale_pixel_float(value[0]),
            self._scale_pixel_float(value[1]),
        )

    @staticmethod
    def _as_channel_array(value: float | tuple[float, float, float]) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.repeat(arr[None], 3)
        if arr.shape != (3,):
            raise ValueError("mean/std must be scalar or length-3")
        return arr.astype(np.float32)

    def _normalize_task_pool(
        self,
        task: str | Iterable[int | str],
        tasks: Iterable[int | str] | None,
    ) -> tuple[str, ...]:
        if tasks is not None:
            if not isinstance(task, str) or task != "mixed":
                raise ValueError("Specify either task or tasks, not both.")
            values = list(tasks)
        elif isinstance(task, str):
            if task not in self.valid_tasks:
                raise ValueError(f"task must be one of {self.valid_tasks}")
            values = list(TASK_TO_ID) if task == "mixed" else [task]
        else:
            values = list(task)

        if not values:
            raise ValueError("At least one task must be specified.")

        normalized: list[str] = []
        for value in values:
            if isinstance(value, str):
                if value == "mixed":
                    raise ValueError("tasks cannot include 'mixed'; pass explicit task names or IDs.")
                if value not in TASK_TO_ID:
                    raise ValueError(f"Unknown attention task: {value}")
                task_name = value
            elif isinstance(value, (int, np.integer)):
                task_id = int(value)
                if task_id not in ID_TO_TASK:
                    raise ValueError(f"Unknown attention task ID: {task_id}")
                task_name = ID_TO_TASK[task_id]
            else:
                raise TypeError("tasks must contain task name strings or integer task IDs.")
            if task_name not in normalized:
                normalized.append(task_name)
        return tuple(normalized)

    def _choose_task(self, rng: np.random.Generator) -> str:
        if len(self.task_pool) == 1:
            return self.task_pool[0]
        return str(rng.choice(self.task_pool))

    def _choose_popout_mode(self, rng: np.random.Generator) -> str:
        if self.popout_mode != "mixed":
            return self.popout_mode
        return str(rng.choice(["class", "rotation", "velocity"]))

    def _sample_class_ids(
        self,
        rng: np.random.Generator,
        count: int,
        exclude: set[int] | None = None,
        replace: bool = False,
    ) -> list[int]:
        exclude = exclude or set()
        available = [idx for idx in range(len(self.sprites)) if idx not in exclude]
        if not available:
            available = list(range(len(self.sprites)))
        if count <= len(available) and not replace:
            return [int(v) for v in rng.choice(available, size=count, replace=False)]
        return [int(v) for v in rng.choice(available, size=count, replace=True)]

    def _random_scale(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.scale_range))

    def _random_speed(self, rng: np.random.Generator, speed_range: tuple[float, float] | None = None) -> float:
        speed_range = speed_range or self.speed_range
        return float(rng.uniform(*speed_range))

    def _random_angular_speed(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.angular_speed_range))

    def _make_object(
        self,
        rng: np.random.Generator,
        class_id: int,
        speed_range: tuple[float, float] | None = None,
        speed: float | None = None,
        angular_speed: float | None = None,
        seq_len: int | None = None,
    ) -> _ObjectSpec:
        seq_len = seq_len or self.seq_len
        scale = self._random_scale(rng)
        speed = self._random_speed(rng, speed_range) if speed is None else float(speed)
        direction = float(rng.uniform(0.0, 2.0 * math.pi))
        velocity = np.array([math.cos(direction), math.sin(direction)], dtype=np.float32) * speed
        angular_speed = self._random_angular_speed(rng) if angular_speed is None else float(angular_speed)
        positions, rotations, scales = self._roll_trajectory(
            rng,
            seq_len,
            scale,
            velocity,
            angular_speed,
            class_id=class_id,
        )
        return _ObjectSpec(
            class_id=class_id,
            positions=positions,
            rotations=rotations,
            scales=scales,
            velocity=velocity,
            angular_speed=angular_speed,
            speed=speed,
        )

    def _roll_trajectory(
        self,
        rng: np.random.Generator,
        seq_len: int,
        scale: float,
        velocity: np.ndarray,
        angular_speed: float,
        class_id: int = 0,
        initial_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = self.output_size
        sprite_radius = self._sprite_radius(class_id, scale)
        x_min, x_max = sprite_radius + 1.0, width - sprite_radius - 1.0
        y_min, y_max = sprite_radius + 1.0, height - sprite_radius - 1.0
        if initial_position is None:
            pos = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32)
        else:
            pos = initial_position.astype(np.float32).copy()
            pos[0] = np.clip(pos[0], x_min, x_max)
            pos[1] = np.clip(pos[1], y_min, y_max)
        vel = velocity.astype(np.float32).copy()
        initial_rotation = float(rng.uniform(0.0, 360.0))

        positions = np.zeros((seq_len, 2), dtype=np.float32)
        rotations = np.zeros(seq_len, dtype=np.float32)
        scales = np.full(seq_len, scale, dtype=np.float32)

        for t in range(seq_len):
            positions[t] = pos
            rotations[t] = (initial_rotation + angular_speed * t) % 360.0
            if t == seq_len - 1:
                break
            next_pos = pos + vel
            if next_pos[0] < x_min:
                next_pos[0] = 2.0 * x_min - next_pos[0]
                vel[0] = abs(vel[0])
            elif next_pos[0] > x_max:
                next_pos[0] = 2.0 * x_max - next_pos[0]
                vel[0] = -abs(vel[0])
            if next_pos[1] < y_min:
                next_pos[1] = 2.0 * y_min - next_pos[1]
                vel[1] = abs(vel[1])
            elif next_pos[1] > y_max:
                next_pos[1] = 2.0 * y_max - next_pos[1]
                vel[1] = -abs(vel[1])
            pos = next_pos

        return positions, rotations, scales

    def _make_object_recognition(self, rng: np.random.Generator) -> dict[str, Any]:
        class_id = self._sample_class_ids(rng, 1)[0]
        target = self._make_object(rng, class_id)
        visible = np.ones((self.seq_len, 1), dtype=bool)
        return self._spec(
            task="object_recognition",
            objects=[target],
            target_index=0,
            prompt_class=NO_PROMPT_CLASS,
            visible=visible,
            render_order=[0],
            occluders=self._make_occluders(rng),
        )

    def _make_popout(self, rng: np.random.Generator, mode: str) -> dict[str, Any]:
        count = self.num_distractors + 1
        target_index = int(rng.integers(0, count))
        visible = np.ones((self.seq_len, count), dtype=bool)

        if mode == "class":
            target_class, distractor_class = self._sample_class_ids(rng, 2)
            objects = []
            for obj_idx in range(count):
                class_id = target_class if obj_idx == target_index else distractor_class
                objects.append(self._make_object(rng, class_id))
        elif mode == "rotation":
            class_ids = self._sample_class_ids(rng, count)
            base_sign = int(rng.choice([-1, 1]))
            objects = []
            for obj_idx, class_id in enumerate(class_ids):
                sign = -base_sign if obj_idx == target_index else base_sign
                objects.append(self._make_object(rng, class_id, angular_speed=sign * self.rotation_popout_speed))
        elif mode == "velocity":
            class_ids = self._sample_class_ids(rng, count)
            kind = self.velocity_popout_kind
            if kind == "mixed":
                kind = str(rng.choice(["fast", "slow"]))
            target_range = self.fast_speed_range if kind == "fast" else self.slow_speed_range
            distractor_range = self.slow_speed_range if kind == "fast" else self.fast_speed_range
            objects = []
            for obj_idx, class_id in enumerate(class_ids):
                speed_range = target_range if obj_idx == target_index else distractor_range
                objects.append(self._make_object(rng, class_id, speed_range=speed_range))
        else:
            raise ValueError(f"Unknown popout mode: {mode}")

        render_order = list(rng.permutation(count))
        return self._spec(
            task="popout",
            objects=objects,
            target_index=target_index,
            prompt_class=NO_PROMPT_CLASS,
            visible=visible,
            render_order=render_order,
            popout_mode=mode,
        )

    def _make_tracking(self, rng: np.random.Generator) -> dict[str, Any]:
        target_class = self._sample_class_ids(rng, 1)[0]
        distractor_classes = self._sample_class_ids(
            rng, self.crowd_size - 1, exclude={target_class}, replace=False
        )
        target = self._make_object(rng, target_class)
        distractors = [
            self._make_border_reveal_object(rng, class_id, self.cue_frames)
            for class_id in distractor_classes
        ]
        objects = [target] + distractors
        visible = np.ones((self.seq_len, len(objects)), dtype=bool)
        visible[: self.cue_frames, 1:] = False
        render_order = [0] + list(rng.permutation(np.arange(1, len(objects))))
        if rng.random() < 0.7:
            render_order = list(rng.permutation(len(objects)))
        return self._spec(
            task="tracking",
            objects=objects,
            target_index=0,
            prompt_class=NO_PROMPT_CLASS,
            visible=visible,
            render_order=render_order,
        )

    def _make_border_reveal_object(
        self,
        rng: np.random.Generator,
        class_id: int,
        cue_frames: int,
    ) -> _ObjectSpec:
        seq_len = self.seq_len - cue_frames
        scale = self._random_scale(rng)
        speed = self._random_speed(rng)
        direction = float(rng.uniform(0.0, 2.0 * math.pi))
        velocity = np.array([math.cos(direction), math.sin(direction)], dtype=np.float32) * speed
        angular_speed = self._random_angular_speed(rng)
        initial_position = self._sample_border_position(rng, class_id, scale)
        obj_positions, obj_rotations, obj_scales = self._roll_trajectory(
            rng,
            seq_len,
            scale,
            velocity,
            angular_speed,
            class_id=class_id,
            initial_position=initial_position,
        )
        positions = np.repeat(obj_positions[:1], self.seq_len, axis=0)
        rotations = np.repeat(obj_rotations[:1], self.seq_len, axis=0)
        scales = np.repeat(obj_scales[:1], self.seq_len, axis=0)
        positions[cue_frames:] = obj_positions
        rotations[cue_frames:] = obj_rotations
        scales[cue_frames:] = obj_scales
        return _ObjectSpec(
            class_id=class_id,
            positions=positions,
            rotations=rotations,
            scales=scales,
            velocity=velocity,
            angular_speed=angular_speed,
            speed=speed,
        )

    def _sample_border_position(
        self,
        rng: np.random.Generator,
        class_id: int,
        scale: float,
    ) -> np.ndarray:
        height, width = self.output_size
        sprite_radius = self._sprite_radius(class_id, scale)
        x_min, x_max = sprite_radius + 1.0, width - sprite_radius - 1.0
        y_min, y_max = sprite_radius + 1.0, height - sprite_radius - 1.0
        side = int(rng.integers(0, 4))
        if side == 0:
            return np.array([x_min, rng.uniform(y_min, y_max)], dtype=np.float32)
        if side == 1:
            return np.array([x_max, rng.uniform(y_min, y_max)], dtype=np.float32)
        if side == 2:
            return np.array([rng.uniform(x_min, x_max), y_min], dtype=np.float32)
        return np.array([rng.uniform(x_min, x_max), y_max], dtype=np.float32)

    def _make_top_down_search(self, rng: np.random.Generator) -> dict[str, Any]:
        class_ids = self._sample_class_ids(rng, self.crowd_size)
        target_index = int(rng.integers(0, self.crowd_size))
        objects = [self._make_object(rng, class_id) for class_id in class_ids]
        visible = np.ones((self.seq_len, len(objects)), dtype=bool)
        render_order = list(rng.permutation(len(objects)))
        return self._spec(
            task="top_down_search",
            objects=objects,
            target_index=target_index,
            prompt_class=objects[target_index].class_id,
            visible=visible,
            render_order=render_order,
        )

    def _make_perceptual_grouping(self, rng: np.random.Generator) -> dict[str, Any]:
        class_ids = self._sample_class_ids(rng, self.crowd_size)
        target_index = int(rng.integers(0, self.crowd_size))
        objects = []
        for obj_idx, class_id in enumerate(class_ids):
            if obj_idx == target_index:
                objects.append(self._hold_target_at_last_fixation_frame(self._make_object(rng, class_id)))
            else:
                objects.append(self._make_border_reveal_object(rng, class_id, self.cue_frames))
        visible = np.ones((self.seq_len, len(objects)), dtype=bool)
        visible[: self.cue_frames, :] = False

        render_order = [idx for idx in rng.permutation(len(objects)) if idx != target_index]
        render_order.append(target_index)
        fixation_visible = np.zeros(self.seq_len, dtype=bool)
        fixation_visible[: self.cue_frames] = True
        return self._spec(
            task="perceptual_grouping",
            objects=objects,
            target_index=target_index,
            prompt_class=NO_PROMPT_CLASS,
            visible=visible,
            render_order=render_order,
            fixation_positions=objects[target_index].positions,
            fixation_visible=fixation_visible,
        )

    def _hold_target_at_last_fixation_frame(self, obj: _ObjectSpec) -> _ObjectSpec:
        positions = obj.positions.copy()
        rotations = obj.rotations.copy()
        scales = obj.scales.copy()
        positions[self.cue_frames :] = obj.positions[self.cue_frames - 1 : -1]
        rotations[self.cue_frames :] = obj.rotations[self.cue_frames - 1 : -1]
        scales[self.cue_frames :] = obj.scales[self.cue_frames - 1 : -1]
        return _ObjectSpec(
            class_id=obj.class_id,
            positions=positions,
            rotations=rotations,
            scales=scales,
            velocity=obj.velocity,
            angular_speed=obj.angular_speed,
            speed=obj.speed,
        )

    def _spec(
        self,
        task: str,
        objects: list[_ObjectSpec],
        target_index: int,
        prompt_class: int,
        visible: np.ndarray,
        render_order: list[int],
        popout_mode: str = "none",
        occluders: list[dict[str, Any]] | None = None,
        fixation_positions: np.ndarray | None = None,
        fixation_visible: np.ndarray | None = None,
    ) -> dict[str, Any]:
        target_class = objects[target_index].class_id
        return {
            "task": task,
            "task_id": TASK_TO_ID[task],
            "popout_mode": popout_mode,
            "popout_mode_id": POPOUT_MODE_TO_ID[popout_mode],
            "objects": objects,
            "object_count": len(objects),
            "target_index": target_index,
            "target_class": target_class,
            "prompt_class": prompt_class,
            "visible": visible,
            "render_order": render_order,
            "occluders": occluders or [],
            "fixation_positions": fixation_positions,
            "fixation_visible": fixation_visible,
            "noise_level": self._noise_level_for_task(task),
        }

    def _noise_level_for_task(self, task: str) -> float:
        if self.noise_level is not None:
            return float(self.noise_level)
        if task == "object_recognition":
            return self.object_recognition_noise_level
        return self.training_noise_level

    def _make_occluders(self, rng: np.random.Generator) -> list[dict[str, Any]]:
        height, width = self.output_size
        occluders = []
        for _ in range(self.occluder_count):
            size_w = int(rng.integers(self.occluder_min_size, self.occluder_max_size + 1))
            size_h = int(rng.integers(self.occluder_min_size, self.occluder_max_size + 1))
            x0 = int(rng.integers(0, max(1, width - size_w)))
            y0 = int(rng.integers(0, max(1, height - size_h)))
            shade = int(rng.integers(15, 235))
            alpha = int(rng.integers(190, 256))
            occluders.append(
                {
                    "shape": str(rng.choice(["ellipse", "rectangle"])),
                    "bbox": (x0, y0, x0 + size_w, y0 + size_h),
                    "fill": (shade, shade, shade, alpha),
                }
            )
        return occluders

    def _render(self, spec: dict[str, Any], rng: np.random.Generator) -> torch.Tensor:
        height, width = self.output_size
        noise_level = spec["noise_level"]
        noise = self._make_noise(rng, noise_level)

        video = torch.full(
            (self.seq_len, 3, height, width),
            self.background,
            dtype=torch.float32,
            device=self.device,
        )
        if self.noise_type and not self.noise_on_top:
            video = torch.clamp(video + noise, 0.0, 1.0)

        transformed_by_object = self._transform_rendered_objects(spec)
        for obj_idx in spec["render_order"]:
            obj_idx = int(obj_idx)
            if obj_idx not in transformed_by_object:
                continue
            visible_times, transformed = transformed_by_object[obj_idx]
            rgb = transformed[:, :3]
            alpha = torch.clamp(transformed[:, 3:4], 0.0, 1.0)
            time_index = torch.as_tensor(visible_times, dtype=torch.long, device=self.device)
            video[time_index] = alpha * rgb + (1.0 - alpha) * video[time_index]

        if spec["fixation_visible"] is not None:
            self._draw_fixations(video, spec["fixation_positions"], spec["fixation_visible"])

        if spec["occluders"]:
            self._apply_occluders(video, spec["occluders"])

        if self.noise_type and self.noise_on_top:
            video = torch.clamp(video + noise, 0.0, 1.0)

        return video

    def _transform_rendered_objects(self, spec: dict[str, Any]) -> dict[int, tuple[np.ndarray, torch.Tensor]]:
        if not self._sprites_have_common_shape:
            transformed_by_object = {}
            for obj_idx in spec["render_order"]:
                obj_idx = int(obj_idx)
                visible_times = np.nonzero(spec["visible"][:, obj_idx])[0]
                if visible_times.size > 0:
                    transformed_by_object[obj_idx] = (
                        visible_times,
                        self._transform_object(spec["objects"][obj_idx], visible_times),
                    )
            return transformed_by_object

        batch_imgs = []
        batch_positions = []
        batch_rotations = []
        batch_scales = []
        spans: dict[int, tuple[np.ndarray, int, int]] = {}

        for obj_idx in spec["render_order"]:
            obj_idx = int(obj_idx)
            visible_times = np.nonzero(spec["visible"][:, obj_idx])[0]
            if visible_times.size == 0:
                continue
            obj = spec["objects"][obj_idx]
            start = len(batch_imgs)
            batch_imgs.extend(self.sprites[obj.class_id] for _ in range(len(visible_times)))
            batch_positions.append(obj.positions[visible_times])
            batch_rotations.append(obj.rotations[visible_times])
            batch_scales.append(obj.scales[visible_times])
            spans[obj_idx] = (visible_times, start, len(batch_imgs))

        if not batch_imgs:
            return {}

        imgs = torch.stack(batch_imgs, dim=0)
        positions = torch.as_tensor(
            np.concatenate(batch_positions, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        rotations = torch.as_tensor(
            np.concatenate(batch_rotations, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        scales = torch.as_tensor(
            np.concatenate(batch_scales, axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        transformed = self._batch_apply_transform(imgs, positions, rotations, scales)
        return {
            obj_idx: (visible_times, transformed[start:end])
            for obj_idx, (visible_times, start, end) in spans.items()
        }

    def _transform_object(self, obj: _ObjectSpec, time_indices: np.ndarray) -> torch.Tensor:
        sprite = self.sprites[obj.class_id]
        batch_size = len(time_indices)
        batch_img = sprite.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        positions = torch.as_tensor(
            obj.positions[time_indices],
            dtype=torch.float32,
            device=self.device,
        )
        rotations = torch.as_tensor(
            obj.rotations[time_indices],
            dtype=torch.float32,
            device=self.device,
        )
        scales = torch.as_tensor(
            obj.scales[time_indices],
            dtype=torch.float32,
            device=self.device,
        )
        return self._batch_apply_transform(batch_img, positions, rotations, scales)

    def _batch_apply_transform(
        self,
        imgs: torch.Tensor,
        positions: torch.Tensor,
        rotations: torch.Tensor,
        scales: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpriteVideoDataset-style affine transforms to a sprite batch."""
        batch_size = imgs.shape[0]
        _, _, img_h, img_w = imgs.shape
        center_x, center_y = img_w / 2.0, img_h / 2.0

        angles_rad = rotations * (math.pi / 180.0)
        cos_angles = torch.cos(angles_rad)
        sin_angles = torch.sin(angles_rad)

        transform_matrices = torch.zeros(batch_size, 2, 3, dtype=imgs.dtype, device=imgs.device)
        transform_matrices[:, 0, 0] = cos_angles * scales
        transform_matrices[:, 0, 1] = -sin_angles * scales
        transform_matrices[:, 1, 0] = sin_angles * scales
        transform_matrices[:, 1, 1] = cos_angles * scales
        transform_matrices[:, 0, 2] = positions[:, 0] - (
            center_x * cos_angles * scales - center_y * sin_angles * scales
        )
        transform_matrices[:, 1, 2] = positions[:, 1] - (
            center_x * sin_angles * scales + center_y * cos_angles * scales
        )

        return K.geometry.transform.warp_affine(
            imgs,
            transform_matrices,
            dsize=self.output_size,
            align_corners=True,
        )

    def _draw_fixations(
        self,
        video: torch.Tensor,
        positions: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        height, width = self.output_size
        fill = torch.tensor([1.0, 245.0 / 255.0, 40.0 / 255.0], dtype=video.dtype, device=video.device)
        outline = torch.tensor([20.0 / 255.0, 20.0 / 255.0, 20.0 / 255.0], dtype=video.dtype, device=video.device)
        half = self.fixation_size / 2.0

        for t in np.nonzero(visible)[0]:
            x, y = float(positions[t, 0]), float(positions[t, 1])
            x0 = max(0, int(math.floor(x - half)))
            x1 = min(width, int(math.ceil(x + half)))
            y0 = max(0, int(math.floor(y - half)))
            y1 = min(height, int(math.ceil(y + half)))
            if x1 <= x0 or y1 <= y0:
                continue
            video[int(t), :, y0:y1, x0:x1] = fill.view(3, 1, 1)
            video[int(t), :, y0, x0:x1] = outline.view(3, 1)
            video[int(t), :, y1 - 1, x0:x1] = outline.view(3, 1)
            video[int(t), :, y0:y1, x0] = outline.view(3, 1)
            video[int(t), :, y0:y1, x1 - 1] = outline.view(3, 1)

    def _apply_occluders(self, video: torch.Tensor, occluders: list[dict[str, Any]]) -> None:
        height, width = self.output_size
        for occluder in occluders:
            x0, y0, x1, y1 = occluder["bbox"]
            x0 = max(0, min(width, int(x0)))
            x1 = max(0, min(width, int(x1)))
            y0 = max(0, min(height, int(y0)))
            y1 = max(0, min(height, int(y1)))
            if x1 <= x0 or y1 <= y0:
                continue

            fill = occluder["fill"]
            color = torch.tensor(fill[:3], dtype=video.dtype, device=video.device).view(1, 3, 1, 1) / 255.0
            alpha = float(fill[3]) / 255.0
            region = video[:, :, y0:y1, x0:x1]
            blended = alpha * color + (1.0 - alpha) * region

            if occluder["shape"] == "ellipse":
                ys, xs = torch.meshgrid(
                    torch.arange(y0, y1, dtype=video.dtype, device=video.device),
                    torch.arange(x0, x1, dtype=video.dtype, device=video.device),
                    indexing="ij",
                )
                cx = (x0 + x1 - 1.0) / 2.0
                cy = (y0 + y1 - 1.0) / 2.0
                rx = max((x1 - x0) / 2.0, 1e-6)
                ry = max((y1 - y0) / 2.0, 1e-6)
                mask = (((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0).view(1, 1, y1 - y0, x1 - x0)
                region.copy_(torch.where(mask, blended, region))
            else:
                region.copy_(blended)

    def _make_noise(self, rng: np.random.Generator, noise_level: float) -> torch.Tensor:
        height, width = self.output_size
        shape = (self.seq_len, height, width, 3)
        if not self.noise_type or noise_level <= 0.0:
            return torch.zeros((self.seq_len, 3, height, width), dtype=torch.float32, device=self.device)
        base_shape = (1, height, width, 3) if self.freeze_noise else shape
        if self.noise_type == "gaussian":
            noise = rng.normal(0.0, noise_level, size=base_shape).astype(np.float32)
        elif self.noise_type == "salt_pepper":
            samples = rng.random(base_shape, dtype=np.float32)
            noise = (samples < noise_level / 2.0).astype(np.float32)
            noise -= (samples > 1.0 - noise_level / 2.0).astype(np.float32)
        else:
            raise RuntimeError(f"Unhandled noise type: {self.noise_type}")
        if self.freeze_noise:
            noise = np.repeat(noise, self.seq_len, axis=0)
        return torch.from_numpy(noise.transpose(0, 3, 1, 2)).to(self.device)

    def _make_labels(self, spec: dict[str, Any]) -> dict[str, Any]:
        object_count = spec["object_count"]
        target_class = spec["target_class"]
        prompt_class = spec["prompt_class"]
        prompt_given = prompt_class != NO_PROMPT_CLASS
        task_info = torch.tensor(
            [spec["task_id"], target_class, int(prompt_given)],
            dtype=torch.long,
        )

        object_class = np.full(self.max_objects, -1, dtype=np.int64)
        object_mask = np.zeros(self.max_objects, dtype=bool)
        is_target = np.zeros(self.max_objects, dtype=bool)
        positions = np.full((self.seq_len, self.max_objects, 2), np.nan, dtype=np.float32)
        rotations = np.full((self.seq_len, self.max_objects), np.nan, dtype=np.float32)
        scales = np.full((self.seq_len, self.max_objects), np.nan, dtype=np.float32)
        visible = np.zeros((self.seq_len, self.max_objects), dtype=bool)
        velocities = np.full((self.max_objects, 2), np.nan, dtype=np.float32)
        speeds = np.full(self.max_objects, np.nan, dtype=np.float32)
        angular_speeds = np.full(self.max_objects, np.nan, dtype=np.float32)
        render_order = np.full(self.max_objects, -1, dtype=np.int64)

        for idx, obj in enumerate(spec["objects"]):
            object_class[idx] = obj.class_id
            object_mask[idx] = True
            is_target[idx] = idx == spec["target_index"]
            positions[:, idx] = obj.positions
            rotations[:, idx] = obj.rotations
            scales[:, idx] = obj.scales
            visible[:, idx] = spec["visible"][:, idx]
            velocities[idx] = obj.velocity
            speeds[idx] = obj.speed
            angular_speeds[idx] = obj.angular_speed

        render_order[: len(spec["render_order"])] = np.asarray(spec["render_order"], dtype=np.int64)

        return {
            "task_info": task_info,
            "task": spec["task"],
            "task_id": torch.tensor(spec["task_id"], dtype=torch.long),
            "popout_mode": spec["popout_mode"],
            "popout_mode_id": torch.tensor(spec["popout_mode_id"], dtype=torch.long),
            "target_class": torch.tensor(target_class, dtype=torch.long),
            "target_class_dense": torch.full((self.seq_len,), target_class, dtype=torch.long),
            "prompt_class": torch.tensor(prompt_class, dtype=torch.long),
            "prompt_class_dense": torch.full((self.seq_len,), prompt_class, dtype=torch.long),
            "prompt_given": torch.tensor(prompt_given, dtype=torch.bool),
            "object_count": torch.tensor(object_count, dtype=torch.long),
            "object_class": torch.from_numpy(object_class),
            "object_mask": torch.from_numpy(object_mask),
            "is_target": torch.from_numpy(is_target),
            "positions": torch.from_numpy(positions),
            "rotations": torch.from_numpy(rotations),
            "scales": torch.from_numpy(scales),
            "visible": torch.from_numpy(visible),
            "velocities": torch.from_numpy(velocities),
            "speeds": torch.from_numpy(speeds),
            "angular_speeds": torch.from_numpy(angular_speeds),
            "render_order": torch.from_numpy(render_order),
            "cue_frames": torch.tensor(self.cue_frames, dtype=torch.long),
            "noise_level": torch.tensor(spec["noise_level"], dtype=torch.float32),
        }


__all__ = [
    "MovingAnimalAttentionDataset",
    "ID_TO_TASK",
    "NO_PROMPT_CLASS",
    "TASK_TO_ID",
    "POPOUT_MODE_TO_ID",
]
