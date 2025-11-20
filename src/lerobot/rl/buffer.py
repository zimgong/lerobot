#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from collections.abc import Callable, Sequence
from contextlib import suppress
from multiprocessing import shared_memory
from typing import TypedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGE, REWARD
from lerobot.utils.transition import Transition


def _is_distributed() -> bool:
    """Check if distributed training is enabled."""
    return dist.is_available() and dist.is_initialized()


def _get_distributed_info() -> tuple[int, int]:
    """
    Get rank and world_size for distributed training.
    Returns (rank, world_size) or (0, 1) if not distributed.
    """
    if _is_distributed():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


class BatchTransition(TypedDict):
    state: dict[str, torch.Tensor]
    action: torch.Tensor
    reward: torch.Tensor
    next_state: dict[str, torch.Tensor]
    done: torch.Tensor
    truncated: torch.Tensor
    complementary_info: dict[str, torch.Tensor | float | int] | None = None


def random_crop_vectorized(images: torch.Tensor, output_size: tuple) -> torch.Tensor:
    """
    Perform a per-image random crop over a batch of images in a vectorized way.
    (Same as shown previously.)
    """
    B, C, H, W = images.shape  # noqa: N806
    crop_h, crop_w = output_size

    if crop_h > H or crop_w > W:
        raise ValueError(
            f"Requested crop size ({crop_h}, {crop_w}) is bigger than the image size ({H}, {W})."
        )

    tops = torch.randint(0, H - crop_h + 1, (B,), device=images.device)
    lefts = torch.randint(0, W - crop_w + 1, (B,), device=images.device)

    rows = torch.arange(crop_h, device=images.device).unsqueeze(0) + tops.unsqueeze(1)
    cols = torch.arange(crop_w, device=images.device).unsqueeze(0) + lefts.unsqueeze(1)

    rows = rows.unsqueeze(2).expand(-1, -1, crop_w)  # (B, crop_h, crop_w)
    cols = cols.unsqueeze(1).expand(-1, crop_h, -1)  # (B, crop_h, crop_w)

    images_hwcn = images.permute(0, 2, 3, 1)  # (B, H, W, C)

    # Gather pixels
    cropped_hwcn = images_hwcn[torch.arange(B, device=images.device).view(B, 1, 1), rows, cols, :]
    # cropped_hwcn => (B, crop_h, crop_w, C)

    cropped = cropped_hwcn.permute(0, 3, 1, 2)  # (B, C, crop_h, crop_w)
    return cropped


def random_shift(images: torch.Tensor, pad: int = 4):
    """Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
    _, _, h, w = images.shape
    images = F.pad(input=images, pad=(pad, pad, pad, pad), mode="replicate")
    return random_crop_vectorized(images=images, output_size=(h, w))


class SharedMemoryManager:
    """
    Helper class to manage cross-process shared memory blocks for ReplayBuffer.

    This class handles creating shared memory blocks on rank 0 and attaching to them
    on other ranks, allowing multiple processes to share the same buffer data.
    """

    def __init__(self, buffer_name: str = "replay_buffer"):
        """
        Initialize shared memory manager.

        Args:
            buffer_name: Unique identifier for this buffer (used in shared memory names)
        """
        self.buffer_name = buffer_name
        self.shared_memories = {}  # Dict mapping tensor names to SharedMemory objects

    def _get_shared_name(self, tensor_name: str) -> str:
        """Generate a unique shared memory name for a tensor."""
        return f"{self.buffer_name}_{tensor_name}"

    def create_shared_tensor(
        self, tensor_name: str, shape: tuple, dtype: torch.dtype, rank: int
    ) -> torch.Tensor:
        """
        Create a shared memory tensor. On rank 0, creates the shared memory block.
        On other ranks, attaches to the existing shared memory block.

        Args:
            tensor_name: Name identifier for this tensor (e.g., "states_image", "actions")
            shape: Shape of the tensor
            dtype: Data type of the tensor
            rank: Current process rank

        Returns:
            torch.Tensor backed by shared memory
        """
        # Convert torch dtype to numpy dtype
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        np_dtype = dtype_map.get(dtype)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Calculate size in bytes
        size_bytes = int(np.prod(shape) * np.dtype(np_dtype).itemsize)
        shared_name = self._get_shared_name(tensor_name)

        if rank == 0:
            # Rank 0: Create shared memory block
            try:
                shm = shared_memory.SharedMemory(create=True, size=size_bytes, name=shared_name)
                self.shared_memories[tensor_name] = shm
            except FileExistsError:
                shm = shared_memory.SharedMemory(name=shared_name)
                shm.close()
                shm.unlink()
                shm = shared_memory.SharedMemory(create=True, size=size_bytes, name=shared_name)
                self.shared_memories[tensor_name] = shm

            # Create numpy array backed by shared memory
            np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

            # Convert to torch tensor (shares the same memory via numpy)
            # torch.from_numpy creates a view that shares memory with numpy array
            # We need to keep references to keep shared memory alive
            tensor = torch.from_numpy(np_array)
            # Store references to keep shared memory alive
            tensor._shared_memory_np = np_array
            tensor._shared_memory_shm = shm
        else:
            # Other ranks: Attach to existing shared memory block
            shm = shared_memory.SharedMemory(name=shared_name)
            self.shared_memories[tensor_name] = shm

            # Create numpy array backed by shared memory
            np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

            # Convert to torch tensor (shares the same memory via numpy)
            tensor = torch.from_numpy(np_array)
            # Store references to keep shared memory alive
            tensor._shared_memory_np = np_array
            tensor._shared_memory_shm = shm

        return tensor

    def broadcast_tensor_metadata(
        self, tensor_metadata: dict[str, tuple[tuple, torch.dtype]], rank: int
    ) -> dict[str, tuple[tuple, torch.dtype]]:
        """
        Broadcast tensor metadata (shapes and dtypes) from rank 0 to other ranks.

        Args:
            tensor_metadata: Dict mapping tensor names to (shape, dtype) tuples
            rank: Current process rank

        Returns:
            Dict mapping tensor names to (shape, dtype) tuples (same on all ranks)
        """
        if not dist.is_initialized():
            return tensor_metadata

        if rank == 0:
            # Rank 0: Broadcast metadata
            # Convert to list of (name, shape, dtype_int) for broadcasting
            metadata_list = []
            for name, (shape, dtype) in tensor_metadata.items():
                # Convert dtype to int for broadcasting
                metadata_list.append((name, shape, dtype))

            # Broadcast using dist.broadcast_object_list
            dist.broadcast_object_list([metadata_list], src=0)
        else:
            # Other ranks: Receive metadata
            metadata_list = [None]
            dist.broadcast_object_list(metadata_list, src=0)
            metadata_list = metadata_list[0]

            # Convert back to dict
            tensor_metadata = {}
            for name, shape, dtype in metadata_list:
                tensor_metadata[name] = (shape, dtype)

        return tensor_metadata

    def attach_to_shared_tensor(
        self, tensor_name: str, shape: tuple, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Attach to an existing shared memory tensor (for other ranks).
        This assumes the shared memory block already exists (created by rank 0).

        Args:
            tensor_name: Name identifier for this tensor
            shape: Shape of the tensor
            dtype: Data type of the tensor

        Returns:
            torch.Tensor backed by shared memory
        """
        # Convert torch dtype to numpy dtype
        dtype_map = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.uint8: np.uint8,
            torch.bool: np.bool_,
        }
        np_dtype = dtype_map.get(dtype)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        shared_name = self._get_shared_name(tensor_name)

        # Attach to existing shared memory block
        shm = shared_memory.SharedMemory(name=shared_name)
        self.shared_memories[tensor_name] = shm

        # Create numpy array backed by shared memory
        np_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)

        # Convert to torch tensor (shares the same memory via numpy)
        tensor = torch.from_numpy(np_array)
        # Store references to keep shared memory alive
        tensor._shared_memory_np = np_array
        tensor._shared_memory_shm = shm

        return tensor

    def cleanup(self, rank: int):
        """
        Clean up shared memory blocks. Only rank 0 should unlink.

        Args:
            rank: Current process rank
        """
        for tensor_name, shm in self.shared_memories.items():
            shm.close()
            if rank == 0:
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
        self.shared_memories.clear()


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        preprocessor: None = None,
        buffer_name: str = "replay_buffer",
    ):
        """
        Replay buffer for storing transitions.
        It will allocate tensors on the specified device, when the first transition is added.
        NOTE: If you encounter memory issues, you can try to use the `optimize_memory` flag to save memory or
        and use the `storage_device` flag to store the buffer on a different device.
        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
            device (str): The device where the tensors will be moved when sampling ("cuda:0" or "cpu").
            state_keys (List[str]): The list of keys that appear in `state` and `next_state`.
            image_augmentation_function (Optional[Callable]): A function that takes a batch of images
                and returns a batch of augmented images. If None, a default augmentation function is used.
            use_drq (bool): Whether to use the default DRQ image augmentation style, when sampling in the buffer.
            storage_device: The device (e.g. "cpu" or "cuda:0") where the data will be stored.
                Using "cpu" can help save GPU memory.
            optimize_memory (bool): If True, optimizes memory by not storing duplicate next_states when
                they can be derived from states. This is useful for large datasets where next_state[i] = state[i+1].
            preprocessor (Optional[Callable]): A function that takes a batch of states and returns a batch of preprocessed states.
            buffer_name (str): The name of the replay buffer for shared memory identification.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0.")

        self.capacity = capacity
        self.device = device
        self.storage_device = storage_device

        # Initialize shared memory manager for distributed training
        rank, world_size = _get_distributed_info()
        use_shared_memory = world_size > 1 and storage_device == torch.device("cpu")
        
        if use_shared_memory:
            # Use a fixed buffer ID - all ranks must use the same ID to share memory
            # In practice, each training run typically has one buffer, so this is fine
            # If multiple buffers are needed, this could be made a parameter
            self.shared_memory_manager = SharedMemoryManager(buffer_name=buffer_name)
            # Use shared memory for position and size
            self.position = self.shared_memory_manager.create_shared_tensor(
                "position", (1,), torch.long, rank
            )
            self.size = self.shared_memory_manager.create_shared_tensor(
                "size", (1,), torch.long, rank
            )
        else:
            self.shared_memory_manager = None
            self.position = 0
            self.size = 0

        self.initialized = False
        self.optimize_memory = optimize_memory
        self.preprocessor = preprocessor

        # Track episode boundaries for memory optimization
        if use_shared_memory:
            self.episode_ends = self.shared_memory_manager.create_shared_tensor(
                "episode_ends", (capacity,), torch.bool, rank
            )
        else:
            self.episode_ends = torch.zeros(capacity, dtype=torch.bool, device=storage_device)

        # If no state_keys provided, default to an empty list
        self.state_keys = state_keys if state_keys is not None else []

        self.image_augmentation_function = image_augmentation_function

        if image_augmentation_function is None:
            base_function = functools.partial(random_shift, pad=4)
            self.image_augmentation_function = torch.compile(base_function)
        self.use_drq = use_drq

    def _sync_metadata(
        self, 
        state: dict[str, torch.Tensor] | None = None, 
        action: torch.Tensor | None = None,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Sync metadata between ranks."""
        rank, _ = _get_distributed_info()

        # Collect metadata from existing tensors for broadcasting
        tensor_metadata = {}
        for key, tensor in self.states.items():
            tensor_metadata[f"states_{key}"] = (tuple(tensor.shape), tensor.dtype)
        if hasattr(self, 'actions') and self.actions is not None:
            tensor_metadata["actions"] = (tuple(self.actions.shape), self.actions.dtype)
        if hasattr(self, 'rewards') and self.rewards is not None:
            tensor_metadata["rewards"] = (tuple(self.rewards.shape), self.rewards.dtype)
        if not self.optimize_memory and hasattr(self, 'next_states'):
            for key, tensor in self.next_states.items():
                tensor_metadata[f"next_states_{key}"] = (tuple(tensor.shape), tensor.dtype)
        if hasattr(self, 'dones') and self.dones is not None:
            tensor_metadata["dones"] = (tuple(self.dones.shape), self.dones.dtype)
        if hasattr(self, 'truncateds') and self.truncateds is not None:
            tensor_metadata["truncateds"] = (tuple(self.truncateds.shape), self.truncateds.dtype)
        if hasattr(self, 'complementary_info') and self.complementary_info:
            for key, tensor in self.complementary_info.items():
                tensor_metadata[f"complementary_info_{key}"] = (tuple(tensor.shape), tensor.dtype)
        # Broadcast metadata so other ranks can attach
        self.shared_memory_manager.broadcast_tensor_metadata(tensor_metadata, rank)

    def _initialize_storage(
        self,
        state: dict[str, torch.Tensor] | None = None,
        action: torch.Tensor | None = None,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """Initialize the storage tensors based on the first transition."""
        rank, _ = _get_distributed_info()
        use_shared_memory = self.shared_memory_manager is not None

        # Pre-allocate tensors for storage (using shared memory if available)
        if use_shared_memory:
            if rank == 0:
                # Not initialized yet - proceed with normal initialization
                # Determine shapes from the first transition
                state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
                action_shape = action.squeeze(0).shape

                # Initialize storage for complementary_info first (needed for metadata collection)
                self.has_complementary_info = complementary_info is not None
                self.complementary_info_keys = []
                self.complementary_info = {}
                if self.has_complementary_info:
                    self.complementary_info_keys = list(complementary_info.keys())
                # Rank 0: Create shared memory for all storage tensors
                self.states = {
                    key: self.shared_memory_manager.create_shared_tensor(
                        f"states_{key}", (self.capacity, *shape), torch.float32, rank
                    )
                    for key, shape in state_shapes.items()
                }
                self.actions = self.shared_memory_manager.create_shared_tensor(
                    "actions", (self.capacity, *action_shape), torch.float32, rank
                )
                self.rewards = self.shared_memory_manager.create_shared_tensor(
                    "rewards", (self.capacity,), torch.float32, rank
                )

                if not self.optimize_memory:
                    # Standard approach: store states and next_states separately
                    self.next_states = {
                        key: self.shared_memory_manager.create_shared_tensor(
                            f"next_states_{key}", (self.capacity, *shape), torch.float32, rank
                        )
                        for key, shape in state_shapes.items()
                    }
                else:
                    # Memory-optimized approach: don't allocate next_states buffer
                    # Just create a reference to states for consistent API
                    self.next_states = self.states  # Just a reference for API consistency

                self.dones = self.shared_memory_manager.create_shared_tensor(
                    "dones", (self.capacity,), torch.bool, rank
                )
                self.truncateds = self.shared_memory_manager.create_shared_tensor(
                    "truncateds", (self.capacity,), torch.bool, rank
                )

                # Create complementary_info shared memory if needed
                if self.has_complementary_info:
                    for key, value in complementary_info.items():
                        if isinstance(value, torch.Tensor):
                            value_shape = value.squeeze(0).shape
                            self.complementary_info[key] = self.shared_memory_manager.create_shared_tensor(
                                f"complementary_info_{key}", (self.capacity, *value_shape), torch.float32, rank
                            )
                        elif isinstance(value, (int | float)):
                            # Handle scalar values similar to reward
                            self.complementary_info[key] = self.shared_memory_manager.create_shared_tensor(
                                f"complementary_info_{key}", (self.capacity,), torch.float32, rank
                            )
                        else:
                            raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

                # Collect ALL metadata for broadcasting (including complementary_info)
                tensor_metadata = {}
                for key in state_shapes:
                    tensor_metadata[f"states_{key}"] = ((self.capacity, *state_shapes[key]), torch.float32)
                tensor_metadata["actions"] = ((self.capacity, *action_shape), torch.float32)
                tensor_metadata["rewards"] = ((self.capacity,), torch.float32)
                if not self.optimize_memory:
                    for key in state_shapes:
                        tensor_metadata[f"next_states_{key}"] = ((self.capacity, *state_shapes[key]), torch.float32)
                tensor_metadata["dones"] = ((self.capacity,), torch.bool)
                tensor_metadata["truncateds"] = ((self.capacity,), torch.bool)

                # Add complementary_info metadata
                if self.has_complementary_info:
                    for key, value in complementary_info.items():
                        if isinstance(value, torch.Tensor):
                            value_shape = value.squeeze(0).shape
                            tensor_metadata[f"complementary_info_{key}"] = (
                                (self.capacity, *value_shape),
                                torch.float32,
                            )
                        elif isinstance(value, (int | float)):
                            tensor_metadata[f"complementary_info_{key}"] = ((self.capacity,), torch.float32)
                dist.barrier()
                tensor_metadata = self.shared_memory_manager.broadcast_tensor_metadata(tensor_metadata, rank)
            else:
                # For non-main ranks: receive metadata from rank 0
                dist.barrier()  # Wait for rank 0 to potentially initialize
                tensor_metadata = self.shared_memory_manager.broadcast_tensor_metadata({}, rank)
                # Extract state_shapes from metadata for iteration
                state_shapes = {}
                for tensor_name, (shape, dtype) in tensor_metadata.items():
                    if tensor_name.startswith("states_"):
                        key = tensor_name[len("states_"):]
                        # Extract the shape without capacity dimension
                        state_shapes[key] = shape[1:]  # Remove capacity dimension

                # Initialize storage for complementary_info from metadata
                self.has_complementary_info = any(
                    name.startswith("complementary_info_") for name in tensor_metadata.keys()
                )
                self.complementary_info_keys = []
                self.complementary_info = {}
                if self.has_complementary_info:
                    self.complementary_info_keys = [
                        name[len("complementary_info_"):]
                        for name in tensor_metadata.keys()
                        if name.startswith("complementary_info_")
                    ]

                # Now attach to shared memory using received metadata
                self.states = {}
                for key in state_shapes:
                    tensor_name = f"states_{key}"
                    if tensor_name in tensor_metadata:
                        shape, dtype = tensor_metadata[tensor_name]
                        self.states[key] = self.shared_memory_manager.attach_to_shared_tensor(
                            tensor_name, shape, dtype
                        )

                if "actions" in tensor_metadata:
                    shape, dtype = tensor_metadata["actions"]
                    self.actions = self.shared_memory_manager.attach_to_shared_tensor("actions", shape, dtype)

                if "rewards" in tensor_metadata:
                    shape, dtype = tensor_metadata["rewards"]
                    self.rewards = self.shared_memory_manager.attach_to_shared_tensor("rewards", shape, dtype)

                if not self.optimize_memory:
                    self.next_states = {}
                    for key in state_shapes:
                        tensor_name = f"next_states_{key}"
                        if tensor_name in tensor_metadata:
                            shape, dtype = tensor_metadata[tensor_name]
                            self.next_states[key] = self.shared_memory_manager.attach_to_shared_tensor(
                                tensor_name, shape, dtype
                            )
                else:
                    # Memory-optimized approach: reference to states
                    self.next_states = self.states

                if "dones" in tensor_metadata:
                    shape, dtype = tensor_metadata["dones"]
                    self.dones = self.shared_memory_manager.attach_to_shared_tensor("dones", shape, dtype)

                if "truncateds" in tensor_metadata:
                    shape, dtype = tensor_metadata["truncateds"]
                    self.truncateds = self.shared_memory_manager.attach_to_shared_tensor("truncateds", shape, dtype)

                # Attach to complementary_info shared memory if it exists
                if self.has_complementary_info:
                    for key in self.complementary_info_keys:
                        tensor_name = f"complementary_info_{key}"
                        if tensor_name in tensor_metadata:
                            shape, dtype = tensor_metadata[tensor_name]
                            self.complementary_info[key] = self.shared_memory_manager.attach_to_shared_tensor(
                                tensor_name, shape, dtype
                            )
        else:
            # Determine shapes from the first transition
            state_shapes = {key: val.squeeze(0).shape for key, val in state.items()}
            action_shape = action.squeeze(0).shape

            # Initialize storage for complementary_info first (needed for metadata collection)
            self.has_complementary_info = complementary_info is not None
            self.complementary_info_keys = []
            self.complementary_info = {}
            if self.has_complementary_info:
                self.complementary_info_keys = list(complementary_info.keys())
            # Regular tensor allocation
            self.states = {
                key: torch.empty((self.capacity, *shape), device=self.storage_device)
                for key, shape in state_shapes.items()
            }
            self.actions = torch.empty((self.capacity, *action_shape), device=self.storage_device)
            self.rewards = torch.empty((self.capacity,), device=self.storage_device)

            if not self.optimize_memory:
                # Standard approach: store states and next_states separately
                self.next_states = {
                    key: torch.empty((self.capacity, *shape), device=self.storage_device)
                    for key, shape in state_shapes.items()
                }
            else:
                # Memory-optimized approach: don't allocate next_states buffer
                # Just create a reference to states for consistent API
                self.next_states = self.states  # Just a reference for API consistency

            self.dones = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)
            self.truncateds = torch.empty((self.capacity,), dtype=torch.bool, device=self.storage_device)

            # Regular tensor allocation for complementary_info
            if self.has_complementary_info:
                for key, value in complementary_info.items():
                    if isinstance(value, torch.Tensor):
                        value_shape = value.squeeze(0).shape
                        self.complementary_info[key] = torch.empty(
                            (self.capacity, *value_shape), device=self.storage_device
                        )
                    elif isinstance(value, (int | float)):
                        # Handle scalar values similar to reward
                        self.complementary_info[key] = torch.empty((self.capacity,), device=self.storage_device)
                    else:
                        raise ValueError(f"Unsupported type {type(value)} for complementary_info[{key}]")

        self.initialized = True

    def _get_size(self) -> int:
        """Get buffer size as integer, handling both tensor and int cases."""
        if isinstance(self.size, torch.Tensor):
            return int(self.size.item())
        return int(self.size)

    def _get_position(self) -> int:
        """Get buffer position as integer, handling both tensor and int cases."""
        if isinstance(self.position, torch.Tensor):
            return int(self.position.item())
        return int(self.position)

    def __len__(self):
        return self._get_size()

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        """
        Saves a transition, ensuring tensors are stored on the designated storage device.
        In distributed mode, only rank 0 adds transitions; other ranks skip this operation.
        """
        rank, world_size = _get_distributed_info()
        
        # In distributed mode, only rank 0 should add transitions
        # Other ranks should have already attached to shared memory during initialization
        if world_size > 1 and rank != 0:
            # Ensure we're initialized (should have attached to shared memory already)
            if not self.initialized:
                # This shouldn't happen - if it does, it means initialization failed
                raise RuntimeError(
                    "Other ranks must attach to shared memory before add() is called. "
                    "This usually means _initialize_storage() was not called on other ranks."
                )
            return

        # Initialize storage if this is the first transition
        if not self.initialized:
            self._initialize_storage(state=state, action=action, complementary_info=complementary_info)

        # Get current position (handles both tensor and int)
        current_pos = self._get_position()

        # Store the transition in pre-allocated tensors
        for key in self.states:
            self.states[key][current_pos].copy_(state[key].squeeze(dim=0))

            if not self.optimize_memory:
                # Only store next_states if not optimizing memory
                self.next_states[key][current_pos].copy_(next_state[key].squeeze(dim=0))

        self.actions[current_pos].copy_(action.squeeze(dim=0))
        self.rewards[current_pos] = reward
        self.dones[current_pos] = done
        self.truncateds[current_pos] = truncated

        # Handle complementary_info if provided and storage is initialized
        if complementary_info is not None and self.has_complementary_info:
            # Store the complementary_info
            for key in self.complementary_info_keys:
                if key in complementary_info:
                    value = complementary_info[key]
                    if isinstance(value, torch.Tensor):
                        self.complementary_info[key][current_pos].copy_(value.squeeze(dim=0))
                    elif isinstance(value, (int | float)):
                        self.complementary_info[key][current_pos] = value

        # Update position and size (handles both tensor and int)
        new_position = (current_pos + 1) % self.capacity
        new_size = min(self._get_size() + 1, self.capacity)
        
        if isinstance(self.position, torch.Tensor):
            self.position[0] = new_position
            self.size[0] = new_size
        else:
            self.position = new_position
            self.size = new_size

    def sample(self, batch_size: int) -> BatchTransition:
        """Sample a random batch of transitions and collate them into batched tensors."""
        if not self.initialized:
            raise RuntimeError("Cannot sample from an empty buffer. Add transitions first.")

        current_size = self._get_size()
        batch_size = min(batch_size, current_size)
        high = max(0, current_size - 1) if self.optimize_memory and current_size < self.capacity else current_size

        # Handle distributed sampling with shared memory
        rank, world_size = _get_distributed_info()
        device = torch.device(f"cuda:{rank}")
        if world_size > 1:
            # Global index sampler approach for shared memory buffers
            # Calculate N on all ranks to ensure tensor size consistency
            N = batch_size * world_size
            N = min(N, high)
            
            # All ranks must create tensors of the same size for broadcast to work
            if rank == 0:
                # Sample without replacement for better diversity
                if N < high:
                    idx_global = torch.randperm(high, device=device)[:N]
                else:
                    # If we need all indices, just use arange
                    idx_global = torch.arange(high, device=device)
            else:
                # Other ranks: placeholder that will be overwritten by broadcast
                # Must match the size that rank 0 will create
                idx_global = torch.zeros(N, dtype=torch.long, device=device)

            # Broadcast indices from rank 0 to all ranks
            dist.broadcast(idx_global, src=0)

            # Each rank takes every world_size-th index starting from its rank
            # This ensures non-overlapping samples: rank 0 gets indices
            # [0, world_size, 2*world_size, ...]
            # rank 1 gets [1, world_size+1, 2*world_size+1, ...], etc.
            idx_rank = idx_global[rank::world_size]

            # Ensure we don't exceed batch_size
            if len(idx_rank) > batch_size:
                idx_rank = idx_rank[:batch_size]

            # Move indices to storage device
            idx = idx_rank.to(self.storage_device)
        else:
            # Random indices for sampling - create on the same device as storage
            idx = torch.randint(low=0, high=high, size=(batch_size,), device=self.storage_device)

        # Identify image keys that need augmentation
        image_keys = [k for k in self.states if k.startswith(OBS_IMAGE)] if self.use_drq else []

        # Create batched state and next_state
        batch_state = {}
        batch_next_state = {}

        # First pass: load all state tensors to target device
        for key in self.states:
            batch_state[key] = self.states[key][idx]
            if key not in image_keys:
                batch_state[key] = batch_state[key].to(self.device)

            if not self.optimize_memory:
                # Standard approach - load next_states directly
                batch_next_state[key] = self.next_states[key][idx]
                if key not in image_keys:
                    batch_next_state[key] = batch_next_state[key].to(self.device)
            else:
                # Memory-optimized approach - get next_state from the next index
                next_idx = (idx + 1) % self.capacity
                batch_next_state[key] = self.states[key][next_idx]
                if key not in image_keys:
                    batch_next_state[key] = batch_next_state[key].to(self.device)

        # Apply image augmentation in a batched way if needed
        if self.use_drq and image_keys:
            # Concatenate all images from state and next_state
            all_images = []
            for key in image_keys:
                all_images.append(batch_state[key])
                all_images.append(batch_next_state[key])

            # Optimization: Batch all images and apply augmentation once
            all_images_tensor = torch.cat(all_images, dim=0)
            augmented_images = self.image_augmentation_function(all_images_tensor)

            # Split the augmented images back to their sources
            for i, key in enumerate(image_keys):
                # Calculate offsets for the current image key:
                # For each key, we have 2*batch_size images (batch_size for states, batch_size for next_states)
                # States start at index i*2*batch_size and take up batch_size slots
                batch_state[key] = augmented_images[i * 2 * batch_size : (i * 2 + 1) * batch_size]
                # Next states start after the states at index (i*2+1)*batch_size and also take up batch_size slots
                batch_next_state[key] = augmented_images[(i * 2 + 1) * batch_size : (i + 1) * 2 * batch_size]

        # Sample other tensors
        batch_actions = self.actions[idx].to(self.device)
        batch_rewards = self.rewards[idx].to(self.device)
        batch_dones = self.dones[idx].to(self.device).float()
        batch_truncateds = self.truncateds[idx].to(self.device).float()

        # Sample complementary_info if available
        batch_complementary_info = None
        if self.has_complementary_info:
            batch_complementary_info = {}
            for key in self.complementary_info_keys:
                batch_complementary_info[key] = self.complementary_info[key][idx].to(self.device)

        return BatchTransition(
            state=batch_state,
            action=batch_actions,
            reward=batch_rewards,
            next_state=batch_next_state,
            done=batch_dones,
            truncated=batch_truncateds,
            complementary_info=batch_complementary_info,
        )

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2,
    ):
        """
        Creates an infinite iterator that yields batches of transitions.
        Will automatically restart when internal iterator is exhausted.

        Args:
            batch_size (int): Size of batches to sample
            async_prefetch (bool): Whether to use asynchronous prefetching with threads (default: True)
            queue_size (int): Number of batches to prefetch (default: 2)

        Yields:
            BatchTransition: Batched transitions
        """
        while True:  # Create an infinite loop
            if async_prefetch:
                # Get the standard iterator
                iterator = self._get_async_iterator(queue_size=queue_size, batch_size=batch_size)
            else:
                iterator = self._get_naive_iterator(batch_size=batch_size, queue_size=queue_size)

            # Yield all items from the iterator
            with suppress(StopIteration):
                yield from iterator

    def _get_async_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Create an iterator that continuously yields prefetched batches in a
        background thread. The design is intentionally simple and avoids busy
        waiting / complex state management.

        Args:
            batch_size (int): Size of batches to sample.
            queue_size (int): Maximum number of prefetched batches to keep in
                memory.

        Yields:
            BatchTransition: A batch sampled from the replay buffer.
        """
        import queue
        import threading

        data_queue: queue.Queue = queue.Queue(maxsize=queue_size)
        shutdown_event = threading.Event()

        def producer() -> None:
            """Continuously put sampled batches into the queue until shutdown."""
            while not shutdown_event.is_set():
                try:
                    batch = self.sample(batch_size)
                    if self.preprocessor is not None:
                        batch['state'] = self.preprocessor(batch['state'])
                        batch['next_state'] = self.preprocessor(batch['next_state'])
                    # The timeout ensures the thread unblocks if the queue is full
                    # and the shutdown event gets set meanwhile.
                    data_queue.put(batch, block=True, timeout=0.5)
                except queue.Full:
                    # Queue is full – loop again (will re-check shutdown_event)
                    continue
                except Exception:
                    # Surface any unexpected error and terminate the producer.
                    shutdown_event.set()

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        try:
            while not shutdown_event.is_set():
                try:
                    yield data_queue.get(block=True)
                except Exception:
                    # If the producer already set the shutdown flag we exit.
                    if shutdown_event.is_set():
                        break
        finally:
            shutdown_event.set()
            # Drain the queue quickly to help the thread exit if it's blocked on `put`.
            while not data_queue.empty():
                _ = data_queue.get_nowait()
            # Give the producer thread a bit of time to finish.
            producer_thread.join(timeout=1.0)

    def _get_naive_iterator(self, batch_size: int, queue_size: int = 2):
        """
        Creates a simple non-threaded iterator that yields batches.

        Args:
            batch_size (int): Size of batches to sample
            queue_size (int): Number of initial batches to prefetch

        Yields:
            BatchTransition: Batch transitions
        """
        import collections

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(batch_size)
                queue.append(data)

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    @classmethod
    def from_lerobot_dataset(
        cls,
        lerobot_dataset: LeRobotDataset,
        device: str = "cuda:0",
        state_keys: Sequence[str] | None = None,
        capacity: int | None = None,
        image_augmentation_function: Callable | None = None,
        use_drq: bool = True,
        storage_device: str = "cpu",
        optimize_memory: bool = False,
        buffer_name: str = "replay_buffer",
    ) -> "ReplayBuffer":
        """
        Convert a LeRobotDataset into a ReplayBuffer.

        Args:
            lerobot_dataset (LeRobotDataset): The dataset to convert.
            device (str): The device for sampling tensors. Defaults to "cuda:0".
            state_keys (Sequence[str] | None): The list of keys that appear in `state` and `next_state`.
            capacity (int | None): Buffer capacity. If None, uses dataset length.
            action_mask (Sequence[int] | None): Indices of action dimensions to keep.
            image_augmentation_function (Callable | None): Function for image augmentation.
                If None, uses default random shift with pad=4.
            use_drq (bool): Whether to use DrQ image augmentation when sampling.
            storage_device (str): Device for storing tensor data. Using "cpu" saves GPU memory.
            optimize_memory (bool): If True, reduces memory usage by not duplicating state data.

        Returns:
            ReplayBuffer: The replay buffer with dataset transitions.
        """
        if capacity is None:
            capacity = len(lerobot_dataset)

        if capacity < len(lerobot_dataset):
            raise ValueError(
                "The capacity of the ReplayBuffer must be greater than or equal to the length of the LeRobotDataset."
            )

        # In distributed mode, only rank 0 loads data; other ranks wait
        rank, world_size = _get_distributed_info()

        if world_size > 1 and rank != 0:
            dist.barrier()

        # Create replay buffer with image augmentation and DrQ settings
        replay_buffer = cls(
            capacity=capacity,
            device=device,
            state_keys=state_keys,
            image_augmentation_function=image_augmentation_function,
            use_drq=use_drq,
            storage_device=storage_device,
            optimize_memory=optimize_memory,
            buffer_name=buffer_name,
        )

        if world_size > 1 and rank == 0:
            dist.barrier()
        
        if world_size > 1 and rank == 0:
            # Rank 0: Load all data
            # Convert dataset to transitions
            list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

            # Initialize the buffer with the first transition to set up storage tensors
            if list_transition:
                first_transition = list_transition[0]
                first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
                first_action = first_transition[ACTION].to(device)

                # Get complementary info if available
                first_complementary_info = None
                if (
                    "complementary_info" in first_transition
                    and first_transition["complementary_info"] is not None
                ):
                    first_complementary_info = {
                        k: v.to(device) for k, v in first_transition["complementary_info"].items()
                    }

                replay_buffer._initialize_storage(
                    state=first_state, action=first_action, complementary_info=first_complementary_info
                )

            # Fill the buffer with all transitions
            for data in list_transition:
                for k, v in data.items():
                    if isinstance(v, dict):
                        for key, tensor in v.items():
                            v[key] = tensor.to(storage_device)
                    elif isinstance(v, torch.Tensor):
                        data[k] = v.to(storage_device)

                action = data[ACTION]

                replay_buffer.add(
                    state=data["state"],
                    action=action,
                    reward=data["reward"],
                    next_state=data["next_state"],
                    done=data["done"],
                    truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
                    complementary_info=data.get("complementary_info", None),
                )
        elif world_size > 1:
            # This will attach to shared memory blocks created by rank 0
            replay_buffer._initialize_storage()
        else:
            # Non-distributed: Load data normally
            # Convert dataset to transitions
            list_transition = cls._lerobotdataset_to_transitions(dataset=lerobot_dataset, state_keys=state_keys)

            # Initialize the buffer with the first transition to set up storage tensors
            if list_transition:
                first_transition = list_transition[0]
                first_state = {k: v.to(device) for k, v in first_transition["state"].items()}
                first_action = first_transition[ACTION].to(device)

                # Get complementary info if available
                first_complementary_info = None
                if (
                    "complementary_info" in first_transition
                    and first_transition["complementary_info"] is not None
                ):
                    first_complementary_info = {
                        k: v.to(device) for k, v in first_transition["complementary_info"].items()
                    }

                replay_buffer._initialize_storage(
                    state=first_state, action=first_action, complementary_info=first_complementary_info
                )

            # Fill the buffer with all transitions
            for data in list_transition:
                for k, v in data.items():
                    if isinstance(v, dict):
                        for key, tensor in v.items():
                            v[key] = tensor.to(storage_device)
                    elif isinstance(v, torch.Tensor):
                        data[k] = v.to(storage_device)

                action = data[ACTION]

                replay_buffer.add(
                    state=data["state"],
                    action=action,
                    reward=data["reward"],
                    next_state=data["next_state"],
                    done=data["done"],
                    truncated=False,  # NOTE: Truncation are not supported yet in lerobot dataset
                    complementary_info=data.get("complementary_info", None),
                )

        return replay_buffer

    def to_lerobot_dataset(
        self,
        repo_id: str,
        fps=1,
        root=None,
        task_name="from_replay_buffer",
    ) -> LeRobotDataset:
        """
        Converts all transitions in this ReplayBuffer into a single LeRobotDataset object.
        """
        if self._get_size() == 0:
            raise ValueError("The replay buffer is empty. Cannot convert to a dataset.")

        # Create features dictionary for the dataset
        features = {
            "index": {"dtype": "int64", "shape": [1]},  # global index across episodes
            "episode_index": {"dtype": "int64", "shape": [1]},  # which episode
            "frame_index": {"dtype": "int64", "shape": [1]},  # index inside an episode
            "timestamp": {"dtype": "float32", "shape": [1]},  # for now we store dummy
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        # Add "action"
        sample_action = self.actions[0]
        act_info = guess_feature_info(t=sample_action, name=ACTION)
        features[ACTION] = act_info

        # Add "reward" and "done"
        features[REWARD] = {"dtype": "float32", "shape": (1,)}
        features[DONE] = {"dtype": "bool", "shape": (1,)}

        # Add state keys
        for key in self.states:
            sample_val = self.states[key][0]
            f_info = guess_feature_info(t=sample_val, name=key)
            features[key] = f_info

        # Add complementary_info keys if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                sample_val = self.complementary_info[key][0]
                if isinstance(sample_val, torch.Tensor) and sample_val.ndim == 0:
                    sample_val = sample_val.unsqueeze(0)
                f_info = guess_feature_info(t=sample_val, name=f"complementary_info.{key}")
                features[f"complementary_info.{key}"] = f_info

        # Create an empty LeRobotDataset
        lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=None,
            features=features,
            use_videos=True,
        )

        # Start writing images if needed
        lerobot_dataset.start_image_writer(num_processes=0, num_threads=3)

        # Convert transitions into episodes and frames
        current_size = self._get_size()
        current_position = self._get_position()

        for idx in range(current_size):
            actual_idx = (current_position - current_size + idx) % self.capacity

            frame_dict = {}

            # Fill the data for state keys
            for key in self.states:
                frame_dict[key] = self.states[key][actual_idx].cpu()

            # Fill action, reward, done
            frame_dict[ACTION] = self.actions[actual_idx].cpu()
            frame_dict[REWARD] = torch.tensor([self.rewards[actual_idx]], dtype=torch.float32).cpu()
            frame_dict[DONE] = torch.tensor([self.dones[actual_idx]], dtype=torch.bool).cpu()
            frame_dict["task"] = task_name

            # Add complementary_info if available
            if self.has_complementary_info:
                for key in self.complementary_info_keys:
                    val = self.complementary_info[key][actual_idx]
                    # Convert tensors to CPU
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            val = val.unsqueeze(0)
                        frame_dict[f"complementary_info.{key}"] = val.cpu()
                    # Non-tensor values can be used directly
                    else:
                        frame_dict[f"complementary_info.{key}"] = val

            # Add to the dataset's buffer
            lerobot_dataset.add_frame(frame_dict)

            # If we reached an episode boundary, call save_episode, reset counters
            if self.dones[actual_idx] or self.truncateds[actual_idx]:
                lerobot_dataset.save_episode()

        # Save any remaining frames in the buffer
        if lerobot_dataset.episode_buffer["size"] > 0:
            lerobot_dataset.save_episode()

        lerobot_dataset.stop_image_writer()

        return lerobot_dataset

    @staticmethod
    def _lerobotdataset_to_transitions(
        dataset: LeRobotDataset,
        state_keys: Sequence[str] | None = None,
    ) -> list[Transition]:
        """
        Convert a LeRobotDataset into a list of RL (s, a, r, s', done) transitions.

        Args:
            dataset (LeRobotDataset):
                The dataset to convert. Each item in the dataset is expected to have
                at least the following keys:
                {
                    "action": ...
                    "next.reward": ...
                    "next.done": ...
                    "episode_index": ...
                }
                plus whatever your 'state_keys' specify.

            state_keys (Sequence[str] | None):
                The dataset keys to include in 'state' and 'next_state'. Their names
                will be kept as-is in the output transitions. E.g.
                ["observation.state", "observation.environment_state"].
                If None, you must handle or define default keys.

        Returns:
            transitions (List[Transition]):
                A list of Transition dictionaries with the same length as `dataset`.
        """
        if state_keys is None:
            raise ValueError("State keys must be provided when converting LeRobotDataset to Transitions.")

        transitions = []
        num_frames = len(dataset)

        # Check if the dataset has "next.done" key
        sample = dataset[0]
        has_done_key = DONE in sample

        # Check for complementary_info keys
        complementary_info_keys = [key for key in sample if key.startswith("complementary_info.")]
        has_complementary_info = len(complementary_info_keys) > 0

        # If not, we need to infer it from episode boundaries
        if not has_done_key:
            print("'next.done' key not found in dataset. Inferring from episode boundaries...")

        for i in tqdm(range(num_frames)):
            current_sample = dataset[i]

            # ----- 1) Current state -----
            current_state: dict[str, torch.Tensor] = {}
            for key in state_keys:
                val = current_sample[key]
                current_state[key] = val.unsqueeze(0)  # Add batch dimension

            # ----- 2) Action -----
            action = current_sample[ACTION].unsqueeze(0)  # Add batch dimension

            # ----- 3) Reward and done -----
            reward = float(current_sample[REWARD].item())  # ensure float

            # Determine done flag - use next.done if available, otherwise infer from episode boundaries
            if has_done_key:
                done = bool(current_sample[DONE].item())  # ensure bool
            else:
                # If this is the last frame or if next frame is in a different episode, mark as done
                done = False
                if i == num_frames - 1:
                    done = True
                elif i < num_frames - 1:
                    next_sample = dataset[i + 1]
                    if next_sample["episode_index"] != current_sample["episode_index"]:
                        done = True

            # TODO: (azouitine) Handle truncation (using the same value as done for now)
            truncated = done

            # ----- 4) Next state -----
            # If not done and the next sample is in the same episode, we pull the next sample's state.
            # Otherwise (done=True or next sample crosses to a new episode), next_state = current_state.
            next_state = current_state  # default
            if not done and (i < num_frames - 1):
                next_sample = dataset[i + 1]
                if next_sample["episode_index"] == current_sample["episode_index"]:
                    # Build next_state from the same keys
                    next_state_data: dict[str, torch.Tensor] = {}
                    for key in state_keys:
                        val = next_sample[key]
                        next_state_data[key] = val.unsqueeze(0)  # Add batch dimension
                    next_state = next_state_data

            # ----- 5) Complementary info (if available) -----
            complementary_info = None
            if has_complementary_info:
                complementary_info = {}
                for key in complementary_info_keys:
                    # Strip the "complementary_info." prefix to get the actual key
                    clean_key = key[len("complementary_info.") :]
                    val = current_sample[key]
                    # Handle tensor and non-tensor values differently
                    if isinstance(val, torch.Tensor):
                        complementary_info[clean_key] = val.unsqueeze(0)  # Add batch dimension
                    else:
                        # TODO: (azouitine) Check if it's necessary to convert to tensor
                        # For non-tensor values, use directly
                        complementary_info[clean_key] = val

            # ----- Construct the Transition -----
            transition = Transition(
                state=current_state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                truncated=truncated,
                complementary_info=complementary_info,
            )
            transitions.append(transition)

        return transitions


# Utility function to guess shapes/dtypes from a tensor
def guess_feature_info(t, name: str):
    """
    Return a dictionary with the 'dtype' and 'shape' for a given tensor or scalar value.
    If it looks like a 3D (C,H,W) shape, we might consider it an 'image'.
    Otherwise default to appropriate dtype for numeric.
    """

    shape = tuple(t.shape)
    # Basic guess: if we have exactly 3 dims and shape[0] in {1, 3}, guess 'image'
    if len(shape) == 3 and shape[0] in [1, 3]:
        return {
            "dtype": "image",
            "shape": shape,
        }
    else:
        # Otherwise treat as numeric
        return {
            "dtype": "float32",
            "shape": shape,
        }


def concatenate_batch_transitions(
    left_batch_transitions: BatchTransition, right_batch_transition: BatchTransition
) -> BatchTransition:
    """
    Concatenates two BatchTransition objects into one.

    This function merges the right BatchTransition into the left one by concatenating
    all corresponding tensors along dimension 0. The operation modifies the left_batch_transitions
    in place and also returns it.

    Args:
        left_batch_transitions (BatchTransition): The first batch to concatenate and the one
            that will be modified in place.
        right_batch_transition (BatchTransition): The second batch to append to the first one.

    Returns:
        BatchTransition: The concatenated batch (same object as left_batch_transitions).

    Warning:
        This function modifies the left_batch_transitions object in place.
    """
    # Concatenate state fields
    left_batch_transitions["state"] = {
        key: torch.cat(
            [left_batch_transitions["state"][key], right_batch_transition["state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["state"]
    }

    # Concatenate basic fields
    left_batch_transitions[ACTION] = torch.cat(
        [left_batch_transitions[ACTION], right_batch_transition[ACTION]], dim=0
    )
    left_batch_transitions["reward"] = torch.cat(
        [left_batch_transitions["reward"], right_batch_transition["reward"]], dim=0
    )

    # Concatenate next_state fields
    left_batch_transitions["next_state"] = {
        key: torch.cat(
            [left_batch_transitions["next_state"][key], right_batch_transition["next_state"][key]],
            dim=0,
        )
        for key in left_batch_transitions["next_state"]
    }

    # Concatenate done and truncated fields
    left_batch_transitions["done"] = torch.cat(
        [left_batch_transitions["done"], right_batch_transition["done"]], dim=0
    )
    left_batch_transitions["truncated"] = torch.cat(
        [left_batch_transitions["truncated"], right_batch_transition["truncated"]],
        dim=0,
    )

    # Handle complementary_info
    left_info = left_batch_transitions.get("complementary_info")
    right_info = right_batch_transition.get("complementary_info")

    # Only process if right_info exists
    if right_info is not None:
        # Initialize left complementary_info if needed
        if left_info is None:
            left_batch_transitions["complementary_info"] = right_info
        else:
            # Concatenate each field
            for key in right_info:
                if key in left_info:
                    left_info[key] = torch.cat([left_info[key], right_info[key]], dim=0)
                else:
                    left_info[key] = right_info[key]

    return left_batch_transitions
