# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
import logging

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model, get_vla_model_config_and_processor
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs


class MultiStepRolloutWorker(Worker):
    """Rollout worker for huggingface models with multi-step environment interaction."""

    def __init__(self, cfg: DictConfig):
        """Initialize the rollout worker.

        Args:
            cfg (DictConfig): Configuration for the rollout worker.
        """
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()
        self.actor_group_name = cfg.actor.group_name
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.get_batch_cnt = 0

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            model_config, input_processor = get_vla_model_config_and_processor(
                self.cfg.actor
            )
            self.hf_model.setup_config_and_processor(
                model_config, self.cfg, input_processor
            )

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def load_checkpoint(self, load_path):
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    def load_checkpoint(self, load_path):
        try:
            model_dict = torch.load(load_path)
            self.hf_model.load_state_dict(model_dict)
        except Exception as e:
            logging.warning("Failed to load checkpoint from %s: %s", load_path, e)
            pass

    def setup_sample_params(self):
        """Setup sampling parameters for rollout."""
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self, env_output: dict[str, torch.Tensor], extracted_obs: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                if (
                    final_obs_list is not None
                    and isinstance(final_obs_list, list)
                    and batch_env_sizes is not None
                ):
                    final_values_all = self._compute_bootstrap_values(
                        final_obs_list, batch_env_sizes, dones
                    )
                    rewards[:, -1] += self.cfg.algorithm.gamma * final_values_all

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs
                    )
                    actions, result = self.predict(extracted_obs)
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)

    def get_batch(self, input_channel: Channel, num_group_envs: int):
        """Get a batch of group environment outputs from the input channel."""
        env_outputs: list[EnvOutput] = []
        for _ in range(num_group_envs):
            env_output: EnvOutput = input_channel.get(key=self.get_batch_cnt)
            env_outputs.append(env_output)
        env_batch = EnvOutput.merge_to_batch(env_outputs)
        self.get_batch_cnt += 1
        return env_batch, env_outputs

    def put_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | list | dict,
        env_outputs: list[EnvOutput],
        num_group_envs: int,
    ):
        """Put actions into the output channel to send to the environment.

        It first splits the actions according to the number of environment groups,
        then sends each split action to the corresponding environment based on the env_outputs' worker_rank and stage_id.
        An env ID is also sent along with the action to identify which env the action belongs to.

        Args:
            output_channel (Channel): Channel to send actions to the environment.
            chunk_actions (torch.Tensor | list | dict): Actions to be sent to the environment.
            env_outputs (list[EnvOutput]): List of EnvOutput corresponding to each environment.
            num_group_envs (int): Number of group environment.
        """
        split_actions = EnvOutput.split_value(chunk_actions, split_size=num_group_envs)
        assert len(env_outputs) == num_group_envs, (
            f"Number of env outputs {len(env_outputs)} does not match the num_group_envs per stage in rollout {num_group_envs}"
        )
        for action, env_output in zip(split_actions, env_outputs):
            assert env_output.num_group_envs == 1, (
                "The put_actions should only put the actions for one group env number."
            )
            assert len(env_output.group_env_ids) == 1, (
                "The put_actions should only put the actions for one group env number."
            )
            # The key is (worker_rank, stage_id) to ensure the action is sent to the correct env worker and stage
            # The group env ID is not added to the key but sent as part of the item to avoid creating too many queues in the channel (each key creates a separate queue), because num_group_envs can be large while num_pipeline_stages and env worker size are relatively small.
            output_channel.put(
                item=(env_output.group_env_ids[0], action),
                key=(env_output.worker_rank, env_output.stage_id),
            )

    def put_train_batch(self, actor_channel: Channel, stage_id: int):
        """Put the rollout batch into the actor channel for training."""
        # send rollout_batch to actor
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        # Split only along batch dimension, keep time dimension as-is.
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i])

    def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        """Generate rollout by interacting with the environment.

        Args:
            input_channel (Channel): Channel to receive environment outputs.
            output_channel (Channel): Channel to send actions to the environment.
            actor_channel (Channel): Channel to send rollout data to the actor for training.
        """
        if self.enable_offload:
            self._load_model()
        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )

        self.device_lock.acquire()
        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    # Get env output
                    self.device_lock.release()  # Release lock to allow EnvWorker to run
                    env_batch, env_outputs = self.get_batch(
                        input_channel, self.train_num_group_envs_per_stage
                    )
                    self.device_lock.acquire()  # Re-acquire lock for prediction

                    dones, rewards = self.get_dones_and_rewards(env_batch)

                    # Predict actions
                    with self.worker_timer():
                        actions, result = self.predict(env_batch["obs"])

                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=result["forward_inputs"],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)

                    # Send actions to env
                    self.put_actions(
                        output_channel,
                        actions,
                        env_outputs,
                        self.train_num_group_envs_per_stage,
                    )

            for stage_id in range(self.num_pipeline_stages):
                self.device_lock.release()  # Release lock to allow EnvWorker to run
                env_batch, _ = self.get_batch(
                    input_channel, self.train_num_group_envs_per_stage
                )
                self.device_lock.acquire()  # Re-acquire lock for prediction

                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards = self.get_dones_and_rewards(env_batch)
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].rewards.append(rewards)

                with self.worker_timer():
                    actions, result = self.predict(env_batch["obs"])

                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )

        for stage_id in range(self.num_pipeline_stages):
            self.put_train_batch(actor_channel, stage_id)

        if self.enable_offload:
            self._offload_model()

        self.device_lock.release()

    def evaluate(self, input_channel: Channel, output_channel: Channel):
        """Evaluate the model by interacting with the environment."""
        if self.enable_offload:
            self._load_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_episode_steps
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_batch, env_outputs = self.get_batch(
                        input_channel, self.eval_num_envs_per_stage
                    )
                    actions, _ = self.predict(env_batch["obs"], mode="eval")
                    self.put_actions(
                        output_channel,
                        actions,
                        env_outputs,
                        self.eval_num_envs_per_stage,
                    )

        if self.enable_offload:
            self._offload_model()

    def _offload_model(self):
        """Offload the model to CPU to save GPU memory."""
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def _load_model(self):
        self.hf_model = self.hf_model.to(self.device)
