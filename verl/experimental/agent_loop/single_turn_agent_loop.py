# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import THINK_INTERRUPT_PHRASE
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion.

    Supports think-interrupt (same mechanism as ToolAgentLoop) when
    rollout.multi_turn.thinking_budget is set. In CoT mode the interrupt
    redirects the model from runaway thinking straight to the final answer —
    no tool call step in between.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Think-interrupt. Reuses the same config surface as ToolAgentLoop so the
        # TIR↔CoT ablation uses identical budget semantics. `tool_call_budget` is
        # read for parity but not used here — the post-interrupt budget absorbs
        # the tool_call + tool_response + answer share of the TIR budget so that
        # total response length is identical between modes (fair ablation).
        self.thinking_budget = self.rollout_config.multi_turn.thinking_budget
        self.tool_call_budget = self.rollout_config.multi_turn.tool_call_budget
        if self.thinking_budget is not None:
            self._interrupt_ids: list[int] = self.tokenizer.encode(
                THINK_INTERRUPT_PHRASE, add_special_tokens=False
            )
            self._think_end_id: int = self.tokenizer.convert_tokens_to_ids("</think>")
            # Drift guard — see tool_agent_loop.py for rationale.
            _shell_hint = os.environ.get("INTERRUPT_LEN")
            if _shell_hint is not None:
                assert int(_shell_hint) == len(self._interrupt_ids), (
                    f"INTERRUPT_LEN mismatch: launcher says {_shell_hint}, "
                    f"tokenizer says {len(self._interrupt_ids)}. "
                    f"Update INTERRUPT_LEN in the launcher script to {len(self._interrupt_ids)}."
                )
            self._post_interrupt_budget: int = (
                self.response_length - self.thinking_budget - len(self._interrupt_ids)
            )
            assert self._post_interrupt_budget > 0, (
                f"No room for answer after interrupt: response_length={self.response_length} "
                f"- thinking_budget={self.thinking_budget} "
                f"- interrupt={len(self._interrupt_ids)} "
                f"= {self._post_interrupt_budget}"
            )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        metrics: dict[str, Any] = {}
        request_id = uuid4().hex

        sc1_params = (
            {**sampling_params, "max_tokens": self.thinking_budget}
            if self.thinking_budget is not None
            else sampling_params
        )
        with simple_timer("generate_sequences", metrics):
            out1: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sc1_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = out1.num_preempted if out1.num_preempted is not None else -1

        response_ids: list[int] = list(out1.token_ids)
        response_mask: list[int] = [1] * len(out1.token_ids)
        response_logprobs: list[float] | None = list(out1.log_probs) if out1.log_probs else None
        extra_fields = dict(out1.extra_fields) if out1.extra_fields else {}
        routed_experts = out1.routed_experts

        # Think-interrupt: fire when thinking hit the budget without emitting </think>.
        interrupt_fired = (
            self.thinking_budget is not None
            and len(out1.token_ids) >= self.thinking_budget
            and self._think_end_id not in out1.token_ids
        )
        if interrupt_fired:
            response_ids.extend(self._interrupt_ids)
            response_mask.extend([0] * len(self._interrupt_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(self._interrupt_ids))

            sc2_prompt_ids = list(prompt_ids) + list(out1.token_ids) + list(self._interrupt_ids)
            sc2_params = {**sampling_params, "max_tokens": self._post_interrupt_budget}
            with simple_timer("generate_sequences", metrics):
                out2: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=sc2_prompt_ids,
                    sampling_params=sc2_params,
                    image_data=images,
                    video_data=videos,
                )
            metrics["num_preempted"] = metrics.get("num_preempted", 0) + (out2.num_preempted or 0)

            response_ids.extend(out2.token_ids)
            response_mask.extend([1] * len(out2.token_ids))
            if response_logprobs is not None:
                response_logprobs.extend(
                    out2.log_probs if out2.log_probs else [0.0] * len(out2.token_ids)
                )
            if out2.extra_fields.get("max_global_steps"):
                extra_fields["max_global_steps"] = out2.extra_fields["max_global_steps"]
            if out2.routed_experts is not None:
                routed_experts = out2.routed_experts

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                routed_experts[: len(prompt_ids) + self.response_length]
                if routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=extra_fields,
        )

        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
