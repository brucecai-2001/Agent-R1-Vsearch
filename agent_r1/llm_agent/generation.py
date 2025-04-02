"""
Tool generation manager for LLM agents
"""

import torch
import re
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import random

from .tensor_helper import TensorHelper, TensorConfig
from agent_r1.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_tool_response_length: int  # Renamed from max_obs_length
    num_gpus: int
    # use_parallel_tool_calls: bool = False
    use_batch_tool_calls: bool = False  # New option for batch execution
    tool_call_start: str = "<tool_call>"
    tool_call_end: str = "</tool_call>"
    tool_response_start: str = "<tool_response>"
    tool_response_end: str = "</tool_response>"
    tool_custom_response_template: str = ""
    
class ToolGenerationManager:
    """Manager for handling LLM tool-based generation and interaction"""
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,  # Renamed
            max_start_length=config.max_start_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_str) -> Tuple[List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        def process_single_response(resp):
            # Look for tool call pattern: <tool_call>tool_name(args)</tool_call>
            tool_pattern = r'<tool_call>(.*?)</tool_call>'
            match = re.search(tool_pattern, resp, re.DOTALL)
            
            if not match:
                return resp + self.tokenizer.eos_token, False  # No tool call found
            
            resp = resp.split(self.config.tool_call_end)[0] + self.config.tool_call_end
            # tool_content = match.group(0)
            
            # Replace all subsequent answer tag pairs with their content
            # rest_of_string = resp[match.end():]
            # cleaned_rest = re.sub(r'<tool_call>(.*?)</tool_call>', r'\1', rest_of_string, flags=re.DOTALL)
            
            return resp + self.tokenizer.eos_token, True
        
        # Process each response string
        return [process_single_response(resp)[0] for resp in responses_str], [process_single_response(resp)[1] for resp in responses_str]

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        # Extract the first tool call from each response
        responses_str, active_masks = self._process_tool_call(responses_str)
        
        return responses_str, torch.tensor(active_masks, dtype=torch.bool)
    
    def _process_tool_responses(self, tool_responses: List[str]) -> torch.Tensor:
        """Process tool responses to token ids"""
        
        tool_responses_ids = self.tokenizer(
            tool_responses, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if tool_responses_ids.shape[1] > self.config.max_tool_response_length:
            print("[WARNING] TOOL RESPONSE TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            tool_responses_ids = tool_responses_ids[:, :self.config.max_tool_response_length]
            
        return tool_responses_ids
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]  # Extract observation from (observation, reward, done, info)
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses
    
    def _execute_tool_calls_batch(self, response_strs: List[str], 
                                 envs: List[ToolEnv], 
                                 active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls in batch for tools that support batch operations."""
        # Convert torch tensor to list of booleans
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Filter active environments and responses
        active_envs = []
        active_responses = []
        active_indices = []
        
        for i, (env, resp, active) in enumerate(zip(envs, response_strs, active_list)):
            if active:
                active_envs.append(env)
                active_responses.append(resp)
                active_indices.append(i)
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        
        if not active_envs:
            return tool_responses
            
        # Use the independent step_batch function for active environments
        batch_results = step_batch(active_envs, active_responses)
        
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
            else:
                tool_response = result[0]  # Extract observation from (observation, reward, done, info)
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
        return tool_responses
    
    def _update_rolling_state(self, rollings, cur_responses: List[str], 
                            tool_responses: List[str]) -> Dict:
        """Update rolling state with new responses and observations."""

        responses = [cur_response + tool_response for cur_response, tool_response in zip(cur_responses, tool_responses)]
        responses_ids = self._batch_tokenize(responses)

        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []
        for raw_prompt_id, response in zip(raw_prompt_ids, responses):
            if len(response) > 0:
                new_raw_prompt_ids.append(raw_prompt_id + self.tokenizer.encode(response, add_special_tokens=False))
            else:
                new_raw_prompt_ids.append(raw_prompt_id)
        
        new_raw_prompt_ids = np.array(new_raw_prompt_ids, dtype=object)
        
        return DataProto.from_single_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:],
            'raw_prompt_ids': new_raw_prompt_ids
        })

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: List[str],
                          tool_responses: List[str]) -> Dict:
        """Update right side state."""
        responses = [cur_response + tool_response for cur_response, tool_response in zip(cur_responses, tool_responses)]
        responses_ids = self._batch_tokenize(responses)

        responses = self.tensor_fn.concatenate_with_padding([
            right_side['responses'],
            responses_ids
        ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None,
                    initial_input_ids: torch.Tensor = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            active_batch = {k: v[active_mask] for k, v in rollings.batch.items()}
            active_non_tensor_batch = {}

            for k, v in rollings.non_tensor_batch.items():
                try:
                    # Try direct boolean indexing first
                    active_non_tensor_batch[k] = v[active_mask.numpy()]
                except (TypeError, ValueError, IndexError):
                    # Fall back to explicit indexing if direct indexing fails
                    active_indices = torch.where(active_mask)[0].tolist()
                    active_non_tensor_batch[k] = np.array([v[i] for i in active_indices], dtype=object)
            
            rollings_active = DataProto.from_dict(tensors=active_batch, non_tensors=active_non_tensor_batch)
            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            meta_info = gen_output.meta_info
            responses_str, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_str = self.tensor_fn._example_level_pad(responses_str, active_mask)

            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses = self._execute_tool_calls(responses_str, envs, active_mask)

            active_num_list.append(active_mask.sum().item())
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_str,
                tool_responses
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_str,
                tool_responses
            )
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        original_right_side['turns'] = turns
        
        # Save trajectory and return final output
        return self._compose_final_output(original_left_side, original_right_side, meta_info)


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output