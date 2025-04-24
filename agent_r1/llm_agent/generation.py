"""
Tool generation manager for LLM agents
"""

import torch
import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .tensor_helper import TensorHelper, TensorConfig
from agent_r1.tool.tool_env import ToolEnv, step, step_batch
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class ToolGenerationConfig:
    """Configuration for tool-based generation"""
    max_turns: int
    max_prompt_length: int 
    max_response_length: int
    max_response_length_single_turn: int
    max_tool_response_length: int
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
        processor,
        actor_rollout_wg,
        config: ToolGenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        # Ensure tool configuration is properly set
        self.tool_response_start = config.tool_response_start
        self.tool_response_end = config.tool_response_end
        self.tool_call_start = config.tool_call_start
        self.tool_call_end = config.tool_call_end

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_tool_response_length=config.max_tool_response_length,
        ))

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _process_tool_call(self, responses_ids: torch.Tensor, responses_str: List[str]) -> Tuple[torch.Tensor, List[str], List[bool]]:
        """
        Process a list of response strings to extract the first tool call
        while preserving the rest of the string content.
        
        Args:
            responses_str (List[str]): List of response strings potentially containing tool calls
            
        Returns:
            List[str]: Processed responses with only first tool call preserved
        """
        # Define stop strings for truncation
        stop = []
        # Define stop token IDs for truncation (empty list as default)
        stop_ids = [self.tokenizer.encode(self.tool_call_end, add_special_tokens=False)[0]]
        
        def match(response_str):
            tool_pattern = f'{self.tool_call_start}(.*?){self.tool_call_end}'
            match = re.search(tool_pattern, response_str, re.DOTALL)
            return match

        def process_single_response(response_ids: torch.Tensor, stop: List[str], stop_ids: List[int]):
            # Remove padding tokens
            response_ids = response_ids[response_ids != self.tokenizer.pad_token_id]
            
            # Initialize truncation index to the length of the sequence
            truncation_idx = len(response_ids)
            
            # First check for stop token IDs - this maintains token-level alignment
            for i in range(len(response_ids)):
                if response_ids[i].item() in stop_ids:
                    truncation_idx = i + 1
                    break
            
            # For stop strings, we need a more careful approach
            if stop:
                response_tokens = self.tokenizer.convert_ids_to_tokens(response_ids[i])
                response_sofar = ""
                for i, token in enumerate(response_tokens[:truncation_idx]):
                    response_sofar += token
                    if any(stop_str in response_sofar for stop_str in stop):
                        truncation_idx = i + 1
                        break
            
            # Return the truncated sequence
            return torch.cat([response_ids[:truncation_idx], torch.tensor([self.tokenizer.pad_token_id]*(self.config.max_response_length_single_turn - truncation_idx), dtype=torch.long)], dim=0)
        
        # Process each response string
        processed_responses = []
        active_masks = []
        
        for resp_ids, resp_str in zip(responses_ids, responses_str):
            active_mask = match(resp_str) is not None
            active_masks.append(active_mask)

            if not active_mask:
                processed_responses.append(resp_ids)
            else:
                truncated_ids = process_single_response(resp_ids, stop, stop_ids)
                processed_responses.append(truncated_ids)
            
        processed_responses = torch.stack(processed_responses, dim=0)
        processed_responses_str = self.tokenizer.batch_decode(processed_responses, skip_special_tokens=False)
        processed_responses_str = [response.split(self.tokenizer.pad_token)[0] for response in processed_responses_str]
        
        return processed_responses_str, processed_responses, active_masks

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to extract tool calls."""
        responses = responses[:, :self.config.max_response_length_single_turn]
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=False
        )

        responses_str = [response.split(self.tokenizer.pad_token)[0] for response in responses_str]

        # Extract the first tool call from each response
        responses_str, responses_ids, active_masks = self._process_tool_call(responses, responses_str)
        
        return responses_str, responses_ids, torch.tensor(active_masks, dtype=torch.bool)
    
    def _execute_tool_calls(self, response_strs: List[str], 
                          envs: List[ToolEnv], 
                          active_mask: torch.Tensor) -> List[str]:
        """Execute tool calls sequentially and return tool responses."""
        # Convert torch tensor to list of booleans if needed
        active_list = active_mask.tolist() if isinstance(active_mask, torch.Tensor) else active_mask
        
        # Initialize result list with empty strings
        tool_responses = [""] * len(response_strs)
        tool_response_images = [None] * len(response_strs)
        # Process each environment sequentially
        for i, (resp, env, active) in enumerate(zip(response_strs, envs, active_list)):
            if not active:
                continue
                
            # Step the environment using the agent's response
            result = step(env, resp)
            tool_response = result[0]['content']  # Extract observation from (observation, reward, done, info)
            tool_response_images[i] = result[0]['image']
            tool_responses[i] = self.config.tool_custom_response_template.format(tool_response=tool_response)            
        return tool_responses, tool_response_images
    
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
        tool_response_images = [None] * len(response_strs)
        
        if not active_envs:
            return tool_responses, tool_response_images
            
        # Use the independent step_batch function for active environments
        batch_results = step_batch(active_envs, active_responses)
        
        # Map results back to original indices
        for idx, result in zip(active_indices, batch_results):
            if result is None:
                tool_responses[idx] = ""
                tool_response_images[idx] = None
            else:
                tool_response = result[0]['content']  # Extract observation from (observation, reward, done, info)
                tool_responses[idx] = self.config.tool_custom_response_template.format(tool_response=tool_response)
                tool_response_images[idx] = result[0]['image']
        return tool_responses, tool_response_images
    
    def _create_response_action_mask(self, responses_ids_list: List[List[int]], tool_responses_ids_list: List[List[int]]) -> List[List[int]]:
        """
        Create action masks for responses identifying which tokens are from the model vs external.
        
        Args:
            responses_ids_list: List of lists containing model's response token IDs
            tool_responses_ids_list: List of lists containing tool response token IDs
            
        Returns:
            action_masks: List of lists with 1s for model-generated tokens and 0s for external tokens
        """
        action_masks = []
        
        for model_ids, tool_ids in zip(responses_ids_list, tool_responses_ids_list):
            # Create mask: 1 for model tokens, 0 for tool tokens
            action_mask = [1] * len(model_ids) + [0] * len(tool_ids)
            action_masks.append(action_mask)

        return action_masks

    def _update_rolling_state(self, rollings, responses_ids: torch.Tensor, 
                            tool_responses: List[str], tool_responses_images: List[List[Image.Image]]) -> Dict:
        """Update rolling state with new responses and observations.
        rollings : last llm input DataProto
        cur_responses: llm output action 
        tool_responses: tool response text
        tool_responses_images: tool response image 
        """

        is_multi_modal = "multi_modal_data" in rollings.non_tensor_batch.keys()

        row_dict_list = []
        formatted_tool_responses = []
        raw_tool_responses = []
        action_masks = []
        
        for i, (tool_response, tool_responses_image) in enumerate(zip(tool_responses, tool_responses_images)):
            row_dict={}
            if is_multi_modal and '<image>' in tool_response and tool_responses_image is not None:
                assert len(tool_responses_image) == tool_response.count('<image>'), f"[WARNING] TOOL RESPONSE IMAGE NUMBER NOT MATCH, {len(tool_responses_image)} != {tool_response.count('<image>')} for {tool_response}"
                raw_tool_responses.append(tool_response.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>'))
                row_dict['multi_modal_data'] = {'image': tool_responses_image}
                image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
                row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
                image_grid_thw = image_inputs['image_grid_thw']
                if image_grid_thw is not None:
                    merge_length = self.processor.image_processor.merge_size**2
                    index = 0
                    while '<image>' in tool_response:
                        tool_response = tool_response.replace(
                            '<image>',
                            '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                            '<|vision_end|>',
                            1,
                        )
                        index += 1

                    tool_response = tool_response.replace('<|placeholder|>', self.processor.image_token)

            else:
                raw_tool_responses.append(tool_response)
            formatted_tool_responses.append(tool_response)
            row_dict_list.append(row_dict)

        tool_responses_ids = self._batch_tokenize(formatted_tool_responses)

        if "responses" not in rollings.batch.keys():
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)
        else:
            rollings.batch['responses'] = self.tensor_fn.concatenate_with_padding([
                rollings.batch['responses'],
                responses_ids,
                tool_responses_ids
            ], pad_to_left=False)

        rollings.batch['responses'] = rollings.batch['responses'][:, :self.config.max_response_length]

        responses_ids_list = []
        tool_responses_ids_list = []

        for i, (responses_ids_, tool_responses_ids_) in enumerate(zip(responses_ids, tool_responses_ids)):
            responses_ids_ = responses_ids_[responses_ids_ != self.tokenizer.pad_token_id].tolist()
            tool_responses_ids_ = tool_responses_ids_[tool_responses_ids_ != self.tokenizer.pad_token_id].tolist()
            responses_ids_list.append(responses_ids_)
            tool_responses_ids_list.append(tool_responses_ids_)

        action_masks = self._create_response_action_mask(responses_ids_list, tool_responses_ids_list)

        if "action_mask" not in rollings.non_tensor_batch.keys():
            rollings.non_tensor_batch['action_mask'] = np.array(action_masks, dtype=object)
        else:
            new_action_masks = []
            for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
                new_action_masks.append(action_mask + action_masks[i])
            rollings.non_tensor_batch['action_mask'] = np.array(new_action_masks, dtype=object)

        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            responses_ids,
            tool_responses_ids
        ])

        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        
        if is_multi_modal:
            multi_modal_data = rollings.non_tensor_batch['multi_modal_data']
            multi_modal_inputs = rollings.non_tensor_batch['multi_modal_inputs']

            new_multi_modal_data = []
            new_multi_modal_inputs = []

            for row_dict, multi_modal_data_, multi_modal_inputs_ in zip(row_dict_list, multi_modal_data, multi_modal_inputs):
                if 'multi_modal_data' in row_dict.keys():
                    new_multi_modal_data.append({"image":multi_modal_data_['image'] + row_dict['multi_modal_data']['image']})
                else:
                    new_multi_modal_data.append({"image":multi_modal_data_['image']})
                if 'multi_modal_inputs' in row_dict.keys():
                    new_multi_modal_inputs.append({key: torch.cat((val,row_dict['multi_modal_inputs'][key]),dim=0) for key, val in multi_modal_inputs_.items()})
                else:
                    new_multi_modal_inputs.append({key: val for key, val in multi_modal_inputs_.items()})

            rollings.non_tensor_batch['multi_modal_data'] = np.array(new_multi_modal_data, dtype=object)
            rollings.non_tensor_batch['multi_modal_inputs'] = np.array(new_multi_modal_inputs, dtype=object)

            from verl.models.transformers.qwen2_vl import get_rope_index
            new_postion_ids = []
            for i in range(len(new_multi_modal_data)):
                new_postion_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=new_input_ids[i],
                    image_grid_thw=new_multi_modal_inputs[i]['image_grid_thw'],
                    attention_mask=new_attention_mask[i],
                ))

            new_position_ids = torch.stack(new_postion_ids, dim=0)
        else:
            new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        rollings.batch['input_ids'] = new_input_ids
        rollings.batch['position_ids'] = new_position_ids
        rollings.batch['attention_mask'] = new_attention_mask

        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids'].tolist()
        new_raw_prompt_ids = []

        for raw_prompt_id, responses_ids_, raw_tool_response in zip(raw_prompt_ids, responses_ids_list, raw_tool_responses):
            if len(responses_ids_) > 0 or len(raw_tool_response) > 0:
                tool_response_ids = self.tokenizer.encode(raw_tool_response, add_special_tokens=False)
                new_raw_prompt_ids.append(raw_prompt_id + responses_ids_ + tool_response_ids)
            else:
                new_raw_prompt_ids.append(raw_prompt_id)

        rollings.non_tensor_batch['raw_prompt_ids'] = np.array(new_raw_prompt_ids, dtype=object)

        return rollings
    
    def run_llm_loop(self, gen_batch, envs: List[Any] = None) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        batch_size = gen_batch.batch['input_ids'].shape[0]
        
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns = torch.zeros(batch_size, dtype=torch.int32)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        prompts = gen_batch.batch['input_ids'][:, -self.config.max_prompt_length:].clone()

        # Main generation loop
        for _ in range(self.config.max_turns):
            if not active_mask.sum():
                break

            # Check if any sequence exceeds max length
            effective_len = rollings.batch['attention_mask'].sum(dim=1)
            length_exceeded = effective_len > self.config.max_prompt_length

            if length_exceeded.sum() > 0:
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                active_mask[length_exceeded] = 0

            raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
            length_exceeded = [len(prompt_id) > self.config.max_prompt_length for prompt_id in raw_prompt_ids]
            if any(length_exceeded):
                print("[WARNING] SEQUENCE LENGTH EXCEEDED MAX PROMPT LENGTH")
                for prompt_id, length_exceeded_ in zip(raw_prompt_ids, length_exceeded):
                    if length_exceeded_:
                        print(f"[DEBUG] LENGTH {len(prompt_id)}: {self.tokenizer.decode(prompt_id)}")
                active_mask[length_exceeded] = 0
            
            if not active_mask.sum():
                print("[WARNING] NO ACTIVE SEQUENCES")
                break
            
            if hasattr(rollings, 'non_tensor_batch') and rollings.non_tensor_batch:
                # Create active batch with tensor data
                rollings_active = DataProto.from_dict(
                    tensors={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                    non_tensors={
                        k: v[active_mask.numpy()] for k, v in rollings.non_tensor_batch.items()
                    }
                )
            else:
                rollings_active = DataProto.from_dict(
                    batch={
                        k: v[active_mask] for k, v in rollings.batch.items()
                    },
                )

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            responses_str, responses_ids, new_active_masks = self._postprocess_responses(gen_output.batch['responses'])
            responses_str, responses_ids = self.tensor_fn._example_level_pad(responses_str, responses_ids, active_mask)
          
            active_mask[active_mask.clone()] = new_active_masks

            turns[active_mask] += 1

            if self.config.use_batch_tool_calls:
                # Use batch execution for tool calls
                tool_responses, tool_responses_images = self._execute_tool_calls_batch(responses_str, envs, active_mask)
            else:
                # Use sequential execution for tool calls
                tool_responses, tool_responses_images = self._execute_tool_calls(responses_str, envs, active_mask)

            active_num_list.append(active_mask.sum().item())

            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                tool_responses,
                tool_responses_images
            )
 
        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # Compose final output
        final_output = {}
        final_output['turns'] = turns
        final_output['prompts'] = prompts
        final_output['responses'] = rollings.batch['responses']
        final_output['input_ids'] = torch.cat([
            prompts,
            rollings.batch['responses']
        ], dim=1)
        final_output['attention_mask'] = self.tensor_fn.create_attention_mask(final_output['input_ids'])
        if "multi_modal_data" in rollings.non_tensor_batch.keys():
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = []
            for i in range(len(rollings.non_tensor_batch['multi_modal_data'])):
                position_ids.append(get_rope_index(
                    processor=self.processor,
                    input_ids=final_output['input_ids'][i],
                    image_grid_thw=rollings.non_tensor_batch['multi_modal_inputs'][i]['image_grid_thw'],
                    attention_mask=final_output['attention_mask'][i],
                ))

            position_ids = torch.stack(position_ids, dim=0)
            final_output['position_ids'] = position_ids
        else:
            final_output['position_ids'] = self.tensor_fn.create_position_ids(final_output['attention_mask'])

        response_length = final_output['responses'].shape[-1]
        response_mask = final_output['attention_mask'][:, -response_length:]

        final_output['action_mask'] = response_mask.clone()

        for i, action_mask in enumerate(rollings.non_tensor_batch['action_mask']):
            mask_len = min(len(action_mask), response_mask.shape[1])
            final_output['action_mask'][i, :mask_len] = torch.tensor(action_mask[:mask_len]) * response_mask[i, :mask_len]
        
        final_output = DataProto.from_dict(final_output)
        final_output.non_tensor_batch = rollings.non_tensor_batch
        
        return final_output