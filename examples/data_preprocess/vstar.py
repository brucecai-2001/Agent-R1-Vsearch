"""
    Preprocess the Vstar dataset to parquet format, Only use direct_attributes as train set
"""
import os
import datasets
import json
import random
from PIL import Image

if __name__ == '__main__':
    data_source = "vstar"
    local_dir = "/Users/caixinyu/datasets/vstar"
    questions_path = "/Users/caixinyu/datasets/vstar/question.jsonl"

    # read the questions
    questions = []
    with open(questions_path, 'r', encoding='utf-8') as file:
        for line in file:
            q = json.loads(line)
            category = q['category']
            if category == 'relative_position':
                continue
            
            # read the gt bbox
            label = None
            data_label_path = os.path.join(local_dir, q['image']).replace('.jpg', '.json')
            with open(data_label_path, 'r', encoding='utf-8') as json_file:
                label = json.load(json_file)
        
            # build data
            data = {
                "question": q['text'],
                "answer": q['label'],
                "bbox": label['bbox'],
                "image": os.path.join(local_dir, q['image'])
            }
            questions.append(data)
    
    # split the dataset and shuffle
    data_num = len(questions)
    train_size = 70
    val_size = data_num - train_size

    random.shuffle(questions)
    train_question = questions[:train_size]
    validation_question = questions[train_size:]


    # build huggingface dataset
    train_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in train_question],
            'answer': [item['answer'] for item in train_question],
            'bbox': [item['bbox'] for item in train_question],
            'image': [item['image'] for item in train_question],
            'level': [str(item.get('level', '')) for item in train_question],
            'type': [str(item.get('type', '')) for item in train_question]
    })
        
    validation_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in validation_question],
            'answer': [item['answer'] for item in validation_question],
            'bbox': [item['bbox'] for item in validation_question],
            'image': [item['image'] for item in validation_question],
            'level': [str(item.get('level', '')) for item in validation_question],
            'type': [str(item.get('type', '')) for item in validation_question]
    })


    instruction_following = """You are a helpful multi modal assistant. Given an image, Answer the given question. You can use the tools provided to you to answer the question. You can use the tool as many times as you want.
You must first conduct reasoning inside <think>...</think>. If you need to use the tool, you can use the tool call <tool_call>...</tool_call> to call the tool after <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for tool call:
<think>
...
</think>
<tool_call>
...
</tool_call>

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

The image's width:{width}
The image's height:{height}

<image>
"""

    # Process each data item
    def make_map_fn(split):
        def process_fn(example, idx):
            image = Image.open(example.pop('image'))
            w, h = image.width, image.height
            instruction_following= instruction_following.format(width=w, height=h)
            question_raw = example.pop('question')
            question = instruction_following + "Question: " + question_raw
            answer_raw = example.pop('answer')

            # Convert all data to string format to avoid type issues
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "vsearch",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer_raw,
                    'ground_truth_tools': example.pop('bbox')[0]
                },
                "extra_info": {
                    'split': split,
                    'index': str(idx),
                    'answer': answer_raw,
                    'question': question_raw,
                    'level': str(example.get('level', '')),
                    'type': str(example.get('type', ''))
                },
                "images": [example.pop('image')]
            }
            
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))