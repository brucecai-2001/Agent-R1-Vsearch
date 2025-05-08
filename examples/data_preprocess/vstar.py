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
    local_dir = "dataset/vstar"
    val_questions_path = "dataset/vstar/question.jsonl"

    questions = []
    visited = []

    # read the train questions
    q_a = {}

    with open(val_questions_path, 'r', encoding='utf-8') as file:
        for line in file:
            q = json.loads(line)
            category = q['category']
            if category == 'relative_position':
                continue
            
            img_pth = q['image']

            label_name = img_pth.split('.')[0] + '.json'
            label_pth = local_dir + '/' + label_name
            
            visited.append(img_pth.split('/')[1])
            
            with open(label_pth, 'r', encoding='utf-8') as json_file:
                label = json.load(json_file)
                data = {
                            "question": q['text'],
                            "answer": q['label'],
                            "bbox": label['bbox'],
                            "image": local_dir + '/' + img_pth
                        }
                questions.append(data)
    

    # read the train questions
    for root, dirs, files in os.walk(local_dir):
        for dir in dirs:
            subfolder_path = local_dir + '/' + dir
            
            for root, dirs, files in os.walk(subfolder_path):
                for file in files:
                    if file.split('.')[1] == 'json':
                        continue 
                    
                    if file in visited:
                        continue

                    label_name = file.split('.')[0] + '.json'
                    label_pth = subfolder_path + '/' + label_name
                    with open(label_pth, 'r', encoding='utf-8') as json_file:
                        label = json.load(json_file)
                        
                        if dir == 'relative_position' and len(label['bbox']) != 1:
                            continue

                        options = label['options']
                        shuffled_options = options.copy()
                        # 打乱列表
                        random.shuffle(shuffled_options)
                        # 初始化字母
                        letter = ord('A')
                        # 用于存储添加字母后的元素
                        new_list = []
                        # 用于存储字母和元素的映射
                        mapping = {}
                        for item in shuffled_options:
                            new_item = f"\n({chr(letter)}) {item}"
                            new_list.append(new_item)
                            mapping[item] = chr(letter)
                            letter += 1
                        # 找到原列表第一个元素在打乱后列表中的对应字母
                        answer = options[0]
                        answer_label = mapping.get(answer)

                        question = label['question'] + "".join(new_list) + "Answer with the option's letter from the given choices directly."

                        data = {
                            "question": question,
                            "answer": answer_label,
                            "bbox": label['bbox'],
                            "image": subfolder_path + '/' + file
                        }
                
                        questions.append(data)

    train_size = 150
    train_questions = questions[:train_size]
    eval_questions = questions[train_size:]

    # build huggingface dataset
    train_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in train_questions],
            'answer': [item['answer'] for item in train_questions],
            'bbox': [item['bbox'] for item in train_questions],
            'image': [item['image'] for item in train_questions],
            'level': [str(item.get('level', '')) for item in train_questions],
            'type': [str(item.get('type', '')) for item in train_questions]
    })
        
    validation_dataset = datasets.Dataset.from_dict({
            'question': [item['question'] for item in eval_questions],
            'answer': [item['answer'] for item in eval_questions],
            'bbox': [item['bbox'] for item in eval_questions],
            'image': [item['image'] for item in eval_questions],
            'level': [str(item.get('level', '')) for item in eval_questions],
            'type': [str(item.get('type', '')) for item in eval_questions]
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
            img_path = example.pop('image')
            image = Image.open(img_path)
            w, h = image.width, image.height
            instruction_following_ = instruction_following.format(width=w, height=h)
            question_raw = example.pop('question')
            question = instruction_following_ + "Question: " + question_raw
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
                "images": [img_path]
            }
            
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    validation_dataset = validation_dataset.map(function=make_map_fn('validation'), with_indices=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    validation_dataset.to_parquet(os.path.join(local_dir, 'validation.parquet'))