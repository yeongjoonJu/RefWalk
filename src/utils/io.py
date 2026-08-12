import json

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            # ensure_ascii=False: 한글 깨짐 방지
            json_record = json.dumps(entry, ensure_ascii=False)
            f.write(json_record + '\n')