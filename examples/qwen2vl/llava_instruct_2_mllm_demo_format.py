import json
import os
import stat

llava_json_path = "./data/llava_instruct_150k.json"
mllm_format_json_path = "./data/mllm_format_llava_instruct_data.json"

with open(llava_json_path, "r") as f:
    info_json = json.loads(f.read())

mllm_format_llava_instruct_data = []
for item in info_json:
    if "image" not in item or "conversations" not in item:
        raise KeyError(f"key 'image' or 'conversations' not found in this dict")
    img_path = os.path.join("./data/COCO2017/train2017", item["image"])
    print(f"img_path: {img_path}")
    if not os.path.exists(img_path):
        continue
    new_item = {
        "images": [img_path],
        "messages": []
    }

    for i, trun in enumerate(item["conversations"]):
        if "from" not in trun or "value" not in trun:
            raise KeyError(f"key 'from' or 'value' not found in this dict")
        if trun["from"] == "human":
            new_item["messages"].append({"role": "user", "content": trun["value"]})
        elif trun["from"] == "gpt":
            new_item["messages"].append({"role": "assistant", "content": trun["value"]})
        else:
            raise ValueError(f"unknown role: {trun['from']}")
    mllm_format_llava_instruct_data.append(new_item)

output_json = json.dumps(mllm_format_llava_instruct_data)
with os.fdopen(os.open(mllm_format_json_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IWUSR | stat.S_IRUSR), "w") as f:
    f.write(output_json)
print("finish converting dataset")
