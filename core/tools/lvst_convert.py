import json

json_file = open("/content/train_full_labels.json", "r", encoding="utf-8")
txt_file = open("/content/icdar2019_label.txt", "w", encoding="utf-8")

data = json_file.read()
dic = json.loads(data)  # json.loads(str) ; json.load(file)

for key in dic:
    values = dic[key]
    label = "icdar_c4_train_imgs/" + key + ".jpg"+ "\t" + str(values).replace("'", '"') + "\n"
    txt_file.write(str(label))

json_file.close()
txt_file.close()
print("finish")