import json

# 读取原始json
with open('/root/autodl-tmp/Wendell/Data/Json/GB1_train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 分类
test_0 = [x for x in data if x['FITNESS'] == 0]
test_low = [x for x in data if 0 < x['FITNESS'] < 2.920655]
test_high = [x for x in data if x['FITNESS'] >= 2.920655]

# 按要求抽取
test_set = test_0[:20] + test_low[:70] + test_high[:10]

# # 剩下的为训练集
# test_ids = set(id(x) for x in test_set)
# train_set = [x for x in data if id(x) not in test_ids]

# 保存
# with open('/root/autodl-tmp/Wendell/Data/Json/GB1_train.json', 'w', encoding='utf-8') as f:
#     json.dump(train_set, f, ensure_ascii=False, indent=2)
with open('/root/autodl-tmp/Wendell/Data/Json/GB1_test_new.json', 'w', encoding='utf-8') as f:
    json.dump(test_set, f, ensure_ascii=False, indent=2)