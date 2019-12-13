# 读取初始数据
with open('results.txt', 'r') as fpr:
    content = fpr.read()
data = ['a','b','c','d']
# 提取关键数据
with open("mAP.txt", "w") as f:
    f.writelines(content)
