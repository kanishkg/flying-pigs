import csv

with open("data/human_eval/human_eval - 1-150.csv", "r") as f:
    data = list(csv.reader(f))[1:]
with open("data/human_eval/human_eval - 151-300.csv", "r") as f:
    data += list(csv.reader(f))[1:]
with open("data/human_eval/human_eval - 301-450.csv", "r") as f:
    data += list(csv.reader(f))[1:]
with open("data/human_eval/human_eval - 451-600.csv", "r") as f:
    data += list(csv.reader(f))[1:]

score = 0
total = 0 
human_scores = {
    'human_1': [],
    'human_2': [],
    'human_3': [],
    'human_4': [],
    'human_5': []
    }
human_gt_scores = {
    'human_1': [],
    'human_2': [],
    'human_3': [],
    'human_4': [],
    'human_5': []
    }
for i, d in enumerate(data):
    ground_truth = d[2]
    gt = 1 if ground_truth == "Yes" else 0
    annotations = [int(x == "Yes") for x in d[3:8]]
    human_estimate = int(sum(annotations) > 2) 
    if gt == int(sum(annotations) > 2):
        score += 1
    for j, h in enumerate(human_scores):
        human_scores[h].append(annotations[j]==human_estimate)
        human_gt_scores[h].append(annotations[j]==gt)
        
print(f"human ensemble accuracy: {score/len(data)}")
for h in human_scores:
    print(f"{h} rel accuracy: {sum(human_scores[h])/len(human_scores[h])}")
    print(f"{h} gt accuracy: {sum(human_gt_scores[h])/len(human_gt_scores[h])}")
