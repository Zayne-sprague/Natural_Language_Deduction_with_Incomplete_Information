import pandas as pd
import numpy as np

scores = pd.read_csv('../eb_scores.csv')
scores = pd.read_csv('../moral_scores.csv')
# scores = pd.read_csv("../3B_R_sv.csv")
scores = pd.read_csv("../3b_step_valid_r.csv")
# scores = pd.read_csv("../3b_step_valid_ur.csv")

valid_steps = {}
total_steps = 0

scores = scores[scores['Step Type'] == 'abductive']
# scores = scores[scores['Step Type'] == 'abduction']

for (idx, row) in scores.iterrows():
    filename = row['File Name']

    scores = [
        row['Valid? (Zayne)'],
        row['Valid? (Greg)'],
        row['Valid? (Kaj)']
    ]

    y = 0
    vnf = 0
    n = 0

    for score in scores:
        if not isinstance(score, str):
            continue
        # if '*' in score or '?' in score:
        #     continue
        if 'Y' in score or 'C' in score:
            y += 1
        elif 'VNF' in score:
            vnf += 1
        else:
            n += 1



    valid = 0
    if y > n and y > vnf:
        valid = 1
    if y == 0 and n == 0 and vnf == 0:
        continue

    filename_score = valid_steps.get(filename, {'valid': 0, 'total': 0, 'y': 0, 'vnf': 0, 'n': 0})

    filename_score['valid'] += valid
    filename_score['total'] += 1

    if y > n and y > vnf:
        filename_score['y'] += 1
    elif vnf > y and vnf > n:
        filename_score['vnf'] += 1
    else:
        filename_score['n'] += 1

    valid_steps[filename] = filename_score
    total_steps += 1

print(f'total steps: {total_steps}')

for k, v in valid_steps.items():
    print('----')
    print(f'{k}')
    print(f'\ttotal: {v["total"]} | percent of total {v["total"]/total_steps * 100:.02f}%')
    print(f'\tvalid: {v["valid"]} | percent of total {v["valid"]/total_steps * 100:.02f}% | percent of file {v["valid"]/v["total"] * 100:.02f}%')
    print('----')



