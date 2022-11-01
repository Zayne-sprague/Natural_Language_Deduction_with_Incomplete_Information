import pandas as pd
import numpy as np

# scores = pd.read_csv('../coverage_scores.csv')
scores = pd.read_csv('../3b_coverage.csv')

recovered_premises = {}
total_steps = 0

for (idx, row) in scores.iterrows():
    filename = row['File Name']

    scores = [
        row['Valid? (Zayne)'],
        row['Valid? (Greg)'],
        row['Valid? (Kaj)']
    ]

    y = 0
    n = 0

    for score in scores:
        if not isinstance(score, str):
            continue
        # if '*' in score or '?' in score:
        #     continue

        invalids = ['L', 'N']
        invalid = any([x in score for x in invalids])

        valids = ['Y', 'R']
        valid = any([x in score for x in valids])

        if invalid:
            n += 1
        elif valid:
            y += 1
        else:
            n += 1

    valid = 0
    if y > n:
        valid = 1
    if y == 0 and n == 0:
        continue

    filename_score = recovered_premises.get(filename, {'valid': 0, 'total': 0})
    filename_score['valid'] += valid
    filename_score['total'] += 1
    recovered_premises[filename] = filename_score
    total_steps += 1

print(f'total steps: {total_steps}')

for k, v in recovered_premises.items():
    print('----')
    print(f'{k}')
    print(f'\ttotal: {v["total"]} | percent of total {v["total"]/total_steps * 100:.02f}%')
    print(f'\tvalid: {v["valid"]} | percent of total {v["valid"]/total_steps * 100:.02f}% | percent of file {v["valid"]/v["total"] * 100:.02f}%')
    print('----')



