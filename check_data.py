import pandas as pd
from collections import Counter

df = pd.read_excel('data/output/call_center_analysis')
issues = df['Key Issues'].dropna()

print('Sample Key Issues from Excel:')
for i, val in enumerate(issues.head(10)):
    print(f'{i+1}. {val}')

print('\n\nCategory counts:')
cats = []
for issue in issues:
    if issue and issue != 'N/A':
        cats.extend([c.strip() for c in str(issue).split('|') if c.strip() and c.strip() != 'N/A'])

c = Counter(cats)
print('\n'.join([f'{k}: {v}' for k,v in sorted(c.items(), key=lambda x: x[1], reverse=True)[:25]]))
