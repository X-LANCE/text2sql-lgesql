import sys
our_file = sys.argv[1]
baseline_file = sys.argv[2]
out_file = 'comparison.txt'

with open(our_file, 'r') as of1, open(baseline_file, 'r') as of2:
    res1, res2 = [], []
    for l in of1:
        l = l.strip()
        if l == '': continue
        res1.append(l)
    for l in of2:
        l = l.strip()
        if l == '': continue
        res2.append(l)
    start, final, cur = 0, [], []
    for idx, (l1, l2) in enumerate(zip(res1, res2)):
        if idx % 4 == 0:
            if cur != []:
                final.append(cur)
            cur = [l1]
        elif idx % 4 == 1:
            cur.append(l1)
        elif idx % 4 == 2:
            cur.append(l1)
            cur.append(l2)
        else:
            if float(l1[l1.index(':')+1:]) > 0.5 and float(l2[l2.index(':')+1:]) < 0.5:
                cur.append(l1)
                cur.append(l2)
                continue
            else:
                cur = []
with open(out_file, 'w') as of:
    for res in final:
        for l in res:
            of.write(l + '\n')
        of.write('\n')
print('In total, %d samples.' % (len(final)))
