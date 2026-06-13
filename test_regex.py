import re

def parse_burden(val):
    val = str(val).upper()
    res = {'sinter_pct': 0.0, 'ore_pct': 0.0, 'pellet_pct': 0.0, 'other_pct': 0.0, 'num_components': 0}
    
    m_100 = re.match(r'100%\s*([A-Z])', val)
    if m_100:
        letter = m_100.group(1)
        res['num_components'] = 1
        if letter == 'S': res['sinter_pct'] = 100.0
        elif letter == 'O': res['ore_pct'] = 100.0
        elif letter == 'P': res['pellet_pct'] = 100.0
        else: res['other_pct'] = 100.0
        return res
        
    matches = re.findall(r'([A-Z])[A-Z0-9/]*[-=]?(\d+(?:\.\d+)?)%', val)
    for letter, pct in matches:
        pct = float(pct)
        res['num_components'] += 1
        if letter == 'S': res['sinter_pct'] += pct
        elif letter == 'O': res['ore_pct'] += pct
        elif letter == 'P': res['pellet_pct'] += pct
        else: res['other_pct'] += pct
    return res

print(parse_burden('S1-70%+O1-30%'))
print(parse_burden('(S24-70%) +(O20-30%)'))
print(parse_burden('100% O11'))
print(parse_burden('(S2-70%) +(I/O1-10%)+(P1-20%)'))
