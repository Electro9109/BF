import re

def regex_parse(text):
    text = text.lower()
    
    # 1. Atmosphere
    co = 0.0
    h2 = 0.0
    n2 = 0.0
    
    co_match = re.search(r'(?:co[=:\s]+|carbon monoxide\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not co_match:
        co_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*co', text)
    if co_match:
        co = float(co_match.group(1))
        
    h2_match = re.search(r'(?:h2[=:\s]+|hydrogen\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not h2_match:
        h2_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*h2', text)
    if h2_match:
        h2 = float(h2_match.group(1))
        
    n2_match = re.search(r'(?:n2[=:\s]+|nitrogen\s*(?:is|at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not n2_match:
        n2_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*n2', text)
    if n2_match:
        n2 = float(n2_match.group(1))
        
    # If N2 is not specified but CO + H2 < 100, N2 is the balance
    if n2 == 0.0 and (co > 0.0 or h2 > 0.0) and (co + h2 < 100.0):
        n2 = 100.0 - co - h2
        
    # 2. Burden
    sinter = 0.0
    ore = 0.0
    pellet = 0.0
    other = 0.0
    
    # Check 100% cases
    if "100% sinter" in text or "completely sinter" in text or "sinter only" in text:
        sinter = 100.0
    elif "100% ore" in text or "completely ore" in text or "ore only" in text:
        ore = 100.0
    elif "100% pellet" in text or "completely pellet" in text or "pellet only" in text:
        pellet = 100.0
        
    # Find percentages
    sinter_match = re.search(r'(?:sinter[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not sinter_match:
        sinter_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*sinter', text)
    if sinter_match and sinter == 0.0:
        sinter = float(sinter_match.group(1))
        
    ore_match = re.search(r'(?:ore[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not ore_match:
        ore_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*ore', text)
    if ore_match and ore == 0.0:
        ore = float(ore_match.group(1))
        
    pellet_match = re.search(r'(?:pellet[s\s]*(?:at)?\s*)(\d+(?:\.\d+)?)%', text)
    if not pellet_match:
        pellet_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*pellet', text)
    if pellet_match and pellet == 0.0:
        pellet = float(pellet_match.group(1))

    # If they sum to 0 but one of them is mentioned, default it to 100 if only that one is mentioned
    mentions = [w for w in ["sinter", "ore", "pellet"] if w in text]
    if sinter + ore + pellet == 0.0 and len(mentions) == 1:
        if mentions[0] == "sinter": sinter = 100.0
        elif mentions[0] == "ore": ore = 100.0
        elif mentions[0] == "pellet": pellet = 100.0
        
    # Test type
    test_type = "SO"
    if sinter > 0 and ore > 0 and pellet > 0:
        test_type = "SOP"
    elif pellet > 0 and sinter == 0 and ore == 0:
        test_type = "P"
    elif sinter > 0 and ore > 0:
        test_type = "SO"
        
    return {
        "CO_pct": co, "H2_pct": h2, "N2_pct": n2,
        "sinter_pct": sinter, "ore_pct": ore, "pellet_pct": pellet, "other_pct": other,
        "test_type": test_type
    }

print("Scenario 1:", regex_parse("Atmosphere is CO=40% and N2=60%. Burden is 100% S1 sinter."))
print("Scenario 2:", regex_parse("CO=40%, H2=8%, N2=52%. Burden is 100% Sinter."))
print("Scenario 3:", regex_parse("CO=40% and N2=60%. Burden is 70% Sinter and 30% Ore."))
print("Scenario 5:", regex_parse("Atmosphere has CO=36%, H2=4%, N2=60%. Burden is 65% Sinter, 25% Ore, 10% Pellet."))
print("Scenario 7:", regex_parse("We run with CO=37%, H2=6.5%, N2=56.5%. Burden is 80% Sinter and 20% Ore."))
print("Scenario 8:", regex_parse("Run CO=34%, H2=6%, N2=60%. Burden is 100% Pellet."))
