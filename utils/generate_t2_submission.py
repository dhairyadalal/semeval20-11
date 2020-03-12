from typing import List

def generate_t2_sub(preds: List[str], source: str) -> List[str]:
    """ Take a list of prediction and update the TC template
        with those predictions """
        
    if source == "dev":
        with open("data/dev-task-TC-template.out", "r") as f:
            lines = f.readlines()
    else:
        with open("data/test-task-TC-template.out", "r") as f:
            lines = f.readlines()

    final = []
    for i, line in enumerate(lines):
        pred = preds[i].strip()
        line = line.replace("?", pred)
        final.append(line)
    
    return final

