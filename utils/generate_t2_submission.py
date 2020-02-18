from typing import List

def generate_t2_sub(preds: List[str]) -> List[str]:
    """ Take a list of prediction and update the TC template
        with those predictions """
    with open("data/dev-task-TC-template.out", "r") as f:
        lines = f.readlines()
    
    final = []
    for i, line in enumerate(lines):
        pred = predictions[i].strip()
        line = line.replace("?", pred)
        final.append(line)
    
    return final

