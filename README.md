# Overview
This repo contains the experiment results and code for SemEval 2020 Task 11 -
Proganda Detection. 

## Useful files
The following data files should be helpful for getting up and running quickly
in order to do modelling:
- `data/task1_data.csv` : contains all the training lines and extracted sequences for Task 1 binary sequence labeling. The BIO annotations were generated with simple white space tokenization.
- `data/task1_dev.csv`: contains all the dev lines and article_ids that need
to be classified for submissio to the dev leaderboard.
- `task_2_data.csv`: contains all the train and dev data for task multiclass 
classification challenge. 

## Useful Utilities
In the utils folder, I've created some common utilities that might be useful.

```
Generate Task 2 formatted submission
    generate_t2_sub(preds: List[str]) -> List[str]

Get Original Article text
    get_article(file: str) -> str:

Get file path given a train article id
    get_task1_file(article_id: int) -> str:

Get file path given a dev article id
    get_task2_file(article_id: int) -> str:
```
