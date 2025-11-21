import inspect
from trl.experimental.gold import GOLDTrainer

with open("gold_log_source.py", "w") as f:
    f.write(inspect.getsource(GOLDTrainer.log))
