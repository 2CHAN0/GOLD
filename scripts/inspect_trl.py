import inspect
from trl.experimental.gold import gold_trainer

print(inspect.getsource(gold_trainer.build_teacher_inputs_from_texts))
