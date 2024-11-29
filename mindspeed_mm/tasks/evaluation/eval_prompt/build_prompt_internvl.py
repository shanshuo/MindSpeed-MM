from typing import Callable

from mindspeed_mm.tasks.evaluation.eval_prompt.build_prompt_base import BasePromptTemplate
from mindspeed_mm.tasks.evaluation.eval_datastes.datasets_base import datasets_type

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternvlPromptTemplate(BasePromptTemplate):

    def __init__(self, ):
        super().__init__()

    def build_prompt(self, line, dump_image: Callable, dataset_name=None):
        tgt_path = dump_image(line)

        if datasets_type[dataset_name] == "MCQ":
            prompt = self.build_multi_choice_prompt(line)
        elif datasets_type[dataset_name] == "VQA":
            prompt = line['question'] + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    @staticmethod
    def build_multi_choice_prompt(line):
        return super()._build_prompt(line)

    @staticmethod
    def is_use_custom_prompt(dataset_name):
        if dataset_name in ['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'MMBench-Video', 'Video-MME', 'MVBench',
                            'Video']:
            return False
        else:
            return True
