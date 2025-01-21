# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps


def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def process_args(parser):
    parser.conflict_handler = "resolve"
    parser = _add_lora_args(parser)
    parser = _add_training_args(parser)
    parser = _add_network_size_args(parser)
    parser = _add_dummy_optimizer_args(parser)
    parser = _add_logging_args(parser)
    return parser


def _add_lora_args(parser):
    group = parser.add_argument_group(title='lora')

    group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                       help='Use lora in target modules.')
    group.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    group.add_argument('--lora-dropout', type=float, default=0.0, help="lora dropout rate")
    group.add_argument('--lora-r', type=int, default=8,
                       help='Lora rank.')
    group.add_argument('--lora-alpha', type=int, default=16,
                       help='Lora alpha.')
    group.add_argument('--lora-register-forward-hook', nargs='+', type=str,
                       default=['word_embeddings', 'input_layernorm', 'final_layernorm'],
                       help='Lora register forward hook.')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--use-deter-comp',
                       action='store_true',
                       default=False,
                       help='Enable deterministic computing for npu')
    group.add_argument('--jit-compile',
                       action='store_true',
                       default=False,
                       help='Setting jit compile mode to True')
    group.add_argument('--allow-tf32',
                       action='store_true',
                       default=False,
                       help='Use tf32 to train')
    group.add_argument('--allow-internal-format',
                       action='store_true',
                       default=False,
                       help='Use internal format to train')

    return parser


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network_size_args')

    group.add_argument('--padded-vocab-size',
                       type=int,
                       default=None,
                       help='set padded vocab size')

    return parser


def _add_dummy_optimizer_args(parser):
    group = parser.add_argument_group(title='dummy optimizer args')

    group.add_argument('--enable-dummy-optimizer',
                       action='store_true',
                       default=False,
                       help='enable dummy optimizer')

    return parser


def _add_logging_args(parser):
    group = parser.add_argument_group(title='mm_logging')

    group.add_argument('--log-tps',
                       action='store_true',
                       default=False,
                       help='calculate and log average tokens per sample')
    
    return parser