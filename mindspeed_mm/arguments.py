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