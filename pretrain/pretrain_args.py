import dataclasses
from dataclasses import dataclass, field

@dataclass
class PretrainArguments:
    output_dir: str = field(
        default="output/train_kgbart",
        metadata={'help': "The output directory where the model predictions and checkpoints will be written."}
    )

    dataset_dir: str = field(
        default=r"../dataset/pretrain",
        metadata={'help': "The dataset directory."}
    )

    kg_path: str = field(
        default=r"../dataset/kg",
        metadata={'help': "The KG directory."}
    )

    log_dir: str = field(
        default="log/train_kgbart",
        metadata={'help': "The output directory where the log will be written."}
    )

    vocab_save_dir: str = field(
        default=r"../dataset/tokeniazers"
    )

    vocab_root: str = field(
        default=r"../dataset/tokeniazers"
    )

    bart_model: str = field(
        default="facebook/bart-large"
    )

    max_grad_norm: float = field(
        default=1.0
    )

    learning_rate: float = field(
        default=0.00001
    )
    adam_epsilon: float = field(
        default=1e-8
    )

    warmup_steps: int = field(
        default=0
    )

    fp16_opt_level: str = field(
        default="O1"
    )

    from_scratch: str = field(
        default=True
    )

    model_recover_path: str = field(
        default=None
    )

    train_batch_size: int = field(
        default=48,
    )
    eval_batch_size: int = field(
        default=16,
    )

    num_workers: int = field(
        default=5
    )

    num_train_epochs: float = field(
        default=10
    )

    last_pretrained: int = field(
        default=6
    )

    trained_vocab: str = field(
        default=r"../dataset/tokeniazers",
        metadata={'help': "The output directory where the log will be written."}
    )

    local_rank: int = field(
        default=-1,
        metadata={'help': "local_rank for distributed training on gpus"}
    )

    no_cuda: str = field(
        default=False,
        metadata={'help': "Whether not to use CUDA when available"}
    )

    fp16: bool = field(
        default=True,
        metadata={'help': "Whether to use 16-bit float precision instead of 32-bit"}
    )

    input_max_len: int = field(
        default=256
    )

    output_max_len: int = field(
        default=128
    )

    gradient_accumulation_steps: int = field(
        default=6,
        metadata={'help': "Number of updates steps to accumulate before performing a backward/update pass."}
    )

    seed: int = field(
        default=42,
        metadata={'help': "random seed for initialization"}
    )

    do_train: bool = field(
        default=True,
        metadata={'help': "Whether to run training."}
    )

    do_eval: bool = field(
        default=True,
        metadata={'help': "Whether to run eval on the dev set."}
    )

    code_vocab_size: int = field(
        default=50000,
        metadata={'help': 'Maximum size of code vocab'}
    )

    nl_vocab_size: int = field(
        default=30000,
        metadata={'help': 'Maximum size of nl vocab'}
    )

    code_vocab_name: str = field(
        default='code',
        metadata={'help': 'Name of the code vocab'}
    )

    nl_vocab_name: str = field(
        default='nl',
        metadata={'help': 'Name of the nl vocab'}
    )

    # max_code_len: int = field(
    #     default=256,
    #     metadata={'help': 'Maximum length of code sequence'}
    # )
    #
    # max_nl_len: int = field(
    #     default=64,
    #     metadata={'help': 'Maximum length of the nl sequence'}
    # )

    code_tokenize_method: str = field(
        default='bpe',
        metadata={'help': 'Tokenize method of code',
                  'choices': ['word', 'bpe']}
    )

    nl_tokenize_method: str = field(
        default='bpe',
        metadata={'help': 'Tokenize method of nl',
                  'choices': ['word', 'bpe']}
    )

    mass_mask_ratio: float = field(
        default=0.5,
        metadata={'help': 'Ratio between number of masked tokens and number of total tokens, in MASS'}
    )

    keep_last_epochs: int = field(
        default=5,
    )

    loss_scale: float = field(
        default=0
    )


def transfer_arg_name(name):
    return '--' + name.replace('_', '-')

def add_pretrain_args(parser):
    """Add all arguments to the given parser."""
    for data_container in [PretrainArguments]:
        group = parser.add_argument_group(data_container.__name__)
        for data_field in dataclasses.fields(data_container):
            if 'action' in data_field.metadata:
                group.add_argument(transfer_arg_name(data_field.name),
                                   default=data_field.default,
                                   **data_field.metadata)
            else:
                group.add_argument(transfer_arg_name(data_field.name),
                                   type=data_field.type,
                                   default=data_field.default,
                                   **data_field.metadata)