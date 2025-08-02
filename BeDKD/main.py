import sys

listone = sys.path.append('D:\DESK\MyWork\Code\Defense\Moderate-fitting-main')
import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from transformers.utils.versions import require_version
from create_model import *

from models.bedkd import *


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    poisoned_train_file: str = field(
        default="syntactic",
        metadata={"help": "The poisoned_train_file"},
    )
    clean_dev_file: str = field(
        default="syntactic",
        metadata={"help": "The clean_dev_file"},
    )
    poisoned_test_file: str = field(
        default="syntactic",
        metadata={"help": "The poisoned_test_file"},
    )
    clean_test_file: str = field(
        default="syntactic",
        metadata={"help": "The clean_test_file"},
    )


class RemainArgHfArgumentParser(HfArgumentParser):
    def parse_json_file(self, json_file: str, return_remaining_args=True):
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.
        """
        data = json.loads(Path(json_file).read_text())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: data.pop(k) for k in list(data.keys()) if k in keys}
            obj = dtype(**inputs)
            outputs.append(obj)

        remain_args = argparse.ArgumentParser()
        remain_args.__dict__.update(data)
        if return_remaining_args:
            return (*outputs, remain_args)
        else:
            return (*outputs,)



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = RemainArgHfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    DEFENSE=sys.argv[2]
    json_file = os.path.abspath(sys.argv[1])
    model_args, data_args, training_args, delta_args = parser.parse_json_file(json_file,return_remaining_args=True)

    model_args.poisoned_train_file = sys.argv[3] + "train.json"
    model_args.clean_dev_file = sys.argv[3] + "dev.json"
    model_args.poisoned_test_file = sys.argv[3] + "test_poisoned.json"
    model_args.clean_test_file = sys.argv[3] + "test_clean.json"


    training_args.output_dir = sys.argv[4]
    model_args.model_name_or_path = sys.argv[4]


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # Set seed before initializing model.
    set_seed(training_args.seed)

    mask="[MASK]"

    print(training_args.output_dir+"/trainer_state.json")
    if "clean" in training_args.output_dir:
        method="clean"
    elif "hidden" in training_args.output_dir:
        method="hidden"
    elif "sent" in training_args.output_dir:
        method="sent"
    elif "coling" in training_args.output_dir:
        method="coling"
    elif "bgm" in training_args.output_dir:
        method="bgm"
    elif "hidden" in training_args.output_dir:
        method="hidden"
    elif "style" in training_args.output_dir:
        method="style"
    elif "feminist" in training_args.output_dir:
        method = "feminist"
    else:
        print("No Attack Method Match!")

    path_models,path_results,num_labels = get_path(training_args.output_dir+"/trainer_state.json")
    if "agnews" in path_results:
        data_name = "agnews"
    if "sst2" in path_results:
        data_name = "sst2"
    if "olid" in path_results:
        data_name = "olid"

    label=0
    if "agnews" in path_results:
        label=1
    elif DEFENSE=='BeDKD':
        set_seed(42)
        for l in [320]:
            model, tokenizer = get_models(path_models, num_labels, training_args.seed)
            model_s = AutoModelForSequenceClassification.from_pretrained(path_models,
                                                                         output_hidden_states=True,
                                                                         num_labels=num_labels)
            model_s2 = AutoModelForSequenceClassification.from_pretrained(path_models,
                                                                         output_hidden_states=True,
                                                                         num_labels=num_labels)
            if l==320:
                for mn in [32]:
                    model, tokenizer = get_models(path_models, num_labels, training_args.seed)
                    model_s = AutoModelForSequenceClassification.from_pretrained(path_models,
                                                                         output_hidden_states=True,
                                                                         num_labels=num_labels)
                    model_s2 = AutoModelForSequenceClassification.from_pretrained(path_models,
                                                                         output_hidden_states=True,
                                                                         num_labels=num_labels)
                    poisoned_number=mn
                    model_s = BeDKD(model, model_s,model_s2, tokenizer, 2.5, 16, model_args, data_name, method, l,80,poisoned_number)
            else:
                poisoned_number=32
                model_s = BeDKD(model, model_s,model_s2, tokenizer, 2.5, 16, model_args, data_name, method, l,80,poisoned_number)
            


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
