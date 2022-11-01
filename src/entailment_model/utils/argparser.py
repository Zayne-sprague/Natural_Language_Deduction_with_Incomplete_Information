import argparse

from entailment_model.utils.eval_config import eval_config


def add_dataset_arguments(parser: argparse.ArgumentParser):
    """Helper function to define arguments for the generate_dataset script"""

    parser.add_argument('--dataset_name', '-dn', type=str, help='Name of the dataset, if not specified "tmp_dataset" '
                                                                'will be used and it will be overwritten on the next '
                                                                'run',
                        default="tmp_dataset")

    parser.add_argument('--validation_percentage', '-vp', type=float,
                        help='Percent of training that will be held out for validation, can be [0,1)', default=0.333)

    parser.add_argument('--label_columns', '-lc', type=str, nargs='+',
                        help='Columns to use as labels separated by whitespace i.e. (Greg Kaj Zayne)',
                        default=eval_config.csv_label_columns)

    parser.add_argument('--folded', '-f', action='store_true', dest='folded',
                        help="create multiple datasets with unique hypothesis, the number of folds is determined by"
                             "the number of validation folds that can be made given the --validation_percentage, "
                             "i.e. (-vp 0.333 will create 3 folds with 33 percent of the training dataset held out in "
                             "each fold). ", default=False)

    parser.add_argument('--use_all_rows_for_validation', '-ua', action='store_true', dest='use_all_rows_for_validation',
                        help='Use all the rows in the held out validation, if not specified only the first 150 rows'
                             'that were originally annotated by 3 labelers will be used.')

    parser.add_argument('--only_use_original_rows', '-oor', action='store_true', dest='only_use_original_rows',
                        help='Use the original 150 rows ONLY for training and validation.')


def add_training_arguments(parser: argparse.ArgumentParser):
    """Helper function to define arguments for the train script"""

    parser.add_argument('--learning_rate', '-lr', type=float, help='The learning rate of the model during training',
                        default=0.00009)

    parser.add_argument('--epochs', '-e', type=int, help='The number of epochs to train a model', default=2)

    parser.add_argument('--dataset_name', '-dn', type=str, help='Name of the dataset', default='tmp_dataset')

    parser.add_argument('--huggingface_model', '-hm', type=str, help='Name of the HuggingFace model you want to use',
                        default=eval_config.base_evaluation_model)

    parser.add_argument('--run_name', '-rn', type=str,
                        help='Name of the run that will be used, if not set "tmp_model" will be used and it can be '
                             'overwritten',
                        default='tmp_model')


def add_evaluation_arguments(parser: argparse.ArgumentParser):
    """Helper function to define arguments for the evaluate script"""

    parser.add_argument('--run_name', '-rn', type=str,
                        help='Name of the run that will be used, if not set "tmp_model" will be used',
                        default='tmp_model')

    parser.add_argument('--threshold', '-t', type=float, nargs='+', default=[0.47],
                        help='The thresholds you want to specifically test, thresholds are separated by whitespace '
                             '(i.e. 0.1 0.2 0.3 0.47 0.9)')

    parser.add_argument('--test_all_thresholds', '-tat', action='store_true', dest='test_all_thresholds',
                        help='Test every threshold within the range [0,1].')

    parser.add_argument('--all_thresholds_step', '-ats', type=float, default=0.1,
                        help='If --test_all_thresholds is specified, this is the step size of each threshold. (0, 1)')

    parser.add_argument('--include_accuracy', '-ia', action='store_true', dest='include_accuracy')
    parser.add_argument('--include_recall', '-ir', action='store_true', dest='include_recall')
    parser.add_argument('--include_precision', '-ip', action='store_true', dest='include_precision')

    parser.add_argument('--export_bad_predictions', '-ebp', action='store_true', dest='export_bad_predictions')


def add_entropy_arguments(parser: argparse.ArgumentParser):
    """Helper function to define arguments for the entropy_scores script"""

    parser.add_argument('--dataset_name', '-dn', type=str, help='Name of the dataset', default='tmp_dataset')

    parser.add_argument('--run_name', '-rn', type=str,
                        help='Name of the run that will be used, if not set "tmp_model" will be used and it can be '
                             'overwritten',
                        default='tmp_model')


def add_grid_search_arguments(parser: argparse.ArgumentParser):
    """Helper function to define arguments for the grid_search script"""

    parser.add_argument('--learning_rates', '-lr', type=float, help='List of learning rates to search over seperated by'
                                                                    ' white space (i.e. 0.0005 0.000005 0.00005)',
                        nargs="+", default=[0.00009])

    parser.add_argument('--epochs', '-e', type=int, help='List of epochs to search over seperated by white space'
                                                         '(i.e. 1 2 3 4 5)', default=[2],
                        nargs='+')

    parser.add_argument('--dataset_name', '-dn', type=str, help='Name of the dataset', default='tmp_dataset')

    parser.add_argument('--huggingface_models', '-hm', type=str, help='List of hugging face models to search over '
                                                                      'seperated by white space (i.e. model1 model2)',
                        default=[eval_config.base_evaluation_model], nargs='+')

    parser.add_argument('--run_name', '-rn', type=str,
                        help='Name of the run that will be used, if not set "tmp_model" will be used and it can be '
                             'overwritten',
                        default='tmp_model')

    parser.add_argument('--findings_file_name', '-ffn', type=str, help='The file to create that holds the results of'
                                                                       'the search.',
                        default='tmp_grid_search.json')
