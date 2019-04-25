from glob import glob, iglob
from nltk.corpus import stopwords
from typing import List, Iterable, Text, Container, Tuple, Optional, Dict
import numpy as np
import os
import pandas as pd
from argparse import ArgumentTypeError
# Type aliases for filter_texts function
Token = Text
Tokens_str = Token
Tokens = List[Token]


def filter_tokens(tokens: Iterable[Text],
                  stop_words: Optional[Container[Text]]) -> Tokens_str:
    from re import compile as regex
    from string import printable as printable_chars

    email_filter = regex(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    punc_filter = regex(r"[!#$&%()*+./:;<=>?@^_`{|}~,\-\\\"\']+")
    num_filter = regex("[0-9]+")
    enum_filter = regex(r"^\(?\w+([,.]\)?|[,.\)])")

    def yield_filtered_tokens(tokens: Iterable[Token]) -> Iterable[Token]:
        def filter_token(token: Text) -> Text:
            def strip_enum(token: Text) -> Text:
                if not token:
                    return ''
                if token[0] == '(' and token[len(token) - 1] != ')':
                    return ''
                if token[0] != '(' or (token[0] == '(' and token[len(token) -
                                                                 1] == ')'):
                    return ''.join(enum_filter.split(token))
                return ''

            if email_filter.match(token) or (
                stop_words and token in stop_words
            ):
                return ''
            # Strip enumeration from token
            token = strip_enum(token)
            # Strip punctuation from token
            token = ''.join(punc_filter.split(token))
            # Strip numbers from token
            token = ''.join(num_filter.split(token))
            # Remove non-printable characters
            token = ''.join(c for c in token if c in printable_chars)

            return '' if len(token) < 3 else token

        for token in tokens:
            filtered_token = filter_token(token)
            if filtered_token:
                yield filtered_token

    return ' '.join(token for token in yield_filtered_tokens(tokens))


def get_filtered_file(filename: Text,
                      stop_words: Optional[Container[Text]] = None
                      ) -> Tokens_str:
    from re import compile as regex

    ws_filter = regex(r"\s+")
    with open(filename, 'rb') as f:
        decoded_str = f.read().decode(errors="ignore").strip().lower()
        return filter_tokens(ws_filter.split(decoded_str), stop_words)

    raise ValueError("Invalid File name!")


def yield_filtered_files(should_export_extras: bool = False,
                         input_dir: str = "data_clean",
                         output_dir: str = "filtered_data_clean",
                         stop_words: Optional[Container[Text]] = None
                         ) -> Iterable[pd.DataFrame]:
    filtered_folder = os.path.abspath(output_dir)
    if should_export_extras and not os.path.exists(filtered_folder):
        os.makedirs(filtered_folder)

    for folder_path in iglob(os.path.join(os.path.abspath(input_dir), "*")):
        valid_file_data: Dict[Text, Tokens_str] = {
            k: v for k, v in {
                file_name: get_filtered_file(file_name, stop_words)
                for file_name in glob(os.path.join(folder_path, "*.txt"))
            }.items()
            if v
        }

        if valid_file_data:
            folder_name = os.path.basename(folder_path)

            # base_names = [os.path.basename(name) for name in valid_file_data]
            texts_df = pd.DataFrame({
                "filename": list(
                    os.path.basename(name) for name in valid_file_data.keys()
                ),
                "label": folder_name,
                "tokens": list(valid_file_data.values())
            })


            if should_export_extras:
                export_folder = os.path.join(filtered_folder, folder_name)

                if not os.path.exists(export_folder):
                    os.makedirs(export_folder)

                texts_df.to_csv(
                    os.path.join(export_folder, "texts.csv"), index=False,
                    columns=["filename", "tokens"]
                )

            yield texts_df


def split_dataset(should_export_extras: bool = False,
                  split_percent: float = 0.8,
                  input_dir: str = "data",
                  output_dir: str = "filtered_data_clean",
                  stopwords_file_path: Optional[str] = None
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if should_export_extras:
        full_arr: List[Tuple[str, str, str]] = []

    train_arr: List[Tuple[str, str, str]] = []
    test_arr: List[Tuple[str, str, str]] = []

    stop_words = set(stopwords.words('english'))
    if stopwords_file_path:
        with open(stopwords_file_path, "r") as fsmb_stop_words:
            stop_words.update(fsmb_stop_words.read().splitlines())

    for df in yield_filtered_files(should_export_extras=should_export_extras,
                                   input_dir=input_dir, output_dir=output_dir,
                                   stop_words=stop_words):
        if should_export_extras:
            full_arr.extend(df.values)

        sample_train = df.sample(frac=split_percent)

        train_arr.extend(sample_train.values)
        test_arr.extend(df.drop(sample_train.index).values)

    train_df = pd.DataFrame(train_arr,
                            columns=["filename", "label", "tokens"])
    test_df = pd.DataFrame(test_arr,
                           columns=["filename", "label", "tokens"])

    filtered_folder = os.path.abspath(output_dir)
    if should_export_extras:
        if not os.path.exists(filtered_folder):
            os.makedirs(filtered_folder)

        pd.DataFrame(full_arr, columns=["filename", "label", "tokens"]).to_csv(
            os.path.join(filtered_folder, "all_texts.csv"),
            index=False
        )
        train_df.to_csv(os.path.join(filtered_folder, "train_texts.csv"),
                        index=False
                        )
        test_df.to_csv(os.path.join(filtered_folder, "test_texts.csv"),
                       index=False
                       )

    return train_df, test_df


def get_slices(all_texts_df: pd.DataFrame,
               slice_length: int = 25,
               overlap_percent: float = 0) -> pd.DataFrame:

    if overlap_percent >= 1 or overlap_percent < 0:
        raise ValueError("Invalid overlap amount")

    step = max(1, int(slice_length * (1 - overlap_percent)))

    all_slices: List[Tuple[Text, Text]] = []

    for row in all_texts_df.itertuples(index=False):
        tokens = row.tokens.split()
        snippets = [
            ' '.join(tokens[i:min(len(tokens), i + slice_length)])
            for i in range(0, len(tokens), step)
        ]
        all_slices += [(row.label, snippet) for snippet in snippets]

    return pd.DataFrame(all_slices, columns=["label", "slice"])


def export_fasttext_data(df: pd.DataFrame, output_name: str,
                         slice_length: Optional[int] = 25,
                         overlap_percent: float = 0):

    if not os.path.exists(os.path.dirname(output_name)):
        os.makedirs(os.path.dirname(output_name))

    df["label"] = "__label__" + df["label"]
    df.drop(columns=["filename"], inplace=True)
    if slice_length:
        np.savetxt(
            output_name,
            get_slices(df, slice_length, overlap_percent).values,
            fmt="%s"
        )

def construct_parser_preprocessor(subparser):
    def within_percent_interval(interval_str: str) -> float:
        interval = float(interval_str)
        if interval < 0 or interval > 1:
            raise ArgumentTypeError("Input given is out of bounds!")

        return interval
    """
    if subparser:
        preprocess_parser = subparser.add_parser(
            "preprocess",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help="Preprocess given dataset",
        )
    else:
        preprocess_parser = argparse.ArgumentParser(
            description='Preprocess given dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    """

    subparser.add_argument(
        'inputdir', type=str, default="data_clean", metavar="input-dir",
        help='Input directory to preprocess'
    )

    subparser.add_argument(
        '-o', '--outputdir', type=str, default="filtered_data",
        help='Output directory to hold preprocessed data_clean'
    )

    subparser.add_argument(
        '--stopwords', type=str, default=None,
        help='Path to the .csv stop words file'
    )

    # Make file generation options mutually exclusive
    # Note, all 3 of the flags appear. However, we only want 1 of them to
    # appear.
    subparser.add_argument(
        '--export', type=str, default="single",
        choices=["single", "split", "both"],
        help='Indicate whether you only want a single file holding all of the \
        preprocessed data_clean, or both. If "split\" or "both" were chosen, the \
        split is based on "--train-split" or "--test-split" or have the train \
        split be 80%% of the raw data_clean if neither argument was given.'
    )

    subparser.add_argument(
        '--train-split', type=within_percent_interval, default=.8, metavar="[0-1]",
        help="Percentage in interval [0,1] of total data_clean going to the \
        training dataset."
    )

    subparser.set_defaults(run=run_preprocessor)


def run_preprocessor(args):
    train_df, test_df = split_dataset(
        should_export_extras=True,
        split_percent=args.train_split,
        input_dir=args.inputdir,
        output_dir=args.outputdir,
        stopwords_file_path=args.stopwords
    )

def main():
    train_df, test_df = split_dataset(should_export_extras=True,
                                      split_percent=0.8)

    train_slice_name = os.path.join(os.getcwd(), "filtered_data_clean",
                                    "fasttext_train.txt")
    test_slice_name = os.path.join(os.getcwd(), "filtered_data_clean",
                                   "fasttext_test.txt")

    export_fasttext_data(train_df, train_slice_name, slice_length=10)
    export_fasttext_data(test_df, test_slice_name, slice_length=10)


if __name__ == "__main__":
    main()