# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset

# wandbのインポートを追加
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S* # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s* #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            library = "tokenizers/punkt/{}.pickle".format(self.args.lang)
            splitter = nltk.load(library)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        try:
            # パースする前に、行の前後の空白や改行を削除
            line = json_line.strip()
            if not line:
                # 空行の場合はNoneを返してスキップ
                return None, None, len(json_line)
            data = json.loads(line)
        except json.JSONDecodeError:
            # JSONとしてパースできない行は警告を出してスキップ
            print(f"Skipping invalid JSON line: {json_line.strip()}", file=sys.stderr)
            return None, None, len(json_line)

        ids = {}
        lens = {} 

        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]

            all_doc_ids = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    all_doc_ids.extend(sentence_ids)

            if len(all_doc_ids) == 0:
                ids[key] = []
                lens[key] = []
                continue

            # Sliding Windowでチャンクに分割
            window_size = self.args.seq_length
            stride = self.args.sliding_window_stride
            
            chunked_ids = []
            for i in range(0, len(all_doc_ids), stride):
                chunk = all_doc_ids[i : i + window_size]
                if chunk:
                    chunked_ids.append(chunk)

            if not chunked_ids:
                ids[key] = []
                lens[key] = []
                continue

            # 全てのチャンクを一つのリストにフラット化
            ids[key] = [token for chunk in chunked_ids for token in chunk]
            # 各チャンクの長さ（すべて同じ）のリストを作成
            lens[key] = [len(chunk) for chunk in chunked_ids]

        return ids, lens, len(json_line)


class Partition(object):
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            docs_per_sec = count / elapsed
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
            if wandb_available and self.args.wandb_project:
                wandb.log({
                    'docs_per_sec': docs_per_sec,
                    'MB_per_sec': mbs,
                    'processed_documents': count,
                    'processed_bytes': total_bytes_processed,
                })

    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        if wandb_available and self.args.wandb_project:
            run_name = f"{self.args.wandb_name}-split-{os.path.basename(input_file_name)}"
            wandb.init(project=self.args.wandb_project, name=run_name, config=vars(self.args), reinit=True)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()
        if wandb_available and self.args.wandb_project:
            wandb.finish()


    def process_json_file(self, file_name):
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        if wandb_available and self.args.wandb_project:
            run_name = f"{self.args.wandb_name}-encode-{os.path.basename(input_file_name)}"
            wandb.init(project=self.args.wandb_project, name=run_name, config=vars(self.args), reinit=True)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        tokenizer = build_tokenizer(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        # levelは常に'sentence'として扱う
        level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                   impl=self.args.dataset_impl,
                                                   vocab_size=tokenizer.vocab_size)

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            if doc is not None:
                for key in doc.keys():
                    if doc[key]:
                        builders[key].add_doc(doc[key], sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)
        
        fin.close()
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])
        if wandb_available and self.args.wandb_project:
            wandb.finish()


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    # split-sentencesはSliding Windowと併用しない
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences (Do not use with sliding window).')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'HFTokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--seq-length', type=int, required=True,
                       help='Maximum sequence length to process (chunk size for sliding window).')
    group.add_argument('--sliding-window-stride', type=int, default=None,
                       help='Stride for the sliding window. Defaults to seq-length.')
    group.add_argument('--trust-remote-code', action='store_true',
                       help='To run HFTokenizer model from local path.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group = parser.add_argument_group(title='wandb')
    group.add_argument('--wandb-project', type=str, default=None,
                       help='Weights & Biases project name.')
    group.add_argument('--wandb-name', type=str, default='megatron-preprocess',
                       help='Weights & Biases run name prefix.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    # strideが指定されていない場合、シーケンス長と同じにする(オーバーラップなし)
    if args.sliding_window_stride is None:
        args.sliding_window_stride = args.seq_length

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    if args.wandb_project and not wandb_available:
        raise Exception("wandb library is required for logging, but not available.")

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True)
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)

        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)

        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

        if not partitions_present and not split_sentences_present:
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)

            index = 0
            for in_file_name in in_file_names:
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')

                for line in fin:
                    partitioned_input_files[index].write(line)
                    index = (index + 1)%args.partitions
                fin.close()

            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers//args.partitions)

    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # levelは常に'sentence'
    level = "sentence"

    if args.partitions > 1:
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        tokenizer = build_tokenizer(args)

        for key in args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                          key, level)
            builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                                         impl=args.dataset_impl,
                                                         vocab_size=tokenizer.vocab_size)
            for name in in_ss_out_names:
                parition_output_prefix = name['output_prefix']
                full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                                 key, level)
                builders[key].merge_file_(full_partition_output_prefix)
            builders[key].finalize(output_idx_files[key])

    # 統計情報の集計と書き出し
    print("\n--- Aggregating Final Statistics ---")
    grand_total_tokens = 0
    grand_total_samples = 0
    
    # 最初のキーとlevelを使って、最終的なデータセットのプレフィックスを構築
    # 注：すべてのキーで同じサンプル数/トークン数になるわけではない場合、この方法は不正確になる可能性がある
    final_dataset_prefix = "{}_{}_{}".format(args.output_prefix, args.json_keys[0], level)
    
    try:
        # IndexedDatasetを直接読み込んで統計情報を取得
        dataset = indexed_dataset.make_dataset(final_dataset_prefix, impl=args.dataset_impl)
        grand_total_samples = len(dataset.sizes)
        grand_total_tokens = np.sum(dataset.sizes).item() # .item()でPythonの数値に変換
    except Exception as e:
        print(f"Could not read final indexed dataset for statistics: {e}", file=sys.stderr)

    final_stats = {
        'total_tokens': grand_total_tokens,
        'total_samples': grand_total_samples
    }
    
    # 最終的な統計情報をJSONファイルに書き出す
    final_stats_file = f"{args.output_prefix}_final_stats.json"
    with open(final_stats_file, 'w') as f:
        json.dump(final_stats, f, indent=4)
    
    print(f"Wrote final statistics to {final_stats_file}")
    print("\n-------------------------------------------------")
    print("Preprocessing Complete.")
    print(f"Total Tokens: {grand_total_tokens:,}")
    print(f"Total Samples (Chunks/Sentences): {grand_total_samples:,}")
    print("-------------------------------------------------")


if __name__ == '__main__':
    main()