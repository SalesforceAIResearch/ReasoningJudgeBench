import os,json,argparse
from collections import defaultdict

def read_jsonl(file_path):
    with open(file_path, 'r') as fr:
        data = [json.loads(line) for line in fr.readlines()]
    return data

def main(eval_dir):
    total_score = 0
    total_cons = 0
    total = 0

    total_score_by_split = defaultdict(float)
    total_cons_by_split = defaultdict(float)
    total_samples_by_split = defaultdict(int)

    split_names = {
        'aime_4o_pairwise': 'math',
        'aime_pairwise': 'math',
        'arc_challenge_pairwise': 'multihop',
        'bbeh_be_pairwise': 'multihop',
        'bbeh_cu_pairwise': 'everyday',
        'bbeh_dq_pairwise': 'everyday',
        'bbeh_hyper_pairwise': 'everyday',
        'bbeh_ma_pairwise': 'math',
        'reclor_pairwise': 'multihop',
        'strategy_qa_pairwise': 'everyday',
        'folio_pairwise': 'multihop',
        'supergpqa_pairwise': 'domain',
        'olympiadbench_pairwise': 'math',
    }

    for split, bm_split in split_names.items():
        with open(os.path.join(eval_dir, f'eval_result_{split}.json'), 'r') as fr:
            eval_result = json.load(fr)
        
        sample_results = read_jsonl(os.path.join(eval_dir, f'eval_detailed_{split}.jsonl'))
        num_samples = len(sample_results)

        total += num_samples
        total_score += eval_result['accuracy'] * num_samples
        total_cons += eval_result['consistency'] * num_samples

        total_samples_by_split[bm_split] += num_samples
        total_score_by_split[bm_split] += eval_result['accuracy'] * num_samples
        total_cons_by_split[bm_split] += eval_result['consistency'] * num_samples
        
                
    acc_micro_avg = total_score / total
    cons_micro_avg = total_cons / total
    out_data = {
        "accuracy": round(100*acc_micro_avg, 2),
        "consistency": round(100*cons_micro_avg, 2),
    }

    for bm_split in split_names.values():
        total_split, score_bm_split, cons_bm_split = total_samples_by_split[bm_split], total_score_by_split[bm_split], total_cons_by_split[bm_split]
        out_data[bm_split] = {
            "accruacy": round(100*score_bm_split / total_split, 2),
            "consistency": round(100*cons_bm_split / total_split, 2)
        }

    print(out_data)
    with open(os.path.join(eval_dir, f'reasoningjudgebench_eval.json'), 'w') as fw:
        json.dump(out_data, fw, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation parameters
    parser.add_argument("--eval_dir", type=str, help="the directory for storing results")

    args = parser.parse_args()

    main(args.eval_dir)