

import argparse
import random
import os
from pathlib import Path
from collections import defaultdict


def parse_chain_pair(chain_pair_str):
  
    parts = chain_pair_str.strip().split()
    if not parts:
        return None

    chain_pair = parts[0]
    components = chain_pair.split('_')

    if len(components) < 3:
        return None

    pdb_id = components[0]
    chain1 = components[1]
    chain2 = components[2]

    return pdb_id, chain1, chain2


def load_positive_samples(input_file):

    positive_pairs = set()
    pdb_chains = defaultdict(set)

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            result = parse_chain_pair(line)
            if result:
                pdb_id, chain1, chain2 = result
                pair = (pdb_id, min(chain1, chain2), max(chain1, chain2))
                positive_pairs.add(pair)
                pdb_chains[pdb_id].add(chain1)
                pdb_chains[pdb_id].add(chain2)

    return positive_pairs, pdb_chains


def scan_data_directory(data_dir, mode='single'):

    pdb_chains = defaultdict(set)

    pdb_files = list(Path(data_dir).glob('*.pdb'))

    for pdb_file in pdb_files:
        filename = pdb_file.stem  
        if mode == 'single':

            parts = filename.split('_')
            if len(parts) >= 3:
                pdb_id = parts[0]
                chains = parts[1:]
                for chain in chains:
                    pdb_chains[pdb_id].add(chain)

        elif mode == 'pair':

            parts = filename.split('_')
            if len(parts) >= 2:
                pdb_id = parts[0]
                chain = parts[1]
                pdb_chains[pdb_id].add(chain)

    return pdb_chains


def generate_negative_samples_cross_pdb(pdb_chains, positive_pairs, num_samples):

    negative_samples = []
    all_chains = []

    for pdb_id, chains in pdb_chains.items():
        for chain in chains:
            all_chains.append((pdb_id, chain))

    if len(all_chains) < 2:
        raise ValueError("at least two chains are required to generate negative samples")

    attempts = 0
    max_attempts = num_samples * 10

    while len(negative_samples) < num_samples and attempts < max_attempts:
        attempts += 1

        chain1_info = random.choice(all_chains)
        chain2_info = random.choice(all_chains)

        pdb1, c1 = chain1_info
        pdb2, c2 = chain2_info

        if pdb1 == pdb2:
            continue

    
        if c1 <= c2:
            pair = (pdb1, c1, c2)
            pair_str = f"{pdb1}_{c1}_{c2}"
        else:
            pair = (pdb1, c2, c1)
            pair_str = f"{pdb1}_{c2}_{c1}"

        if pair not in positive_pairs and pair_str not in negative_samples:
            negative_samples.append(pair_str)

    return negative_samples


def generate_negative_samples_within_pdb(pdb_chains, positive_pairs, num_samples):
 
    negative_samples = []

    attempts = 0
    max_attempts = num_samples * 10

    while len(negative_samples) < num_samples and attempts < max_attempts:
        attempts += 1

        pdb_id = random.choice(list(pdb_chains.keys()))
        chains = list(pdb_chains[pdb_id])

        if len(chains) < 2:
            continue

        chain1, chain2 = random.sample(chains, 2)

        if chain1 > chain2:
            chain1, chain2 = chain2, chain1

        pair = (pdb_id, chain1, chain2)
        pair_str = f"{pdb_id}_{chain1}_{chain2}"


        if pair not in positive_pairs and pair_str not in negative_samples:
            negative_samples.append(pair_str)

    return negative_samples


def generate_negative_samples_random(pdb_chains, positive_pairs, num_samples):

    negative_samples = []
    all_pdb_ids = list(pdb_chains.keys())

    attempts = 0
    max_attempts = num_samples * 10

    while len(negative_samples) < num_samples and attempts < max_attempts:
        attempts += 1

        pdb1 = random.choice(all_pdb_ids)
        pdb2 = random.choice(all_pdb_ids)

        chains1 = list(pdb_chains[pdb1])
        chains2 = list(pdb_chains[pdb2])

        if not chains1 or not chains2:
            continue

        chain1 = random.choice(chains1)
        chain2 = random.choice(chains2)

   
        if chain1 > chain2:
            chain1, chain2 = chain2, chain1

        pair = (pdb1, chain1, chain2)
        pair_str = f"{pdb1}_{chain1}_{pdb2}_{chain2}"


        if pair not in positive_pairs and pair_str not in negative_samples:
            negative_samples.append(pair_str)

    return negative_samples


def main():
    parser = argparse.ArgumentParser(
        description='generate negative samples for PPI-ViT',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 输入来源
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='input positive samples txt file')
    input_group.add_argument('--scan_dir', type=str,
                            help='scan directory for PDB files')

    # 输出
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='output negative samples txt file')

    # 参数
    parser.add_argument('--num_samples', '-n', type=int, default=1000,
                       help='number of negative samples to generate (default: 1000)')
    parser.add_argument('--strategy', '-s', type=str, default='cross_pdb',
                       choices=['cross_pdb', 'within_pdb', 'random'],
                       help='negative sample generation strategy')
    parser.add_argument('--scan_mode', type=str, default='single',
                       choices=['single', 'pair'],
                       help='scan mode: single (1A0G_A_B.pdb) or pair (1A0G_A.pdb)')
    parser.add_argument('--with_labels', action='store_true',
                       help='output file includes labels (append 0 to the end of each line)')
    parser.add_argument('--seed', type=int, default=42,
                       help='random seed (default: 42)')

    args = parser.parse_args()

    random.seed(args.seed)

    positive_pairs = set()
    pdb_chains = {}

    if args.input:
        print(f"loading samples: {args.input}")
        positive_pairs, pdb_chains = load_positive_samples(args.input)
        print(f"find {len(positive_pairs)} positive samples")
        print(f"inculding {len(pdb_chains)} PDB")

    elif args.scan_dir:
        print(f"scan directory: {args.scan_dir}")
        pdb_chains = scan_data_directory(args.scan_dir, args.scan_mode)
        print(f"find {len(pdb_chains)} PDB")

    if not pdb_chains:
        print("error: no PDB chains found")
        return


    total_chains = sum(len(chains) for chains in pdb_chains.values())
    print(f"total chains: {total_chains}")

    print(f"strategy: {args.strategy}")
    print(f"generate: {args.num_samples} naegative samples")

    if args.strategy == 'cross_pdb':
        negative_samples = generate_negative_samples_cross_pdb(
            pdb_chains, positive_pairs, args.num_samples
        )
    elif args.strategy == 'within_pdb':
        negative_samples = generate_negative_samples_within_pdb(
            pdb_chains, positive_pairs, args.num_samples
        )
    elif args.strategy == 'random':
        negative_samples = generate_negative_samples_random(
            pdb_chains, positive_pairs, args.num_samples
        )

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)

    with open(args.output, 'w') as f:
        for sample in negative_samples:
            if args.with_labels:
                f.write(f"{sample} 0\n")
            else:
                f.write(f"{sample}\n")




if __name__ == "__main__":
    main()
