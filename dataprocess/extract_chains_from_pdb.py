

import argparse
import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import PDBList


class ChainSelector(Select):

    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids


def extract_chains(pdb_file, chain_ids, output_file, mode='single'):

    parser = PDBParser(QUIET=True)
    io = PDBIO()

    try:
        structure = parser.get_structure('protein', pdb_file)
    except Exception as e:
        print(f"error: unrecognized PDB {pdb_file}: {e}")
        return False

    available_chains = set()
    for model in structure:
        for chain in model:
            available_chains.add(chain.get_id())

    missing_chains = set(chain_ids) - available_chains
    if missing_chains:
        print(f"warning: PDB file {pdb_file} is missing chains: {missing_chains}")
        print(f"  available chains: {available_chains}")
        return False

    if mode == 'single':

        io.set_structure(structure)
        selector = ChainSelector(chain_ids)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        io.save(output_file, selector)
        return True

    elif mode == 'separate':
        for chain_id in chain_ids:
            io.set_structure(structure)
            selector = ChainSelector([chain_id])

            output_base = os.path.splitext(output_file)[0]
            chain_output = f"{output_base}_{chain_id}.pdb"

            os.makedirs(os.path.dirname(chain_output), exist_ok=True)
            io.save(chain_output, selector)

        return True


def parse_chain_pair(chain_pair_str):

    parts = chain_pair_str.strip().split()
    if not parts:
        return None

    chain_pair = parts[0]
    components = chain_pair.split('_')

    if len(components) < 2:
        return None

    pdb_id = components[0]
    chains = components[1:]  

    return pdb_id, chains


def batch_extract_from_list(input_file, pdb_dir, output_dir, mode='single',
                            pdb_extension='.pdb', output_extension='.pdb'):
 
    success_count = 0
    fail_count = 0
    processed = set()


    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            result = parse_chain_pair(line)
            if not result:
                print(f"warning: invalid format on line {line_num}, skipping: {line}")
                continue

            pdb_id, chains = result

            input_pdb = os.path.join(pdb_dir, pdb_id + pdb_extension)

            if not os.path.exists(input_pdb):
                print(f"warning: PDB file does not exist: {input_pdb}")
                fail_count += 1
                continue

            if mode == 'single':

                output_filename = '_'.join([pdb_id] + chains) + output_extension
                output_path = os.path.join(output_dir, output_filename)
            elif mode == 'separate':

                output_filename = pdb_id + output_extension
                output_path = os.path.join(output_dir, output_filename)


            if output_path in processed:
                continue


            print(f"processing: {pdb_id} chain {chains} -> {os.path.basename(output_path)}")
            success = extract_chains(input_pdb, chains, output_path, mode)

            if success:
                success_count += 1
                processed.add(output_path)
            else:
                fail_count += 1



def download_pdb_file(pdb_id, output_dir):


    print(f"Download PDB file: {pdb_id}")
    pdbl = PDBList()

    try:

        filename = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format='pdb')
        print(f"success: {filename}")
        return filename
    except Exception as e:
        print(f"failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Extract specified chains from PDB files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
       
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str,
                            help='Input txt file (contains chain pairs list)')
    input_group.add_argument('--pdb_file', type=str,
                            help='Single PDB file path')
    input_group.add_argument('--download', type=str,
                            help='Download PDB ID from RCSB')


    parser.add_argument('--pdb_dir', type=str,
                       help='PDB file directory (required for batch mode)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (required for batch mode)')

    parser.add_argument('--chains', nargs='+',
                       help='Chain IDs to extract (single file mode)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file path (single file mode)')

    parser.add_argument('--mode', '-m', type=str, default='single',
                       choices=['single', 'separate'],
                       help='Extraction mode: single=merge into one file, separate=save separately')
    parser.add_argument('--pdb_extension', type=str, default='.pdb',
                       help='Input PDB file extension (default: .pdb)')
    parser.add_argument('--output_extension', type=str, default='.pdb',
                       help='Output PDB file extension (default: .pdb)')

    args = parser.parse_args()

    if args.input:
        if not args.pdb_dir or not args.output_dir:
            parser.error("Batch mode requires --pdb_dir and --output_dir arguments")

        batch_extract_from_list(
            input_file=args.input,
            pdb_dir=args.pdb_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            pdb_extension=args.pdb_extension,
            output_extension=args.output_extension
        )


    elif args.pdb_file:
        if not args.chains or not args.output:
            parser.error("Single file mode requires --chains and --output arguments")

        success = extract_chains(args.pdb_file, args.chains, args.output, args.mode)

        if success:
            print(f"success from {args.chains} to{args.output}")
        else:
            print(f"failed")


    elif args.download:
        if not args.chains or not args.output:
            parser.error("Download mode requires --chains and --output arguments")


        import tempfile
        temp_dir = tempfile.mkdtemp()
        pdb_file = download_pdb_file(args.download, temp_dir)

        if pdb_file:

            success = extract_chains(pdb_file, args.chains, args.output, args.mode)

            if success:
                print(f"success from {args.chains} to {args.output}")
            else:
                print(f"failed")



if __name__ == "__main__":
    main()
