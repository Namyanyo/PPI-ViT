#!/usr/bin/python
import sys
import os
import requests
from tqdm import tqdm
from Bio.PDB import PDBParser, PDBIO, Select
from default_config.masif_opts import masif_opts

os.makedirs(masif_opts['raw_pdb_dir'], exist_ok=True)
os.makedirs(masif_opts['tmp_dir'], exist_ok=True)

class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = [c.upper() for c in chain_ids]
    
    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids

def download_pdb_http(pdb_id, output_dir):
    """使用 HTTP 直接下载"""
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    output_file = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(output_file):
        print("exist file:", output_file)
        return output_file
    
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        with open(output_file, 'w') as f:
            f.write(response.text)
        return output_file
    else:
        raise Exception(f"HTTP {response.status_code}")

with open(sys.argv[1], 'r') as f:
    entries = [line.strip() for line in f if line.strip()]

parser = PDBParser(QUIET=True)
success, failed = 0, []

for entry in tqdm(entries, desc="Downloading"):
    try:
        pdb_id = entry.split('_')[0]
        chains = entry.split('_')[1:]
        print(f"Downloading {entry}...")
        pdb_file = download_pdb_http(pdb_id, masif_opts['tmp_dir'])
        
        structure = parser.get_structure(pdb_id, pdb_file)
        output = os.path.join(masif_opts['raw_pdb_dir'], f"{entry}.pdb")
        io = PDBIO()
        io.set_structure(structure)
        
        if chains:
            io.save(output, ChainSelect(chains))
        else:
            io.save(output)
        
        success += 1
        print('success num:',success)
        
    except Exception as e:
        failed.append((entry, str(e)))
        tqdm.write(f"✗ {entry}: {e}")

print(f"\n✓ {success}/{len(entries)}")
if failed:
    with open("failed_pdbs.txt", 'w') as f:
        for e, err in failed:
            f.write(f"{e}\t{err}\n")