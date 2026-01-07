"""
PDB Parser for extracting 3D atomic coordinates
"""
import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

warnings.simplefilter('ignore', PDBConstructionWarning)


class ProteinStructureParser:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)

    def parse_pdb(self, pdb_file):
        """
        Parse PDB file and extract atomic coordinates

        Args:
            pdb_file: Path to PDB file

        Returns:
            coords: numpy array of shape (N, 3) containing xyz coordinates
            atoms: list of atom information (name, residue, etc.)
        """
        try:
            structure = self.parser.get_structure('protein', pdb_file)
            coords = []
            
            for model in structure:
                for chain in model:
                    for residue in chain:
                        for atom in residue:
                            coords.append(atom.get_coord())
            
            if not coords:
                raise ValueError(f"No atoms found in {pdb_file}")
            
            return np.array(coords)
            
        except Exception as e:
            raise ValueError(f"Failed to parse {pdb_file}: {e}")

    def get_ca_atoms_only(self, pdb_file):
        """
        Extract only C-alpha atoms (backbone representation)

        Args:
            pdb_file: Path to PDB file

        Returns:
            coords: numpy array of C-alpha coordinates
        """
        structure = self.parser.get_structure('protein', pdb_file)

        coords = []

        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        coords.append(ca_atom.get_coord())

        return np.array(coords)

    def normalize_coordinates(self, coords):
        """
        Normalize coordinates to [0, 1] range

        Args:
            coords: numpy array of shape (N, 3)

        Returns:
            normalized_coords: normalized coordinates
            min_vals: minimum values for each axis
            max_vals: maximum values for each axis
        """
        if coords.size == 0:
            raise ValueError("Cannot normalize empty coordinate array")
        
        if len(coords.shape) != 2 or coords.shape[1] != 3:
            raise ValueError(f"Invalid coords shape: {coords.shape}, expected (N, 3)")

        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        range_vals = max_vals - min_vals
        
   
        range_vals[range_vals == 0] = 1.0

        normalized = (coords - min_vals) / range_vals
        
        return normalized, min_vals, max_vals




