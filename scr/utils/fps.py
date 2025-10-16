from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_morgan_fps(
    smiles: str, radius: int = 2, nBits: int = 4096
) -> Chem.rdchem.Mol:
    """Convert a SMILES string to a Morgan fingerprint.
    Parameters:
    smiles (str): The SMILES string to convert.
    radius (int): The radius of the Morgan fingerprint.
    nBits (int): The number of bits in the fingerprint.
    Returns:
    rdkit.DataStructs.cDataStructs.ExplicitBitVect: The Morgan fingerprint.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    except Exception:
        return None
