from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def smiles_to_morgan_fps(smiles: str, radius: int = 2, nBits: int = 4096):
    """Convert a SMILES string to a Morgan fingerprint.
    Parameters:
        smiles (str): The SMILES string to convert.
        radius (int): The radius of the Morgan fingerprint.
        nBits (int): The number of bits in the fingerprint.
    Returns:
        rdkit.DataStructs.cDataStructs.ExplicitBitVect: The Morgan fingerprint,
        or None if conversion fails.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Use the new Morgan fingerprint generator
        morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits)
        fp = morgan_gen.GetFingerprint(mol)
        return fp

    except Exception:
        return None
