import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import random

# ==========================================
# 1. DATA DEFINITIONS (Taken from TestSetGen.py)
# ==========================================

TAIL_FRAGS = {
    "1a": "[*]CC=C",             # allyl
    "1b": "[*]CCCO",             # 3-hydroxypropyl
    "1c": "[*]C=CC",             # propenyl
    "1d": "[*]C(O)CC",           # 2-hydroxypropyl
    "2a": "[*]Cc1ccccc1",        # benzyl
    "2b": "[*]Cc1c(C)cccc1",     # 2-methyl
    "2c": "[*]Cc1cc(C)ccc1",     # 3-methyl
    "2d": "[*]Cc1ccc(C)cc1",     # 4-methyl
    "2e": "[*]Cc1c(OC)cccc1",    # 2-methoxy
    "2f": "[*]Cc1cc(OC)ccc1",    # 3-methoxy
    "2g": "[*]Cc1ccc(OC)cc1",    # 4-methoxy
    "2h": "[*]Cc1c(N(C)C)cccc1", # 2-amine (NMe2)
    "2i": "[*]Cc1cc(N(C)C)ccc1", # 3-amine
    "2j": "[*]Cc1ccc(N(C)C)cc1", # 4-amine
}

SUB_FRAGS = {
    "a": "[*]H", "b": "[*]O", "c": "[*]S", "d": "[*]OC", "e": "[*]OC(F)(F)F",
    "f": "[*]SC", "g": "[*]N(C)C", "h": "[*]C=O", "i": "[*]C(C)=O", "j": "[*]F",
    "k": "[*]CF", "l": "[*]C(F)F", "m": "[*]C(F)(F)F", "n": "[*]C",
}

CORES = {
    "O": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)O2)c1",
    "S": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)S2)c1",
    "N": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)N2)c1",
    "M": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)N2C)c1",
}

# Mapping colors for visualization (RGB)
# Core: Blue, Tail: Purple, Sub: Orange
COLOR_MAP = {
    0: (0.1, 0.4, 0.6),  # Core - Blue-ish
    1: (0.5, 0.0, 0.5),  # Tail - Purple
    2: (1.0, 0.5, 0.0)   # Sub - Orange
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def tag_atoms(mol, tag_id):
    """Gán thẻ (tag) cho tất cả các atom trong mol để tô màu sau này."""
    for atom in mol.GetAtoms():
        # Tag 0: Core, 1: Tail, 2: Subs
        atom.SetIntProp("block_id", tag_id)
    return mol

def attach_fragment_tagged(core_mol: Chem.Mol, label: int, frag_smiles: str, tag_id: int) -> Chem.Mol:
    """
    Phiên bản sửa đổi của attach_fragment để hỗ trợ tô màu.
    Nó gán tag cho fragment trước khi gắn vào core.
    """
    # 1. Prepare Core (Keep existing properties)
    core = Chem.Mol(core_mol)
    
    # 2. Find dummy on core
    idx_dummy = None
    for a in core.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx_dummy = a.GetIdx(); break
    if idx_dummy is None:
        raise ValueError(f"Core missing [*:{label}]")

    nbrs = [n.GetIdx() for n in core.GetAtomWithIdx(idx_dummy).GetNeighbors()]
    idx_core_attach = nbrs[0]

    # 3. Prepare Fragment and Tag it
    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None: return core # Fail safe
    
    # Tag atoms of the fragment
    tag_atoms(frag, tag_id)

    # Find dummy in fragment
    idx_fd = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0][0]
    fnbr = [n.GetIdx() for n in frag.GetAtomWithIdx(idx_fd).GetNeighbors()]
    idx_fa = fnbr[0]

    # 4. Combine
    combo = Chem.CombineMols(core, frag)
    cn = core.GetNumAtoms()
    
    em = Chem.EditableMol(combo)
    em.AddBond(idx_core_attach, cn + idx_fa, order=Chem.rdchem.BondType.SINGLE)
    
    # Remove dummies
    for ridx in sorted([cn + idx_fd, idx_dummy], reverse=True):
        em.RemoveAtom(ridx)
        
    m = em.GetMol()
    try:
        Chem.SanitizeMol(m)
    except:
        pass # Handle tricky valence cases gracefully
    return m

def remove_dummy_label(mol, label):
    """Xóa dummy atom nếu nhóm thế là 'a' (Hydro)"""
    idx = None
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx = a.GetIdx(); break
    if idx is not None:
        em = Chem.EditableMol(mol)
        em.RemoveAtom(idx)
        m = em.GetMol()
        Chem.SanitizeMol(m)
        return m
    return mol

def build_molecule(head, tail_code, s3, s4, s5):
    # 1. Start with Core (Tag 0)
    core_smi = CORES[head]
    mol = Chem.MolFromSmiles(core_smi)
    tag_atoms(mol, 0) # Tag core atoms as 0

    # 2. Attach Tail (Tag 1)
    tail_frag = TAIL_FRAGS[tail_code]
    mol = attach_fragment_tagged(mol, 9, tail_frag, 1)

    # 3. Attach Substituents (Tag 2)
    subs = [(3, s3), (4, s4), (5, s5)]
    for lab, sub_code in subs:
        if sub_code == "a":
            # Nếu là 'a', chỉ xóa dummy, không thêm atom mới nên không cần highlight
            mol = remove_dummy_label(mol, lab)
        else:
            sub_frag = SUB_FRAGS[sub_code]
            mol = attach_fragment_tagged(mol, lab, sub_frag, 2)
    
    return mol

def mol_to_image(mol):
    """Tạo ảnh SVG với highlight màu"""
    # Create highlight maps
    highlight_atoms = {}
    
    for atom in mol.GetAtoms():
        if atom.HasProp("block_id"):
            tag = atom.GetIntProp("block_id")
            highlight_atoms[atom.GetIdx()] = COLOR_MAP[tag]
            
    # Draw
    d2d = rdMolDraw2D.MolDraw2DSVG(600, 400)
    d2d.drawOptions().addAtomIndices = False
    d2d.drawOptions().bondLineWidth = 2
    
    # Sanitize & Compute Coordinates
    try:
        Chem.SanitizeMol(mol)
        Chem.Compute2DCoords(mol)
        try:
            Chem.Kekulize(mol)
        except:
            pass # Sometimes aromaticity fails in display, visually usually ok
    except:
        pass

    d2d.DrawMoleculeWithHighlights(mol, "Molecule", dict(highlight_atoms), {}, {}, {})
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

# ==========================================
# 3. STREAMLIT APP UI
# ==========================================

st.set_page_config(page_title="Chemical Block Builder", layout="wide")

st.title("🧩 Chemical Building Block Assembler")
st.markdown("Chọn các thành phần Head, Tail và Substituents để tạo cấu trúc hóa học. Các phần được tô màu tương ứng với vai trò của chúng.")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

# Random Button Logic
if 'random_trigger' not in st.session_state:
    st.session_state.random_trigger = False

def randomize():
    st.session_state.head_val = random.choice(list(CORES.keys()))
    st.session_state.tail_val = random.choice(list(TAIL_FRAGS.keys()))
    st.session_state.s3_val = random.choice(list(SUB_FRAGS.keys()))
    st.session_state.s4_val = random.choice(list(SUB_FRAGS.keys()))
    st.session_state.s5_val = random.choice(list(SUB_FRAGS.keys()))

st.sidebar.button("🎲 Random Structure", on_click=randomize)

# Selections
# Use session state to allow random button to update widgets
head_sel = st.sidebar.selectbox("Head (Core - Blue)", list(CORES.keys()), key='head_val')
tail_sel = st.sidebar.selectbox("Tail (Block III - Purple)", list(TAIL_FRAGS.keys()), key='tail_val')

st.sidebar.markdown("---")
st.sidebar.subheader("Substituents (Orange)")
s3_sel = st.sidebar.selectbox("Pos 3", list(SUB_FRAGS.keys()), key='s3_val')
s4_sel = st.sidebar.selectbox("Pos 4", list(SUB_FRAGS.keys()), key='s4_val')
s5_sel = st.sidebar.selectbox("Pos 5", list(SUB_FRAGS.keys()), key='s5_val')

# --- Main Area ---

col1, col2 = st.columns([2, 1])

with col1:
    # Logic Generation
    try:
        final_mol = build_molecule(head_sel, tail_sel, s3_sel, s4_sel, s5_sel)
        
        # Generate Code String
        code_str = f"{head_sel}{tail_sel}3{s3_sel}4{s4_sel}5{s5_sel}"
        
        st.subheader(f"Code: `{code_str}`")
        
        # Display SVG
        svg = mol_to_image(final_mol)
        st.image(svg, use_container_width=False)
        
        # Canonical Smiles
        can_smi = Chem.MolToSmiles(final_mol, isomericSmiles=True)
        with st.expander("Show Canonical SMILES"):
            st.code(can_smi)

    except Exception as e:
        st.error(f"Error constructing molecule: {e}")

with col2:
    st.markdown("### Legend")
    
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 20px; height: 20px; background-color: #1f77b4; margin-right: 10px; border-radius: 3px;"></div>
            <span><b>Core (Head)</b><br>Khung chính (O/S/N/M)</span>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 20px; height: 20px; background-color: #800080; margin-right: 10px; border-radius: 3px;"></div>
            <span><b>Tail (Block III)</b><br>Nhóm gắn tại vị trí 9</span>
        </div>
        """, unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 20px; height: 20px; background-color: #ff7f0e; margin-right: 10px; border-radius: 3px;"></div>
            <span><b>Substituents</b><br>Nhóm thế tại vị trí 3, 4, 5</span>
        </div>
        """, unsafe_allow_html=True
    )

    st.info("Lưu ý: Nếu chọn nhóm thế là 'a' (Hydrogen), nó được coi là ẩn (implicit H) và sẽ không được tô màu highlight.")


