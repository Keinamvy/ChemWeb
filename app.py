import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import random

# ==========================================
# 1. DATA DEFINITIONS
# ==========================================

TAIL_FRAGS = {
    "1a": "[*]CC=C",             "1b": "[*]CCCO",             "1c": "[*]C=CC",
    "1d": "[*]C(O)CC",           "2a": "[*]Cc1ccccc1",        "2b": "[*]Cc1c(C)cccc1",
    "2c": "[*]Cc1cc(C)ccc1",     "2d": "[*]Cc1ccc(C)cc1",     "2e": "[*]Cc1c(OC)cccc1",
    "2f": "[*]Cc1cc(OC)ccc1",    "2g": "[*]Cc1ccc(OC)cc1",    "2h": "[*]Cc1c(N(C)C)cccc1",
    "2i": "[*]Cc1cc(N(C)C)ccc1", "2j": "[*]Cc1ccc(N(C)C)cc1",
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

# Màu sắc (RGB List)
COLOR_MAP = {
    0: (0.1, 0.4, 0.6),  # Core (Xanh) - phần còn lại
    1: (0.5, 0.0, 0.5),  # Tail (Tím)
    2: (1.0, 0.5, 0.0)   # Subs (Cam)
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def tag_atoms(mol, tag_id):
    """Gán thẻ màu cho toàn bộ mol."""
    for atom in mol.GetAtoms():
        atom.SetIntProp("block_id", tag_id)
    return mol

def untag_sub_ring(mol):
    """
    Tìm vòng benzen có chứa các dummy 3, 4, 5 và XÓA thẻ màu của các atom trong vòng đó.
    Điều này giúp vòng benzen hiển thị màu đen trắng mặc định.
    """
    target_labels = [3, 4, 5]
    
    # 1. Tìm các atom trên vòng (anchor) đang nối trực tiếp với dummy 3,4,5
    anchor_indices = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.HasProp("molAtomMapNumber"):
            if int(atom.GetProp("molAtomMapNumber")) in target_labels:
                # Tìm neighbor của dummy (đây chính là atom carbon trên vòng)
                nbrs = atom.GetNeighbors()
                if nbrs:
                    anchor_indices.add(nbrs[0].GetIdx())
    
    if not anchor_indices:
        return mol

    # 2. Lấy thông tin các vòng trong phân tử
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()
    
    atoms_to_untag = set()
    
    # 3. Nếu vòng nào chứa các anchor ở trên -> Đó là vòng cần bỏ tô màu
    for ring in atom_rings:
        # Kiểm tra giao thoa giữa tập index của vòng và tập anchor
        if set(ring).intersection(anchor_indices):
            atoms_to_untag.update(ring)
            
    # 4. Xóa thuộc tính block_id để không tô màu
    for idx in atoms_to_untag:
        atom = mol.GetAtomWithIdx(idx)
        if atom.HasProp("block_id"):
            atom.ClearProp("block_id")
            
    return mol

def attach_fragment_tagged(core_mol, label, frag_smiles, tag_id):
    """Gắn fragment và gán tag màu cho fragment đó."""
    core = Chem.Mol(core_mol)
    
    # Tìm dummy trên core
    idx_dummy = None
    for a in core.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx_dummy = a.GetIdx(); break
    if idx_dummy is None: return core 

    nbrs = [n.GetIdx() for n in core.GetAtomWithIdx(idx_dummy).GetNeighbors()]
    idx_core_attach = nbrs[0]

    frag = Chem.MolFromSmiles(frag_smiles)
    if frag is None: return core
    
    # Tag atoms của fragment mới
    tag_atoms(frag, tag_id)

    idx_fd = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0][0]
    fnbr = [n.GetIdx() for n in frag.GetAtomWithIdx(idx_fd).GetNeighbors()]
    idx_fa = fnbr[0]

    combo = Chem.CombineMols(core, frag)
    cn = core.GetNumAtoms()
    
    em = Chem.EditableMol(combo)
    em.AddBond(idx_core_attach, cn + idx_fa, order=Chem.rdchem.BondType.SINGLE)
    
    for ridx in sorted([cn + idx_fd, idx_dummy], reverse=True):
        em.RemoveAtom(ridx)
        
    m = em.GetMol()
    try: Chem.SanitizeMol(m)
    except: pass
    return m

def remove_dummy_label(mol, label):
    idx = None
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx = a.GetIdx(); break
    if idx is not None:
        em = Chem.EditableMol(mol)
        em.RemoveAtom(idx)
        m = em.GetMol()
        try: Chem.SanitizeMol(m)
        except: pass
        return m
    return mol

def build_molecule(head, tail_code, s3, s4, s5):
    # 1. Load Core & Tag tất cả là màu Core (0)
    core_smi = CORES[head]
    mol = Chem.MolFromSmiles(core_smi)
    tag_atoms(mol, 0) 
    
    # 2. XÓA TAG MÀU CỦA VÒNG BENZEN GẮN SUB
    # Bước này thực hiện khi dummy 3,4,5 vẫn còn trên mạch
    mol = untag_sub_ring(mol)

    # 3. Gắn Tail (Màu 1)
    tail_frag = TAIL_FRAGS[tail_code]
    mol = attach_fragment_tagged(mol, 9, tail_frag, 1)

    # 4. Gắn Subs (Màu 2)
    subs = [(3, s3), (4, s4), (5, s5)]
    for lab, sub_code in subs:
        if sub_code == "a":
            mol = remove_dummy_label(mol, lab)
        else:
            sub_frag = SUB_FRAGS[sub_code]
            mol = attach_fragment_tagged(mol, lab, sub_frag, 2)
    
    return mol

def mol_to_image(mol):
    """
    Tạo ảnh SVG. 
    LƯU Ý QUAN TRỌNG: Chỉ những atom có 'block_id' mới được tô màu.
    Các atom đã bị untag_sub_ring xóa 'block_id' sẽ hiển thị màu mặc định.
    """
    highlight_atoms_list = []
    highlight_atom_colors = {}
    
    for atom in mol.GetAtoms():
        if atom.HasProp("block_id"):
            tag = atom.GetIntProp("block_id")
            idx = atom.GetIdx()
            color = COLOR_MAP[tag]
            highlight_atoms_list.append(idx)
            highlight_atom_colors[idx] = color
            
    d2d = rdMolDraw2D.MolDraw2DSVG(600, 400)
    d2d.drawOptions().addAtomIndices = False
    d2d.drawOptions().bondLineWidth = 2
    
    try:
        Chem.SanitizeMol(mol)
        Chem.Compute2DCoords(mol)
        try: Chem.Kekulize(mol)
        except: pass
    except: pass

    # Sử dụng DrawMolecule và truyền LIST (để tránh lỗi Tuple)
    d2d.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms_list, 
        highlightAtomColors=highlight_atom_colors
    )
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

# ==========================================
# 3. STREAMLIT APP UI
# ==========================================

st.set_page_config(page_title="Chemical Block Builder", layout="wide")

st.title("🧩 Chemical Building Block Assembler")
st.markdown("Cấu trúc với vòng Benzen gắn nhóm thế được giữ nguyên màu gốc (không tô).")

# --- Sidebar Controls ---
st.sidebar.header("Configuration")

if 'random_trigger' not in st.session_state:
    st.session_state.random_trigger = False

def randomize():
    st.session_state.head_val = random.choice(list(CORES.keys()))
    st.session_state.tail_val = random.choice(list(TAIL_FRAGS.keys()))
    st.session_state.s3_val = random.choice(list(SUB_FRAGS.keys()))
    st.session_state.s4_val = random.choice(list(SUB_FRAGS.keys()))
    st.session_state.s5_val = random.choice(list(SUB_FRAGS.keys()))

st.sidebar.button("🎲 Random Structure", on_click=randomize)

head_sel = st.sidebar.selectbox("Head (Core)", list(CORES.keys()), key='head_val')
tail_sel = st.sidebar.selectbox("Tail", list(TAIL_FRAGS.keys()), key='tail_val')

st.sidebar.markdown("---")
st.sidebar.subheader("Substituents")
s3_sel = st.sidebar.selectbox("Pos 3", list(SUB_FRAGS.keys()), key='s3_val')
s4_sel = st.sidebar.selectbox("Pos 4", list(SUB_FRAGS.keys()), key='s4_val')
s5_sel = st.sidebar.selectbox("Pos 5", list(SUB_FRAGS.keys()), key='s5_val')

# --- Main Area ---
col1, col2 = st.columns([2, 1])

with col1:
    try:
        final_mol = build_molecule(head_sel, tail_sel, s3_sel, s4_sel, s5_sel)
        code_str = f"{head_sel}{tail_sel}3{s3_sel}4{s4_sel}5{s5_sel}"
        st.subheader(f"Code: `{code_str}`")
        svg = mol_to_image(final_mol)
        st.image(svg, use_container_width=False)
        
        can_smi = Chem.MolToSmiles(final_mol, isomericSmiles=True)
        with st.expander("Show Canonical SMILES"):
            st.code(can_smi)
    except Exception as e:
        st.error(f"Error: {e}")

with col2:
    st.markdown("### Legend")
    st.markdown("""
    <div style="margin-bottom:10px;">
        <span style="color:#1f77b4; font-weight:bold;">■ Core (Head)</span>: Phần khung dị vòng
    </div>
    <div style="margin-bottom:10px;">
        <span style="color:#800080; font-weight:bold;">■ Tail</span>: Nhóm đuôi tím
    </div>
    <div style="margin-bottom:10px;">
        <span style="color:#ff7f0e; font-weight:bold;">■ Substituents</span>: Nhóm thế cam
    </div>
    """, unsafe_allow_html=True)

