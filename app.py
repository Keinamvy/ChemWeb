import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Chemical Block Configurator", layout="wide")
# Ẩn menu hamburger và footer mặc định của Streamlit để giao diện sạch hơn
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("🧪 Chemical Building Block Configurator")
st.markdown("Công cụ trực quan hóa cấu trúc phân tử dựa trên các mảnh ghép (Fragments).")

# --- DỮ LIỆU ĐẦU VÀO (TỪ CODE GỐC) ---

# Block III tails (gắn tại [*:9])
TAIL_FRAGS = {
    "1a": ("[*:9]CC=C", "Allyl"),
    "1b": ("[*:9]CCCO", "3-hydroxypropyl"),
    "1c": ("[*:9]C=CC", "Propenyl"),
    "1d": ("[*:9]C(O)CC", "2-hydroxypropyl"),
    "2a": ("[*:9]Cc1ccccc1", "Benzyl"),
    "2b": ("[*:9]Cc1c(C)cccc1", "2-methyl"),
    "2c": ("[*:9]Cc1cc(C)ccc1", "3-methyl"),
    "2d": ("[*:9]Cc1ccc(C)cc1", "4-methyl"),
    "2e": ("[*:9]Cc1c(OC)cccc1", "2-methoxy"),
    "2f": ("[*:9]Cc1cc(OC)ccc1", "3-methoxy"),
    "2g": ("[*:9]Cc1ccc(OC)cc1", "4-methoxy"),
    "2h": ("[*:9]Cc1c(N(C)C)cccc1", "2-amine (NMe2)"),
    "2i": ("[*:9]Cc1cc(N(C)C)ccc1", "3-amine"),
    "2j": ("[*:9]Cc1ccc(N(C)C)cc1", "4-amine"),
}

# Block II substituents (3/4/5)
SUB_FRAGS = {
    "a": ("[*]H", "Hydrogen (H)"),
    "b": ("[*]O", "Hydroxy (-OH)"),
    "c": ("[*]S", "Thiol (-SH)"),
    "d": ("[*]OC", "Methoxy (-OMe)"),
    "e": ("[*]OC(F)(F)F", "-OCF3"),
    "f": ("[*]SC", "-SMe"),
    "g": ("[*]N(C)C", "-NMe2"),
    "h": ("[*]C=O", "Formyl (-CHO)"),
    "i": ("[*]C(C)=O", "Acetyl (-Ac)"),
    "j": ("[*]F", "Fluoro (-F)"),
    "k": ("[*]CF", "-CH2F (Giả định)"), 
    "l": ("[*]C(F)F", "-CHF2"),
    "m": ("[*]C(F)(F)F", "-CF3"),
    "n": ("[*]C", "Methyl (-Me)"),
}

# Core cho O/S/N/M
CORES = {
    "O": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)O2)c1",
    "S": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)S2)c1",
    "N": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)N2)c1",
    "M": "[*:9]c1ccc2c(C=C(c3cc([*:3])c([*:4])c([*:5])c3)N2C)c1",
}

# --- HÀM XỬ LÝ RDKIT ---

def attach_fragment(core: Chem.Mol, label: int, frag_smiles: str) -> Chem.Mol:
    """Gắn 1 fragment vào nhãn [*:label] trên core."""
    core = Chem.Mol(core) 
    idx_dummy = None
    for a in core.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx_dummy = a.GetIdx(); break
    
    if idx_dummy is None: return core

    nbrs = [n.GetIdx() for n in core.GetAtomWithIdx(idx_dummy).GetNeighbors()]
    if len(nbrs) != 1: return core
    idx_core_attach = nbrs[0]

    real_smiles = frag_smiles if isinstance(frag_smiles, str) else frag_smiles
    frag = Chem.MolFromSmiles(real_smiles)
    if frag is None: return core
    
    idx_fd_list = [a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0]
    if not idx_fd_list: return core
    idx_fd = idx_fd_list[0]
    
    fnbr = [n.GetIdx() for n in frag.GetAtomWithIdx(idx_fd).GetNeighbors()]
    if len(fnbr) != 1: return core
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
    """Xóa dummy atom có label cụ thể (dùng cho Hydro 'a')"""
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

# --- GIAO DIỆN NGƯỜI DÙNG (SIDEBAR) ---

st.sidebar.header("⚙️ Cấu hình Building Blocks")

# 1. Chọn Core (Head) - Màu Xanh
selected_head = st.sidebar.selectbox("1. Chọn Head (Lõi) [N]", list(CORES.keys()), index=2, format_func=lambda x: f"Head {x}")

# 2. Chọn Tail (Block III) - Màu Tím
tail_options = list(TAIL_FRAGS.keys())
selected_tail = st.sidebar.selectbox(
    "2. Chọn Tail (Nhóm thế) [2h...]", 
    tail_options, 
    index=tail_options.index("2h"),
    format_func=lambda x: f"{x}: {TAIL_FRAGS[x][1]}"
)

# 3. Chọn Substituents (Block II) - Màu Cam
st.sidebar.markdown("---")
st.sidebar.write("3. Chọn Substituents (Nhóm thế vòng) [3i4a5j]")
sub_options = list(SUB_FRAGS.keys())

col_sb1, col_sb2, col_sb3 = st.sidebar.columns(3)
with col_sb1:
    s3 = st.selectbox("Vị trí 3", sub_options, index=sub_options.index("i"), format_func=lambda x: x, key="s3")
with col_sb2:
    s4 = st.selectbox("Vị trí 4", sub_options, index=sub_options.index("a"), format_func=lambda x: x, key="s4")
with col_sb3:
    s5 = st.selectbox("Vị trí 5", sub_options, index=sub_options.index("j"), format_func=lambda x: x, key="s5")

# --- XỬ LÝ LẮP RÁP PHÂN TỬ ---

core_mol = Chem.MolFromSmiles(CORES[selected_head])
tail_smiles = TAIL_FRAGS[selected_tail][0]
current_mol = attach_fragment(core_mol, 9, tail_smiles)

subs_to_attach = [(3, s3), (4, s4), (5, s5)]
for label, code in subs_to_attach:
    if code == "a":
        current_mol = remove_dummy_label(current_mol, label)
    else:
        sub_smiles = SUB_FRAGS[code][0]
        current_mol = attach_fragment(current_mol, label, sub_smiles)

try:
    Chem.SanitizeMol(current_mol)
    AllChem.Compute2DCoords(current_mol)
    final_smiles = Chem.MolToSmiles(current_mol, isomericSmiles=True)
except Exception as e:
    st.error(f"Lỗi khi tạo cấu trúc: {e}")
    final_smiles = ""
    current_mol = None

# --- HIỂN THỊ KẾT QUẢ ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Kết quả cấu trúc")
    if current_mol:
        img = Draw.MolToImage(current_mol, size=(600, 450))
        st.image(img, use_column_width=True)
    else:
        st.warning("Không thể tạo hình ảnh cấu trúc.")
    
with col2:
    st.subheader("Thông tin chi tiết")
    
    # --- PHẦN MỚI: TẠO MÃ MÀU HIGHLIGHT ---
    # Định nghĩa màu sắc (tương tự hình ảnh)
    head_color = "#1f77b4" # Xanh dương
    tail_color = "#9467bd" # Tím
    sub_color = "#ff7f0e"  # Cam
    # Màu nền nhạt tương ứng để làm highlight
    head_bg = "#e6f2ff"
    tail_bg = "#f2e6ff"
    sub_bg = "#fff2e6"
    
    # Style chung cho các thẻ span
    span_style = "padding: 4px 6px; border-radius: 6px; font-weight: 600; font-size: 1.3em;"

    # Tạo chuỗi HTML với highlight
    highlighted_code = (
        f'<span style="background-color: {head_bg}; color: {head_color}; {span_style}">{selected_head}</span>'
        f'<span style="background-color: {tail_bg}; color: {tail_color}; {span_style} margin-left: 4px;">{selected_tail}</span>'
        f'<span style="background-color: {sub_bg}; color: {sub_color}; {span_style} margin-left: 4px;">3{s3}4{s4}5{s5}</span>'
    )

    # Hiển thị bằng st.markdown với unsafe_allow_html=True
    st.markdown(f"### Mã định danh:<br><div style='margin-top: 10px;'>{highlighted_code}</div>", unsafe_allow_html=True)
    # ---------------------------------------

    st.markdown("---")
    st.write("**Thành phần:**")
    # Sử dụng màu sắc tương ứng trong danh sách thành phần để đồng bộ
    st.markdown(f"- **Head:** <span style='color:{head_color}; font-weight:bold;'>{selected_head}</span>", unsafe_allow_html=True)
    st.markdown(f"- **Tail:** <span style='color:{tail_color}; font-weight:bold;'>{selected_tail}</span> ({TAIL_FRAGS[selected_tail][1]})", unsafe_allow_html=True)
    st.markdown(f"- **Sub 3:** <span style='color:{sub_color}; font-weight:bold;'>{s3}</span> ({SUB_FRAGS[s3][1]})", unsafe_allow_html=True)
    st.markdown(f"- **Sub 4:** <span style='color:{sub_color}; font-weight:bold;'>{s4}</span> ({SUB_FRAGS[s4][1]})", unsafe_allow_html=True)
    st.markdown(f"- **Sub 5:** <span style='color:{sub_color}; font-weight:bold;'>{s5}</span> ({SUB_FRAGS[s5][1]})", unsafe_allow_html=True)
    
    st.markdown("---")
    with st.expander("Xem SMILES"):
        st.code(final_smiles, language="text")

# --- KIỂM TRA QUY TẮC ---
non_a_count = sum([1 for x in [s3, s4, s5] if x != "a"])
if non_a_count > 2:
    st.warning(f"⚠️ **Lưu ý:** Cấu hình này có {non_a_count} nhóm thế khác Hydro. Quy tắc gốc (Max2) chỉ cho phép tối đa 2.")
