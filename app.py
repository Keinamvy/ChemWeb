import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Chemical Block Configurator", layout="wide")

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
    "k": ("[*]CF", "-CH2F (Lỗi trong gốc? Giả định -CF)"), # Lưu ý: [*]CF có thể không hợp lệ hóa trị nếu không rõ ràng, giữ nguyên như input
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
    core = Chem.Mol(core) # Copy để không sửa core gốc
    
    # Tìm dummy trên core
    idx_dummy = None
    for a in core.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx_dummy = a.GetIdx(); break
    
    if idx_dummy is None:
        # Nếu không tìm thấy label (có thể đã bị xóa hoặc lỗi), trả về core hiện tại
        return core

    # Nút trên core để nối
    nbrs = [n.GetIdx() for n in core.GetAtomWithIdx(idx_dummy).GetNeighbors()]
    if len(nbrs) != 1: return core # Lỗi cấu trúc
    idx_core_attach = nbrs[0]

    # Xử lý Fragment
    # Lưu ý: frag_smiles trong dict của app này là tuple (smiles, desc), lấy smiles
    real_smiles = frag_smiles if isinstance(frag_smiles, str) else frag_smiles
    
    frag = Chem.MolFromSmiles(real_smiles)
    if frag is None: return core
    
    # Tìm dummy trong fragment
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
    
    # Xóa 2 dummy (xóa index lớn trước)
    for ridx in sorted([cn + idx_fd, idx_dummy], reverse=True):
        em.RemoveAtom(ridx)
    
    m = em.GetMol()
    try:
        Chem.SanitizeMol(m)
    except:
        pass # Bỏ qua lỗi sanitize tạm thời để hiển thị
    return m

def remove_dummy_label(mol, label):
    """Xóa dummy atom có label cụ thể (dùng cho trường hợp Hydro 'a')"""
    idx = None
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 0 and a.HasProp("molAtomMapNumber") and int(a.GetProp("molAtomMapNumber")) == label:
            idx = a.GetIdx(); break
    
    if idx is not None:
        em = Chem.EditableMol(mol)
        em.RemoveAtom(idx)
        m = em.GetMol()
        try:
            Chem.SanitizeMol(m)
        except:
            pass
        return m
    return mol

# --- GIAO DIỆN NGƯỜI DÙNG (SIDEBAR) ---

st.sidebar.header("⚙️ Cấu hình Building Blocks")

# 1. Chọn Core (Head)
selected_head = st.sidebar.selectbox("1. Chọn Head (Lõi)", list(CORES.keys()), index=2, format_func=lambda x: f"Head {x}")

# 2. Chọn Tail (Block III)
tail_options = list(TAIL_FRAGS.keys())
selected_tail = st.sidebar.selectbox(
    "2. Chọn Tail (Nhóm thế N)", 
    tail_options, 
    index=tail_options.index("2h"),
    format_func=lambda x: f"{x}: {TAIL_FRAGS[x][1]}"
)

# 3. Chọn Substituents (Block II)
sub_options = list(SUB_FRAGS.keys())

col_sb1, col_sb2, col_sb3 = st.sidebar.columns(3)
with col_sb1:
    s3 = st.selectbox("Vị trí 3", sub_options, index=sub_options.index("i"), format_func=lambda x: x)
with col_sb2:
    s4 = st.selectbox("Vị trí 4", sub_options, index=sub_options.index("a"), format_func=lambda x: x)
with col_sb3:
    s5 = st.selectbox("Vị trí 5", sub_options, index=sub_options.index("j"), format_func=lambda x: x)

# Tạo mã code tổng hợp
generated_code = f"{selected_head}{selected_tail}3{s3}4{s4}5{s5}"

# --- XỬ LÝ LẮP RÁP PHÂN TỬ ---

# B1: Lấy Core
core_mol = Chem.MolFromSmiles(CORES[selected_head])

# B2: Gắn Tail vào vị trí 9
tail_smiles = TAIL_FRAGS[selected_tail][0]
current_mol = attach_fragment(core_mol, 9, tail_smiles)

# B3: Gắn Substituents vào 3, 4, 5
subs_to_attach = [(3, s3), (4, s4), (5, s5)]

for label, code in subs_to_attach:
    if code == "a":
        # Nếu là 'a' (Hydro), ta chỉ cần xóa dummy placeholder đi
        current_mol = remove_dummy_label(current_mol, label)
    else:
        # Nếu là nhóm thế khác, gắn vào
        sub_smiles = SUB_FRAGS[code][0]
        current_mol = attach_fragment(current_mol, label, sub_smiles)

# Clean up và tạo tọa độ 2D đẹp
try:
    Chem.SanitizeMol(current_mol)
    AllChem.Compute2DCoords(current_mol)
    final_smiles = Chem.MolToSmiles(current_mol, isomericSmiles=True)
except Exception as e:
    st.error(f"Lỗi khi tạo cấu trúc: {e}")
    final_smiles = ""

# --- HIỂN THỊ KẾT QUẢ ---

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Kết quả cấu trúc")
    if current_mol:
        # Vẽ hình
        img = Draw.MolToImage(current_mol, size=(600, 400))
        st.image(img, caption=f"Cấu trúc cho mã: {generated_code}", use_column_width=True)
    
with col2:
    st.subheader("Thông tin chi tiết")
    st.info(f"**Mã định danh:** `{generated_code}`")
    
    st.markdown("---")
    st.write("**Thành phần:**")
    st.write(f"- **Head:** {selected_head}")
    st.write(f"- **Tail:** {TAIL_FRAGS[selected_tail][1]} ({selected_tail})")
    st.write(f"- **Sub 3:** {SUB_FRAGS[s3][1]} ({s3})")
    st.write(f"- **Sub 4:** {SUB_FRAGS[s4][1]} ({s4})")
    st.write(f"- **Sub 5:** {SUB_FRAGS[s5][1]} ({s5})")
    
    st.markdown("---")
    st.text_area("SMILES", value=final_smiles, height=100)

# --- KIỂM TRA QUY TẮC (OPTIONAL) ---
non_a_count = sum([1 for x in [s3, s4, s5] if x != "a"])
if non_a_count > 2:
    st.warning(f"⚠️ **Lưu ý:** Cấu hình này có {non_a_count} nhóm thế khác Hydro. Quy tắc thư viện ban đầu (Max2) chỉ cho phép tối đa 2 nhóm thế.")
else:
    st.success("✅ Cấu hình thỏa mãn quy tắc Max-2.")
