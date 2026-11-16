from pathlib import Path

from pypdf import PdfReader

root = Path(
    r"c:\Users\kaurp\.cursor\projects\c-Users-kaurp-OneDrive-Documents-evalkit"
    r"\attachments\85269643-a7b7-4bdd-854e-88cedbb91efe"
)
out = root / "_pdf_images"
out.mkdir(exist_ok=True)
for name in ["Instructions_to_Train_AI.pdf", "Rubric_criteria.pdf"]:
    reader = PdfReader(str(root / name))
    print("====", name, "pages", len(reader.pages), "====")
    for pi, page in enumerate(reader.pages, 1):
        box = page.mediabox
        print(f" page {pi} mediabox {float(box.width):.0f}x{float(box.height):.0f}")
        res = page["/Resources"].get_object()
        if "/XObject" not in res:
            print("  no images")
            continue
        xo = res["/XObject"].get_object()
        for key, value in xo.items():
            obj = value.get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            width, height = obj.get("/Width"), obj.get("/Height")
            data = obj.get_data()
            label = str(key).replace("/", "")
            dest = out / f"{name[:6]}_p{pi}_{label}_{width}x{height}.jpg"
            dest.write_bytes(data)
            print(f"  {key} {width}x{height} bytes={len(data)} -> {dest.name}")
