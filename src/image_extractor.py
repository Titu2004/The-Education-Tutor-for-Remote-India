from __future__ import annotations
import base64, io
from pypdf import PdfReader
from PIL import Image


def extract_images_from_pdf(pdf_bytes: bytes) -> list:
    reader  = PdfReader(io.BytesIO(pdf_bytes))
    results = []

    for page_num, page in enumerate(reader.pages, start=1):
        resources = page.get("/Resources")
        if not resources:
            continue
        xobject = resources.get("/XObject")
        if not xobject:
            continue

        for idx, name in enumerate(xobject):
            obj = xobject[name]
            if obj.get("/Subtype") != "/Image":
                continue
            try:
                data   = obj.get_data()
                width  = obj.get("/Width",  64)
                height = obj.get("/Height", 64)
                cs     = obj.get("/ColorSpace", "/DeviceRGB")
                mode   = "L" if cs == "/DeviceGray" else "CMYK" if cs == "/DeviceCMYK" else "RGB"

                img = Image.frombytes(mode, (width, height), data)
                if img.width < 40 or img.height < 40:
                    continue

                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                results.append({"page": page_num, "index": idx, "b64": b64})
            except Exception:
                continue

    return results


def find_relevant_images(images: list, relevant_page_nums: list, max_images: int = 2) -> list:
    matched = [img for img in images if img["page"] in relevant_page_nums]
    return matched[:max_images] if matched else []