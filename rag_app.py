import pymupdf4llm
import pathlib

pdf_path = "data/herman-melville-moby-dick.pdf"

markdown_text = pymupdf4llm.to_markdown(pdf_path)

output_path = "data/moby-dick-output.md"
pathlib.Path(output_path).write_text(markdown_text, encoding="utf-8")
