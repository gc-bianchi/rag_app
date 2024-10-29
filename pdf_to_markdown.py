from pdfminer.high_level import extract_text

pdf_path = "data/herman-melville-moby-dick.pdf"
text = extract_text(pdf_path)

output_path = "data/moby-dick-output2.md"
with open(output_path, "w", encoding="utf-8") as file:
    file.write(text)
