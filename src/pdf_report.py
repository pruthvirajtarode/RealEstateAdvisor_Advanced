# src/pdf_report.py
from fpdf import FPDF

def create_property_pdf(input_dict, verdict, confidence, price_5y, out_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 8, "Real Estate Investment Advisor - Report", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 6, f"Verdict: {verdict}", ln=True)
    if confidence is not None:
        pdf.cell(0, 6, f"Confidence: {confidence:.2f}", ln=True)
    pdf.cell(0, 6, f"Estimated Price (5y): {price_5y:.2f} Lakhs", ln=True)
    pdf.ln(6)
    pdf.cell(0,6,"Inputs:", ln=True)
    for k,v in input_dict.items():
        pdf.multi_cell(0,6,f"- {k}: {v}")
    pdf.output(out_path)
    return out_path
