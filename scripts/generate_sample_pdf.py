# scripts/generate_sample_pdf.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_sample_pdf(filename="sample_corpus.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    
    # Página 1
    c.drawString(100, 750, "Política de Reembolsos y Devoluciones (Versión 1.0)")
    c.drawString(100, 720, "1. Los clientes tienen un plazo de 30 días para solicitar reembolsos.")
    c.drawString(100, 700, "2. El producto debe estar en su empaque original sin sellos rotos.")
    c.showPage()
    
    # Página 2
    c.drawString(100, 750, "Soporte Técnico y Mantenimiento")
    c.drawString(100, 720, "3. El horario de atención del equipo de TI es de Lunes a Viernes de 8 a 18 hs.")
    c.drawString(100, 700, "4. Para emergencias fuera de horario, contactar al canal #ops-guardia en Slack.")
    c.save()

if __name__ == "__main__":
    create_sample_pdf()
    print("PDF sintético generado en: sample_corpus.pdf")