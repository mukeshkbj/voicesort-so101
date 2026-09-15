"""Render deck/pitch.md to a simple landscape PDF slide deck."""
import re
from fpdf import FPDF

SRC = "deck/pitch.md"
OUT = "deck/pitch.pdf"


class Deck(FPDF):
    def slide(self, title, lines):
        self.add_page()
        self.set_fill_color(24, 28, 38)
        self.rect(0, 0, 297, 28, "F")
        self.set_text_color(255, 255, 255)
        self.set_font("helvetica", "B", 20)
        self.set_xy(12, 9)
        self.cell(0, 10, title)
        self.set_text_color(30, 30, 30)
        self.set_xy(12, 40)
        self.set_font("helvetica", "", 13)
        for ln in lines:
            if ln.startswith("- "):
                self.set_x(18)
                self.multi_cell(265, 8, "- " + ln[2:])
            elif ln:
                self.multi_cell(265, 8, ln)
            else:
                self.ln(4)


def main():
    txt = open(SRC, encoding="utf-8").read()
    pdf = Deck(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(False)

    # title slide: first '# ' line + first paragraph
    title = re.search(r"^# (.+)$", txt, re.M).group(1)
    pdf.slide("VoiceSort", [title.split(":")[-1].strip(),
                            "",
                            "AI Infra Summit Hackathon - Intel Online track"])

    for m in re.finditer(r"## Slide \d+: (.+?)\n(.*?)(?=\n## Slide|\Z)",
                         txt, re.S):
        lines = [l.rstrip() for l in m.group(2).strip().splitlines()]
        pdf.slide(m.group(1), lines)

    pdf.output(OUT)
    print(f"{OUT}: {pdf.page_no()} slides")


if __name__ == "__main__":
    main()
