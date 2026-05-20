import pypandoc
import os

def convert():
    try:
        print("Downloading pandoc locally if needed...")
        pypandoc.download_pandoc()
        print("Pandoc ready. Converting file...")
        output = pypandoc.convert_file("Project_Final_Report.md", "docx", outputfile="Project_Final_Report.docx")
        print("Successfully created Project_Final_Report.docx")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    convert()
