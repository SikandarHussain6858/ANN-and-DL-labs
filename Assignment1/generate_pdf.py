import base64
import time
import os
from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.print_page_options import PrintOptions

def convert_html_to_pdf(html_path, output_pdf_path):
    abs_html = os.path.abspath(html_path)
    file_url = f"file:///{abs_html.replace(os.sep, '/')}"
    
    print(f"Loading URL: {file_url}")
    
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--allow-file-access-from-files')
    
    driver = webdriver.Edge(options=options)
    
    try:
        driver.get(file_url)
        
        # Wait for MathJax rendering to complete
        print("Waiting for MathJax to finish typesetting...")
        for _ in range(30):
            ready = driver.execute_script("return (typeof MathJax !== 'undefined' && MathJax.startup && MathJax.startup.document && MathJax.startup.document.state() >= 100);")
            if ready:
                print("MathJax finished rendering!")
                break
            time.sleep(0.5)
            
        time.sleep(2) # Extra buffer for layout stability
        
        print_options = PrintOptions()
        print_options.background = True  # Print background colors and images
        
        print("Printing page to PDF...")
        pdf_b64 = driver.print_page(print_options)
        
        with open(output_pdf_path, 'wb') as f:
            f.write(base64.b64decode(pdf_b64))
            
        print(f"Successfully generated: {output_pdf_path} (Size: {os.path.getsize(output_pdf_path)} bytes)")
    finally:
        driver.quit()

if __name__ == '__main__':
    convert_html_to_pdf("report.html", "CS320_F2026_Assign1_502808.pdf")
