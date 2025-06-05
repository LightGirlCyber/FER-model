import os
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import hashlib

def download_images_selenium(search_term, num_images=1000):
    # Setup Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    driver = webdriver.Chrome(options=chrome_options)
    
    # Create folder
    folder_name = search_term.replace(' ', '_')
    os.makedirs(f"duck_face_dataset/{folder_name}", exist_ok=True)
    
    try:
        # Go to Google Images
        driver.get(f"https://www.google.com/search?q={search_term}&tbm=isch")
        
        # Scroll to load images
        for i in range(5):  # Scroll 5 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Get image elements
        images = driver.find_elements(By.CSS_SELECTOR, "img[src]")
        
        downloaded = 0
        for i, img in enumerate(images[:num_images]):
            try:
                img_url = img.get_attribute("src")
                if img_url and img_url.startswith("http"):
                    # Download image
                    response = requests.get(img_url, timeout=10)
                    if response.status_code == 200:
                        filename = f"{folder_name}_{i+1:03d}.jpg"
                        filepath = f"duck_face_dataset/{folder_name}/{filename}"
                        
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded += 1
                        print(f"Downloaded: {filename}")
                        
            except Exception as e:
                continue
                
        print(f"Downloaded {downloaded} images for '{search_term}'")
        
    finally:
        driver.quit()

# Usage
search_terms = ["duck face selfie", "kissy face expression", "pouty lips selfie"]

for term in search_terms:
    download_images_selenium(term, 100)
    time.sleep(3)