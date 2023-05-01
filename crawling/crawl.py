import requests
import json
import os

API_KEY = "API_KEY"                                 # API_KEY
SEARCH_ENGINE_ID = "SEARCH_ENGINE_ID"               # SEARCH_ENGINE_ID
NUM_IMAGES = 50                                     # NUMBER OF IMAGES TO DOWNLOAD(MULTIPLE OF 10)
START_LOCATION = 50                                 # NUMBER OF IMAGES ALREADY DOWNLOADED
headers = requests.utils.default_headers()
headers.update(
    {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
    }
)

def search_images(query, num_results=10, start=1):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "num": num_results,
        "start": start,
        "imgSize": "large",
        "searchType": "image",
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID
    }
    
    response = requests.get(url, params=params)
    response.raise_for_status()
    results = json.loads(response.text)

    images = []
    for item in results["items"]:
        images.append({
            "title": item["title"],
            "link": item["link"],
            "thumbnail": item["image"]["thumbnailLink"]
        })

    return images

def download_image(url, filename):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

if __name__ == "__main__":
    if not os.path.exists("images"):
        os.makedirs("images")

    query = "무신사 로고"
    
    for j in range(1, NUM_IMAGES, 10):
        images = search_images(query, 10, START_LOCATION+j)
        for i, image in enumerate(images):
            print(f"Downloading image {START_LOCATION+j+i}: {image['title']}")
            filename = os.path.join("images", f"image_{START_LOCATION+j+i}.jpg")
            download_image(image["link"], filename)
    
    print(f"{NUM_IMAGES} images downloaded")