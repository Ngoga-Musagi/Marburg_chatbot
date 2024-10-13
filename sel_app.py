from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Setup the WebDriver (Chrome in this case)
driver = webdriver.Chrome(executable_path='/path/to/your/chromedriver')  # Replace with the path to your chromedriver

# Open the X page for Rwanda Ministry of Health
driver.get('https://x.com/RwandaHealth')

# Give some time for the page to load completely
time.sleep(5)

# Scroll down to load more content (you can adjust the number of scrolls)
scroll_pause_time = 3
for i in range(3):  # Adjust the range to scroll more if needed
    # Scroll down to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    # Wait for the new content to load
    time.sleep(scroll_pause_time)

# Extract tweets based on the 'article' tag used by Twitter for each tweet block
tweets = driver.find_elements(By.TAG_NAME, 'article')

# Process and print each tweet content
for tweet in tweets:
    try:
        tweet_content = tweet.text  # Get the text from the tweet
        print(f"Tweet:\n{tweet_content}\n{'-'*80}\n")
    except Exception as e:
        print(f"Error: {e}")

# Close the browser after scraping
driver.quit()
