'''
Web Scraping
============

Source: `Web Scraping With Python – Step-By-Step Guide <https://brightdata.com/blog/how-tos/web-scraping-with-python>`_

Prerequisites

**Requests** library allows you to perform HTTP requests in Python, Install:

::

    conda install requests

**Beautiful Soup** library makes scraping information
from web pages easier. In particular, Beautiful Soup works with any HTML or XML 
parser and provides everything you need to iterate, search, and modify the parse tree.

::

    conda install beautifulsoup4

We will scrape `quotes <https://quotes.toscrape.com>`_ 
Each quote is a bloc of <div> </div>, that look like:

::

    <div class="quote">
        <span class="text" itemprop="text">“The world as ...”</span>
        <span>by <small class="author" itemprop="author">Albert Einstein</small>
        </span>
        <div class="tags">
            Tags:
            <a class="tag" href="/tag/change/page/1/">change</a>  
            <a class="tag" href="/tag/deep-thoughts/page/1/">deep-thoughts</a>
        </div>
    </div>
    <div class="quote">
        <span class="text" itemprop="text">“It is our choices, ...”</span>
        <span>by <small class="author" itemprop="author">J.K. Rowling</small>
        </span>
        <div class="tags">
            Tags:
            <a class="tag" href="/tag/abilities/page/1/">abilities</a>
            <a class="tag" href="/tag/choices/page/1/">choices</a>
        </div>
    </div>
    ...
       
And will return CSV file:

::

                   Text              Author                        Tags
   0   “The world as...     Albert Einstein       change, deep-thoughts
   1   “It is our ch...        J.K. Rowling       abilities, choices
   2   “There are on...     Albert Einstein       inspirational, life

'''

import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_page(soup):

    quotes = list()
    
    # retrieving all the quote <div> HTML element on the page
    # Right / view page source to understand the structure of the HTML page
    # The <div> tag defines a division or a section in an HTML document.
    quote_elements = soup.find_all('div', class_='quote')

    # iterating over the list of quote elements
    # to extract the data of interest and store it
    # in quotes
    for quote_element in quote_elements:
        # DEBUG: quote_element = quote_elements[0]
        # extracting the text of the quote
        # The quote text in a <span> HTML element: <span class="text"> ...
        # The author of the quote in a <small> HTML element
        # A list of tags in a <div> element, each contained in <a> HTML element

        text = quote_element.find('span', class_='text').text
        # extracting the author of the quote
        author = quote_element.find('small', class_='author').text

        # extracting the tag <a> HTML elements related to the quote
        tag_elements = \
            quote_element.find('div', class_='tags').find_all('a', class_='tag')

        # storing the list of tag strings in a list
        tags = [tag_element.text for tag_element in tag_elements]

        # appending a dictionary containing the quote data
        # in a new format in the quote list
        quotes.append([text, author, ', '.join(tags)])

    return quotes

#############################################################
# Url of the home page of the target website

base_url = 'https://quotes.toscrape.com'

# defining the User-Agent header to use in the GET request below
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

#############################################################
# Retrieving the target web page.
# `page.text` will contain the HTML document returned by the server in string format

page = requests.get(base_url, headers=headers)


#############################################################
# Parsing the target web page with Beautiful Soup

soup = BeautifulSoup(page.text, 'html.parser')

#############################################################
# scraping the home page

quotes = scrape_page(soup)


#############################################################
# getting the "Next" HTML element

next_li_element = soup.find('li', class_='next')

# if there is a next page to scrape
while next_li_element is not None:
    next_page_relative_url = next_li_element.find('a', href=True)['href']

    # getting the new page
    page = requests.get(base_url + next_page_relative_url, headers=headers)

    # parsing the new page
    soup = BeautifulSoup(page.text, 'html.parser')

    # scraping the new page and append to the quotes
    quotes += scrape_page(soup)

    # looking for the "Next →" HTML element in the new page
    next_li_element = soup.find('li', class_='next')


# Write csv file
df = pd.DataFrame(quotes,
                  columns=['Text', 'Author', 'Tags'])

df.to_csv('quotes.csv', index=False)

