import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
import nltk
from bs4 import BeautifulSoup
from treelib import Tree
import re
import hashlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
from utils import *

class FacultyPagesFilteredSpider(scrapy.Spider):
    name = 'faculty_pages_filtered'
    allowed_domains = ['stanford.edu']
                      # ['cmu.edu']
                      # ['cornell.edu', 'washington.edu',
                      # 'gatech.edu', 'princeton.edu', 'utexas.edu',
                      #, 'berkeley.edu','illinois.edu']
                      #, 'mit.edu', 'stanford.edu'
    count = 0
    record = {}
    start_urls = ['https://www.stanford.edu/'] 
                  #[https://www.cmu.edu/']
                  #['https://www.cornell.edu/',
                  #'https://www.washington.edu/', 'https://www.gatech.edu/',
                  #'https://www.princeton.edu/', 'https://www.utexas.edu/',
                  #'https://illinois.edu/', 'https://www.berkeley.edu/']
                  # 'https://www.mit.edu/', 'https://www.stanford.edu/'

    exclude_words1 = []
    exclude_words = ['news', 'events', 'publications', 'pub', 'gallery', 
                     'category', 'courses', 'students', 'references', 
                     'reference', 'software', 'softwares', 'tags', 
                     'tutorials', 'workshop', 'festival', 'admissions', 
                     'exhibitions', 'alumni', 'lectures', 'undergraduate', 
                     'about', 'history', 'awards', 'ranking', 'enrollment', 
                     'graduate', 'archive', 'stories', 'post', 'pages', 
                     'magazine', 'curriculum', '404', 'faqs', 'engage', 
                     'campaign', 'career', 'resources', 'services', 
                     'network', 'security', 'donate', 'giving', 'finance', 
                     'forms', 'policies', 'policy', 'alphabetical', 'summer', 
                     'winter', 'spring', 'autumn', 'fall', 'health', 'facilities', 
                     'facility', 'wp', 'information', 'general', 'catalog', 
                     'guides', 'library', 'publish', 'blog', 'collection', 
                     'share', 'search', 'periodicals', 'bookstore', 'store', 
                     'product', 'organisation', 'webstore', 'funding', 'pdf']


    rules = [Rule(LinkExtractor(unique=True), callback='parse', follow=True)]
    #count_limits = {"page_count": 200, "item_count": 200}

    def __init__(self):
        
        self.tree = Tree()
        self.tree.create_node("root", "root")
        self.tree.create_node("unknown", "unknown", parent="root")
        
        self.queue = []
        
        self.bio_identifier = BioIdentifier(model="bio-model")

        for dom in self.allowed_domains:
            domain = dom.split('.')[0]
            if not os.path.exists('filtered_data'):
                os.makedirs('filtered_data')

            folder_name = 'filtered_data/'+domain + '_files'
            self.record[domain] = 0
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

    def parse(self, response):
        
        domain = self.allowed_domains[0].split('.')[-2]
        folder_name = 'filtered_data/'+domain + '_files'

        self.record[domain] = self.record.get(domain, 0) + 1
        
        # print("domain : ", self.allowed_domains[0])
        # print("link : ", response.url)

        if self.record[domain]%1000 == 0:
            print('\n','-'*40, self.record[domain])
            self.tree.save2file(folder_name+"/00__"+str(self.record[domain])+"_tree.txt")

        #text = BeautifulSoup(response.xpath('//*').get(), features="lxml").get_text()
        
        
        # Pass normalized_text to classifier to check whether bio_page
        # to uncomment select lines 94 to 100 and shift+ctrl+k
        
        #if self.bio_identifier.is_bio_text(normalized_text):
        #if self.bio_identifier.is_bio_html(response.xpath('//*').get()):
        #print('*'*40, ' Before Classifier ', '*'*40)
        isBio = self.bio_identifier.is_bio_html_content(response.xpath('//*').get())
        #print('*'*40, isBio, '*'*40)
        #print('*'*40, ' After Classifier ', '*'*40)
        
        if isBio:
            text = BeautifulSoup(response.xpath('//*').get(), features="lxml").get_text()
            tokens = nltk.word_tokenize(text)
            normalized_text = ' '.join([word for word in tokens if word.isalnum()])
            normalized_text += '\n'+response.url
            
            hash_text = hashlib.md5(response.url.encode()) 
            file_name = hash_text.hexdigest()

            with open(folder_name+"/"+file_name+".txt", "w") as file:
                file.write(normalized_text)
                
        # save text file
        
        #hash_text = hashlib.md5(response.url.encode()) 
        
        #file_name = hash_text.hexdigest()

        #with open(folder_name+"/"+file_name+".txt", "w") as file:
            #file.write(normalized_text)
        
        # continue checking for other links
        
        AllLinks = LinkExtractor(allow_domains = self.allowed_domains[0], unique=True).extract_links(response)

        for n, link in enumerate(AllLinks):
            if not any([x in link.url for x in self.exclude_words1]):
                if self.tree.get_node(link.url) == None:
                    referer = response.request.headers.get('Referer', None)

                    if referer == None:
                        self.tree.create_node(link.url, link.url, parent='root')
                    else:
                        referer = referer.decode("utf-8")
                        if self.tree.contains(referer):

                            self.tree.create_node(link.url, link.url, parent=referer)
                        else:
                            self.tree.create_node(link.url, link.url, parent='unknown')
                    
                    #print("n:",n," link:",link.url)
                    self.queue.append(link.url)
                    #print(self.queue)
                    yield scrapy.Request(url=link.url, callback = self.parse)

