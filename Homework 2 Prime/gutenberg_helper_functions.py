import re
import string
CONTENTS_START_TAG = "\ncontents\r".lower()
CONTENTS_CENTER_START_TAG = "contents.\r\n\r"
CONTENTS_TILDE_START_TAG = "\n~contents~"
CONTENTS_OF_THE_BOOKS = "\n[Contents of the Books".lower()

def remove_gutenberg_header_contents_start_tags(text):
    
    start_tag_location = -1
    position = -1
    content_tag_len = -1

    
    '''
    1.0 Check the "CONTENTS" start tag 1st!
    '''
    start_tag_location = text.find(CONTENTS_START_TAG) 
    
    if start_tag_location > -1:
        position = start_tag_location
        content_tag_len = len(CONTENTS_START_TAG)
        updated_position = start_tag_location + content_tag_len + 1
    
    else:
        
        start_tag_location = text.find(CONTENTS_CENTER_START_TAG)
        
        if start_tag_location > -1:
            position = start_tag_location
        
            content_tag_len = len(CONTENTS_CENTER_START_TAG)
            updated_position = start_tag_location + content_tag_len + 1
            
            
        else:
            start_tag_location = text.find(CONTENTS_TILDE_START_TAG)
            
            if start_tag_location > -1:
                position = start_tag_location
                
                content_tag_len = len(CONTENTS_TILDE_START_TAG)
                updated_position = start_tag_location + content_tag_len + 1
            elif start_tag_location == -1:
                
                start_tag_location = text.find(CONTENTS_OF_THE_BOOKS)
                
                if start_tag_location > -1:
                    position = start_tag_location
                
                    content_tag_len = len(CONTENTS_OF_THE_BOOKS)
                    updated_position = start_tag_location + content_tag_len + 1
                   
    
    if position > -1:
        text = text[(updated_position):]
    
  
    return position, text




import re
import string
PREFACE_START_TAG = "\nPREFACE".lower()
PREFACE_TILDE_START_TAG = "\n~PREFACE~".lower()
PREFACE_CENTER_START_TAG = "preface.\r\n\r"

def remove_gutenberg_header_preface_start_tags(text):
    
    start_tag_location = -1
    position = -1
    content_tag_len = -1

    
    '''
    1.0 Check the "PREFACE" start tag 1st!
    '''
    start_tag_location = text.find(PREFACE_START_TAG) 
    
    if start_tag_location > -1:
        position = start_tag_location
        content_tag_len = len(PREFACE_START_TAG)
        updated_position = start_tag_location + content_tag_len + 1
    
    else:
        
        start_tag_location = text.find(PREFACE_CENTER_START_TAG)
        
        if start_tag_location > -1:
            position = start_tag_location
        
            content_tag_len = len(PREFACE_CENTER_START_TAG)
            updated_position = start_tag_location + content_tag_len + 1
            
        else:
            start_tag_location = text.find(PREFACE_TILDE_START_TAG)
            
            if start_tag_location > -1:
                position = start_tag_location
                
                content_tag_len = len(PREFACE_TILDE_START_TAG)
                updated_position = start_tag_location + content_tag_len + 1
                
    
    if position > -1:
        text = text[(updated_position):]
    
  
    return position, text



import re
import string

CHAPITRE_PREMIER_FRENCH_START_TAG = "\nCHAPITRE PREMIER.".lower()

def remove_gutenberg_header_chapitre_premier_french_start_tags(text):
    
    start_tag_location = -1
    position = -1
    content_tag_len = -1

    
    '''
    1.0 Check the "CHAPITRE PREMIER" start tag 1st!
    '''
    start_tag_location = text.find(CHAPITRE_PREMIER_FRENCH_START_TAG) 
    
    if start_tag_location > -1:
        position = start_tag_location
        content_tag_len = len(CHAPITRE_PREMIER_FRENCH_START_TAG)
        updated_position = start_tag_location + content_tag_len + 1
    
    if position > -1:
        text = text[(updated_position):]
    
  
    return position, text


import re
import string

INTRODUCTIONARY_START_TAG = "\r\nintroductory.\r\n\r".lower()

def remove_gutenberg_introductionary_start_tags(text):
    
    start_tag_location = -1
    position = -1
    content_tag_len = -1

    '''
    1.0 Check the "INTRODUCTORY" start tag 1st!
    '''
    start_tag_location = text.find(INTRODUCTIONARY_START_TAG) 
    
    if start_tag_location > -1:
        position = start_tag_location
        content_tag_len = len(INTRODUCTIONARY_START_TAG)
        updated_position = start_tag_location + content_tag_len + 1
    
    if position > -1:
        text = text[(updated_position):]
    
  
    return position, text


import re
import string

LE_CONSEIL_FRENCH_START_TAG = "\nLE CONSEIL".lower()

def remove_gutenberg_header_le_conseil_french_start_tags(text):
    
    start_tag_location = -1
    position = -1
    content_tag_len = -1

    
    '''
    1.0 Check the "LE CONSEIL" start tag 1st!
    '''
    start_tag_location = text.find(LE_CONSEIL_FRENCH_START_TAG) 
    
    if start_tag_location > -1:
        position = start_tag_location
        content_tag_len = len(LE_CONSEIL_FRENCH_START_TAG)
        updated_position = start_tag_location + content_tag_len + 1
    
    if position > -1:
        text = text[(updated_position):]
    
  
    return position, text